#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "microtensor/tensor.hpp"

namespace tensors {

template <typename Dest, typename... Src>
class TensorIterator {
 private:
  static constexpr std::size_t n_terms = sizeof...(Src) + 1;

  std::tuple<Dest*, Src*...> data_;
  std::vector<std::size_t> shapes_;
  std::array<std::vector<std::size_t>, n_terms> strides_;
  std::vector<std::size_t> idx_;
  std::array<std::size_t, n_terms> offsets_{};
  bool has_next_flag_;

  template <std::size_t... Is>
  auto dereference(std::index_sequence<Is...>);

 public:
  template <typename... TensorTypes>
  TensorIterator(Tensor& dest, const TensorTypes&... srcs);

  template <typename... TensorTypes>
  TensorIterator(const Tensor& dest, const TensorTypes&... srcs);

  bool has_next() const;
  auto next();

  void for_each(std::invocable<Dest&, Src&...> auto fn);
};

template <typename Dest, typename... Src>
template <std::size_t... Is>
auto TensorIterator<Dest, Src...>::dereference(std::index_sequence<Is...>) {
  return std::tuple<Dest&, Src&...>{std::get<Is>(data_)[offsets_[Is]]...};
}

template <typename Dest, typename... Src>
template <typename... TensorTypes>
TensorIterator<Dest, Src...>::TensorIterator(Tensor& dest,
                                             const TensorTypes&... srcs)
    : data_(std::make_tuple(dest.data(), srcs.data()...)),
      shapes_(dest.shape()),
      strides_({dest.stride(), srcs.stride()...}),
      idx_(dest.ndim(), 0),
      offsets_{},
      has_next_flag_(!dest.empty()) {
  auto validate = [&dest](const auto& tensor) {
    if (tensor.shape() != dest.shape() ||
        tensor.stride().size() != dest.ndim()) {
      throw std::invalid_argument(
          "TensorIterator operands must match the destination shape and rank");
    }
  };
  if (dest.stride().size() != dest.ndim()) {
    throw std::invalid_argument(
        "TensorIterator destination has invalid strides");
  }
  (validate(srcs), ...);
  offsets_.fill(0);
}

template <typename Dest, typename... Src>
template <typename... TensorTypes>
TensorIterator<Dest, Src...>::TensorIterator(const Tensor& dest,
                                             const TensorTypes&... srcs)
    : data_(std::make_tuple(dest.data(), srcs.data()...)),
      shapes_(dest.shape()),
      strides_({dest.stride(), srcs.stride()...}),
      idx_(dest.ndim(), 0),
      offsets_{},
      has_next_flag_(!dest.empty()) {
  auto validate = [&dest](const auto& tensor) {
    if (tensor.shape() != dest.shape() ||
        tensor.stride().size() != dest.ndim()) {
      throw std::invalid_argument(
          "TensorIterator operands must match the destination shape and rank");
    }
  };
  if (dest.stride().size() != dest.ndim()) {
    throw std::invalid_argument(
        "TensorIterator destination has invalid strides");
  }
  (validate(srcs), ...);
  offsets_.fill(0);
}

template <typename Dest, typename... Src>
bool TensorIterator<Dest, Src...>::has_next() const {
  return has_next_flag_;
}

template <typename Dest, typename... Src>
auto TensorIterator<Dest, Src...>::next() {
  assert(has_next_flag_);

  auto curr = dereference(std::make_index_sequence<n_terms>{});

  if (idx_.empty()) {
    has_next_flag_ = false;
    return curr;
  }

  bool advanced = false;

  for (std::size_t dim = idx_.size(); dim-- > 0;) {
    ++idx_[dim];

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] += strides_[Is][dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    if (idx_[dim] < shapes_[dim]) {
      advanced = true;
      break;
    }

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] -= strides_[Is][dim] * idx_[dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    idx_[dim] = 0;
  }

  if (!advanced) {
    has_next_flag_ = false;
  }

  return curr;
}

template <typename Dest, typename... Src>
void TensorIterator<Dest, Src...>::for_each(
    std::invocable<Dest&, Src&...> auto fn) {
  while (has_next()) {
    auto values = next();
    std::apply(fn, values);
  }
}

}  // namespace tensors
