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

template <typename Dest, typename Src>
class ReductionIterator {
 private:
  Dest* dest_;
  Src* src_;

  std::vector<std::size_t> dest_shape_;
  std::vector<std::size_t> src_dest_strides_;
  std::vector<std::size_t> dest_idx_;
  std::size_t dest_offset_ = 0;

  std::vector<std::size_t> reduce_shape_;
  std::vector<std::size_t> reduce_strides_;
  std::vector<std::size_t> reduce_idx_;
  std::size_t reduce_offset_ = 0;

  bool has_next_flag_ = false;

  void reset_group(std::size_t base);
  void advance_destination();
  bool advance_group();

 public:
  ReductionIterator(Tensor& dest, const Tensor& src,
                    const std::vector<index_t>& dims);

  bool has_next() const;

  auto next();

  void for_each_group(std::invocable<Src&> auto fn);

  void for_each(std::invocable<Dest&, ReductionIterator&> auto fn);
};

template <typename Dest, typename Src>
void ReductionIterator<Dest, Src>::reset_group(std::size_t base) {
  reduce_idx_.assign(reduce_shape_.size(), 0);
  reduce_offset_ = base;
}

template <typename Dest, typename Src>
void ReductionIterator<Dest, Src>::advance_destination() {
  for (std::size_t dim = dest_idx_.size(); dim-- > 0;) {
    ++dest_idx_[dim];
    dest_offset_ += src_dest_strides_[dim];

    if (dest_idx_[dim] < dest_shape_[dim]) {
      return;
    }

    dest_offset_ -= src_dest_strides_[dim] * dest_idx_[dim];
    dest_idx_[dim] = 0;
  }

  has_next_flag_ = false;
}

template <typename Dest, typename Src>
bool ReductionIterator<Dest, Src>::advance_group() {
  for (std::size_t dim = reduce_idx_.size(); dim-- > 0;) {
    ++reduce_idx_[dim];
    reduce_offset_ += reduce_strides_[dim];

    if (reduce_idx_[dim] < reduce_shape_[dim]) {
      return true;
    }

    reduce_offset_ -= reduce_strides_[dim] * reduce_idx_[dim];
    reduce_idx_[dim] = 0;
  }

  return false;
}

template <typename Dest, typename Src>
ReductionIterator<Dest, Src>::ReductionIterator(
    Tensor& dest, const Tensor& src, const std::vector<index_t>& dims)
    : dest_(dest.data()),
      src_(src.data()),
      dest_shape_(dest.shape()),
      dest_idx_(dest.ndim(), 0),
      has_next_flag_(!dest.empty()) {
  if (dest.stride().size() != dest.ndim()) {
    throw std::invalid_argument(
        "ReductionIterator destination has invalid strides");
  }

  if (src.stride().size() != src.ndim()) {
    throw std::invalid_argument("ReductionIterator source has invalid strides");
  }

  std::vector<bool> reduced(src.ndim());

  for (index_t dim : dims) {
    if (dim < 0) {
      dim += static_cast<index_t>(src.ndim());
    }

    if (dim < 0 || static_cast<std::size_t>(dim) >= src.ndim()) {
      throw std::out_of_range(
          "ReductionIterator reduction dimension out of range");
    }

    reduced[static_cast<std::size_t>(dim)] = true;
  }

  for (std::size_t dim = 0; dim < src.ndim(); ++dim) {
    if (reduced[dim]) {
      reduce_shape_.push_back(src.shape()[dim]);
      reduce_strides_.push_back(src.stride()[dim]);
    } else {
      src_dest_strides_.push_back(src.stride()[dim]);

      if (dest_shape_[src_dest_strides_.size() - 1] != src.shape()[dim]) {
        throw std::invalid_argument(
            "ReductionIterator destination shape does not match "
            "non-reduced source dimensions");
      }
    }
  }

  if (dest_shape_.size() != src_dest_strides_.size()) {
    throw std::invalid_argument(
        "ReductionIterator destination rank does not match "
        "the number of non-reduced source dimensions");
  }

  reduce_idx_.resize(reduce_shape_.size());
}

template <typename Dest, typename Src>
bool ReductionIterator<Dest, Src>::has_next() const {
  return has_next_flag_;
}

template <typename Dest, typename Src>
auto ReductionIterator<Dest, Src>::next() {
  assert(has_next_flag_);

  Dest& dst = dest_[dest_offset_];

  reset_group(dest_offset_);
  advance_destination();

  return std::tuple<Dest&, ReductionIterator&>{dst, *this};
}

template <typename Dest, typename Src>
void ReductionIterator<Dest, Src>::for_each_group(
    std::invocable<Src&> auto fn) {
  reset_group(dest_offset_);

  do {
    fn(src_[reduce_offset_]);
  } while (advance_group());
}

template <typename Dest, typename Src>
void ReductionIterator<Dest, Src>::for_each(
    std::invocable<Dest&, ReductionIterator&> auto fn) {
  while (has_next()) {
    auto [dst, group] = next();
    fn(dst, group);
  }
}
}  // namespace tensors
