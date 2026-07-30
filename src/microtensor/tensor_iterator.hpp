#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace tensors {

template <typename T>
class Tensor;

template <typename Dest, typename... Src>
class TensorIterator {
  static constexpr size_t n_terms = sizeof...(Src) + 1;
  std::tuple<Dest*, Src*...> data_;

  std::vector<size_t> shapes_;
  std::array<std::vector<size_t>, n_terms> strides_;

  std::vector<std::size_t> idx_;
  std::array<size_t, n_terms> offsets_;
  bool has_next_flag_;

  template <std::size_t... Is>
  auto dereference(std::index_sequence<Is...>) {
    return std::tuple<Dest&, Src&...>{std::get<Is>(data_)[offsets_[Is]]...};
  }

 public:
  template <typename... TensorTypes>
  TensorIterator(Tensor<Dest>& dest, TensorTypes&... srcs) {
    data_ = std::make_tuple(dest.data(), srcs.data()...);
    shapes_ = dest.shape();
    strides_ = {dest.stride(), srcs.stride()...};
    idx_ = std::vector<size_t>(dest.ndim(), 0);
    offsets_.fill(0);
    has_next_flag_ = !dest.empty();
  }

  bool has_next() const { return has_next_flag_; }

  auto next() {
    assert(!idx_.empty());
    assert(has_next_flag_);

    auto curr = dereference(std::make_index_sequence<n_terms>{});
    bool advanced = false;

    for (size_t dim = idx_.size(); dim-- > 0;) {
      idx_[dim]++;

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
};

}  // namespace tensors