/**
 * @file tensor_iterator.hpp
 * @brief Generic lock-step iterator over one destination and multiple source
 * tensors.
 *
 * TensorIterator provides a common iteration mechanism for elementwise kernels.
 * It walks every logical index of a (broadcasted) tensor exactly once while
 * maintaining independent offsets into each participating tensor.
 *
 * The iterator itself performs no computation—it only yields references to the
 * current elements. Numerical kernels implement the actual operation using
 * these references.
 *
 * Typical usage:
 * @code
 * TensorIterator<float, const float, const float> it(dst, a, b);
 *
 * while (it.has_next()) {
 *     auto [out, x, y] = it.next();
 *     out = x + y;
 * }
 * @endcode
 */

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "microtensor/tensor.hpp"

namespace tensors {

/**
 * @brief Lock-step iterator for elementwise tensor kernels.
 *
 * Dest denotes the destination element type, while Src... denote the source
 * element types. The iterator stores raw pointers together with per-tensor
 * strides and offsets, allowing efficient traversal without repeatedly
 * recomputing multidimensional indices.
 *
 * Iteration order is row-major over the logical tensor shape.
 *
 * @tparam Dest Destination element type.
 * @tparam Src Source element types.
 */
template <typename Dest, typename... Src>
class TensorIterator {
 private:
  /* Number of participating tensors. */
  static constexpr size_t n_terms = sizeof...(Src) + 1;
  /* Base pointer for each tensor. */
  std::tuple<Dest*, Src*...> data_;
  /* Logical iteration shape. */
  std::vector<size_t> shapes_;
  /* Per-tensor strides. */
  std::array<std::vector<size_t>, n_terms> strides_;
  /* Current multidimensional index. */
  std::vector<size_t> idx_;
  /* Current flat offset into every tensor. */
  std::array<size_t, n_terms> offsets_;
  /* Whether another element remains. */
  bool has_next_flag_;

  /**
   * @brief Produces references to the current tensor elements.
   *
   * Expands the pointer tuple using the current offsets and returns a tuple of
   * references suitable for structured bindings.
   *
   * @return Tuple containing references to the current destination and source
   * elements.
   */
  template <std::size_t... Is>
  auto dereference(std::index_sequence<Is...>);

 public:
  /**
   * @brief An iterator over one destination and several source
   * tensors.
   *
   * All tensors are assumed to already have compatible logical shapes and
   * broadcasted strides. The iterator simply traverses them in lock-step.
   *
   * Example:
   * @code
   * TensorIterator<float,const float,const float> it(dst, a, b);
   * @endcode
   *
   * @param dest Destination tensor.
   * @param srcs Source tensors.
   */
  template <typename... TensorTypes>
  TensorIterator(Tensor& dest, const TensorTypes&... srcs);

  /**
   * @brief Checks whether another element is available.
   *
   * @return true if next() may be called.
   */
  bool has_next() const;

  /**
   * @brief Returns the current tensor elements and advances the iterator.
   *
   * The returned tuple contains references into the underlying tensors. After
   * returning the current elements, the iterator advances to the next logical
   * position using row-major ordering.
   *
   * Example:
   * @code
   * auto [out, x, y] = iter.next();
   * out = x * y;
   * @endcode
   *
   * @return Tuple of references to the current destination and source elements.
   */
  auto next();
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
      idx_(std::vector<size_t>(dest.ndim(), 0)),
      offsets_{},
      has_next_flag_(!dest.empty()) {
  offsets_.fill(0);
}

template <typename Dest, typename... Src>
bool TensorIterator<Dest, Src...>::has_next() const {
  return has_next_flag_;
}

template <typename Dest, typename... Src>
auto TensorIterator<Dest, Src...>::next() {
  assert(!idx_.empty());
  assert(has_next_flag_);

  auto curr = dereference(std::make_index_sequence<n_terms>{});
  bool advanced = false;

  for (size_t dim = idx_.size(); dim-- > 0;) {
    idx_[dim]++;

    /* Increment every tensor offset for this dimension. */
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] += strides_[Is][dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    /* No carry required. */
    if (idx_[dim] < shapes_[dim]) {
      advanced = true;
      break;
    }
    /* Carry into the next outer dimension. */
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] -= strides_[Is][dim] * idx_[dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    idx_[dim] = 0;
  }
  /* Entire iteration space has been exhausted. */
  if (!advanced) {
    has_next_flag_ = false;
  }

  return curr;
}

}  // namespace tensors