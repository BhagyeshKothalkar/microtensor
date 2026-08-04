/**
 * @file broadcasting.hpp
 * @brief Utilities implementing tensor broadcasting.
 * Broadcasting allows tensors with compatible shapes to participate in
 * elementwise operations without physically replicating data. Dimensions are
 * matched from the trailing axis towards the leading axis, and dimensions of
 * size one are conceptually expanded to the required size.
 * This file provides utilities for:
 *  - Checking broadcast compatibility.
 *  - Computing the resulting broadcast shape.
 *  - Constructing broadcasted tensor views.
 *  - Broadcasting multiple tensors simultaneously.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "microtensor/tensor.hpp"

namespace tensors {

/**
 * @brief Checks whether a collection of tensors can be broadcast together.
 * Broadcasting follows NumPy/PyTorch rules:
 * - Dimensions are compared from the last axis.
 * - Missing leading dimensions are treated as size one.
 * - Two dimensions are compatible if they are equal or one of them is one.
 * Example:
 * @code
 * {2,3,4} and {3,4}      -> true
 * {5,1,7} and {1,8,7}    -> true
 * {2,3} and {4,3}        -> false
 * @endcode
 * @tparam Types Tensor parameter pack.
 * @param tensors Tensors to compare.
 * @return true if broadcasting is possible.
 */
template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
bool are_broadcastable(const Types&... tensors) {
  if constexpr (sizeof...(tensors) == 0) {
    return true;
  }

  /* Highest rank among all tensors. */
  size_t broadcast_rank = std::max({size_t(0), tensors.shape().size()...});
  /* Examine one logical dimension from the back. */
  for (size_t dim = 0; dim < broadcast_rank; ++dim) {
    size_t target_dim = 1;
    bool valid = true;
    /* Determine whether this tensor agrees with the target dimension. */
    auto check_dim = [&target_dim, &valid, dim](const auto& tensor) {
      if (!valid) return;
      /* default initialization to treat missing leading dimensions as size one.
       */
      size_t curr_dim = 1;
      /* Convert from logical broadcast dimension to tensor dimension. */

      int idx =
          static_cast<int>(tensor.shape().size()) - 1 - static_cast<int>(dim);

      if (idx >= 0) {
        curr_dim = tensor.shape()[idx];
      }
      /* Compare against the currently selected broadcast size. */
      /* if none of the dimensions are 1 and the dimensions are not equal */
      if (curr_dim != 1 && target_dim != 1 && curr_dim != target_dim) {
        valid = false;
        return;
      }
      /* Keep the largest compatible dimension. */
      target_dim = std::max(target_dim, curr_dim);
    };

    (check_dim(tensors), ...);
    /* Any incompatible dimension makes broadcasting impossible. */
    if (!valid) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Computes the common broadcast shape.
 * Returns the shape obtained after applying NumPy broadcasting rules to all
 * tensors.
 * Throws std::invalid_argument if the tensors are incompatible.
 * Example:
 * @code
 * {2,3,4} and {3,4}
 * ->
 * {2,3,4}
 * @endcode
 * @tparam Types Tensor parameter pack.
 * @param tensors Input tensors.
 * @return Broadcasted shape.
 * @throws std::invalid_argument
 *         If the tensors cannot be broadcast together.
 */
template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
std::vector<size_t> get_broadcast_shape(const Types&... tensors) {
  if constexpr (sizeof...(tensors) == 0) {
    return {};
  }

  size_t broadcast_rank = std::max({size_t(0), tensors.shape().size()...});
  /* Initialize every broadcast dimension to one. */
  std::vector<size_t> return_shape(broadcast_rank, 1);

  /* Resolve one logical dimension at a time. */
  for (size_t dim = 0; dim < broadcast_rank; ++dim) {
    bool valid = true;

    auto check_dim = [&return_shape, &valid, dim](const auto& tensor) {
      if (!valid) return;
      /* default initialization to treat missing leading dimensions as size one.
       */
      size_t curr_dim = 1;
      /* Convert from logical broadcast dimension to tensor dimension. */
      int idx =
          static_cast<int>(tensor.shape().size()) - 1 - static_cast<int>(dim);

      if (idx >= 0) {
        curr_dim = tensor.shape()[idx];
      }

      size_t target_idx = return_shape.size() - 1 - dim;
      size_t target_dim = return_shape[target_idx];
      /* Compare against the currently selected broadcast size. */
      if (curr_dim != 1 && target_dim != 1 && curr_dim != target_dim) {
        valid = false;
        return;
      }
      /* Update the resulting dimension with the larger compatible size. */
      return_shape[target_idx] = std::max(target_dim, curr_dim);
    };

    (check_dim(tensors), ...);
    /* Broadcasting rules were violated. */
    if (!valid) {
      throw std::invalid_argument("Tensors are not broadcastable");
    }
  }
  return return_shape;
}

/**
 * @brief Creates a broadcasted view of a tensor.
 * No data is copied. Instead, broadcasted dimensions receive stride zero,
 * causing every logical index along that dimension to reference the same
 * underlying element.
 * The caller is responsible for ensuring that the requested target shape is
 * broadcast-compatible with the input tensor.
 * Example:
 * @code
 * Shape {3,1}
 * ->
 * broadcast_to_shape(...,{3,4})
 * Resulting strides become
 * {1,0}
 * @endcode
 * @param in Input tensor.
 * @param target_shape Desired logical shape.
 * @return Broadcasted tensor view.
 */
inline Tensor broadcast_to_shape(const Tensor& in,
                                 const std::vector<size_t>& target_shape) {
  /* Broadcasted dimensions use stride zero. */
  std::vector<size_t> new_strides(target_shape.size(), 0);
  const auto& curr_shape = in.shape();
  const auto& curr_strides = in.stride();

  size_t target_rank = target_shape.size();
  size_t curr_rank = curr_shape.size();

  /* Compare dimensions starting from the trailing axis. */
  for (size_t i = 0; i < target_rank; i++) {
    if (i < curr_rank) {
      size_t curr_dim_size = curr_shape[curr_rank - 1 - i];
      size_t target_dim_size = target_shape[target_rank - 1 - i];
      /* Preserve the original stride whenever dimensions already match. */
      if (curr_dim_size == target_dim_size) {
        new_strides[target_rank - 1 - i] = curr_strides[curr_rank - 1 - i];
      }
    }
  }

  return Tensor(target_shape, new_strides, in.storage(), in.offset());
}

/**
 * @brief Broadcasts several tensors to a common target shape.
 * Every returned tensor is a lightweight view sharing the original storage.
 * No memory allocation or copying of tensor elements occurs.
 * @tparam Types Tensor parameter pack.
 * @param target_shape Desired broadcast shape.
 * @param tensors Input tensors.
 * @return Tuple containing broadcasted tensor views.
 */
template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
auto broadcast_tensors(const std::vector<size_t>& target_shape,
                       const Types&... tensors) {
  return std::make_tuple(broadcast_to_shape(tensors, target_shape)...);
}

/**
 * @brief Automatically broadcasts several tensors.
 * First computes the common broadcast shape and then constructs broadcasted
 * views of every tensor.
 * Example:
 * @code
 * auto [shape, views] = broadcast_tensors(a, b, c);
 * @endcode
 * @tparam Types Tensor parameter pack.
 * @param tensors Input tensors.
 * @return Pair consisting of:
 *         - the broadcast shape,
 *         - a tuple of broadcasted tensor views.
 */
template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
auto broadcast_tensors(const Types&... tensors) {
  auto target_shape = get_broadcast_shape(tensors...);
  return std::make_pair(target_shape,
                        broadcast_tensors(target_shape, tensors...));
}

}  // namespace tensors