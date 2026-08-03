#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "microtensor/tensor.hpp"

namespace tensors {

template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
bool are_broadcastable(const Types&... tensors) {
  if constexpr (sizeof...(tensors) == 0) {
    return true;
  }

  size_t broadcast_rank = std::max({size_t(0), tensors.shape().size()...});

  for (size_t dim = 0; dim < broadcast_rank; ++dim) {
    size_t target_dim = 1;
    bool valid = true;

    auto check_dim = [&target_dim, &valid, dim](const auto& tensor) {
      size_t curr_dim = 1;
      int idx =
          static_cast<int>(tensor.shape().size()) - 1 - static_cast<int>(dim);

      if (idx >= 0) {
        curr_dim = tensor.shape()[idx];
      }

      if (curr_dim != 1 && target_dim != 1 && curr_dim != target_dim) {
        valid = false;
      }

      target_dim = std::max(target_dim, curr_dim);
    };

    (check_dim(tensors), ...);
    if (!valid) {
      return false;
    }
  }
  return true;
}

template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
std::vector<size_t> get_broadcast_shape(const Types&... tensors) {
  if constexpr (sizeof...(tensors) == 0) {
    return {};
  }

  size_t broadcast_rank = std::max({size_t(0), tensors.shape().size()...});
  std::vector<size_t> return_shape(broadcast_rank, 1);

  for (size_t dim = 0; dim < broadcast_rank; ++dim) {
    bool valid = true;

    auto check_dim = [&return_shape, &valid, dim](const auto& tensor) {
      size_t curr_dim = 1;
      int idx =
          static_cast<int>(tensor.shape().size()) - 1 - static_cast<int>(dim);

      if (idx >= 0) {
        curr_dim = tensor.shape()[idx];
      }

      size_t target_idx = return_shape.size() - 1 - dim;
      size_t target_dim = return_shape[target_idx];

      if (curr_dim != 1 && target_dim != 1 && curr_dim != target_dim) {
        valid = false;
      }

      return_shape[target_idx] = std::max(target_dim, curr_dim);
    };

    (check_dim(tensors), ...);

    if (!valid) {
      throw std::invalid_argument("Tensors are not broadcastable");
    }
  }
  return return_shape;
}

inline Tensor broadcast_to_shape(const Tensor& in,
                                 const std::vector<size_t>& target_shape) {
  std::vector<size_t> new_strides(target_shape.size(), 0);
  const auto& curr_shape = in.shape();
  const auto& curr_strides = in.stride();

  size_t target_rank = target_shape.size();
  size_t curr_rank = curr_shape.size();

  for (size_t i = 0; i < target_rank; i++) {
    if (i < curr_rank) {
      size_t curr_dim_size = curr_shape[curr_rank - 1 - i];
      size_t target_dim_size = target_shape[target_rank - 1 - i];

      if (curr_dim_size == target_dim_size) {
        new_strides[target_rank - 1 - i] = curr_strides[curr_rank - 1 - i];
      }
    }
  }

  return Tensor(target_shape, new_strides, in.storage(), in.offset());
}

template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
auto broadcast_tensors(const std::vector<size_t>& target_shape,
                       const Types&... tensors) {
  return std::make_tuple(broadcast_to_shape(tensors, target_shape)...);
}

template <typename... Types>
  requires(std::same_as<std::decay_t<Types>, Tensor> && ...)
auto broadcast_tensors(const Types&... tensors) {
  auto target_shape = get_broadcast_shape(tensors...);
  return std::make_pair(target_shape,
                        broadcast_tensors(target_shape, tensors...));
}

}  // namespace tensors