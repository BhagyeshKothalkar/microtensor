#pragma once

#include <span>
#include <vector>

#include "tensor.hpp"

namespace microtensor {

std::vector<size_t> broadcast_shape(std::span<const Tensor* const> tensors);

Tensor broadcast_to(const Tensor& input, std::span<const size_t> target_shape);

Tensor sum_to_shape(const Tensor& input, std::span<const size_t> target_shape);

}  // namespace microtensor
