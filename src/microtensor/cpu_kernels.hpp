#pragma once

#include "tensor.hpp"
namespace tensors {

Tensor add(const Tensor& a, const Tensor& b);
Tensor elementwise_multiply(const Tensor& a, const Tensor& b);
Tensor scalar_multiply(const Tensor& a, const float& s);
Tensor naive_matmul(const Tensor& a, const Tensor& b);

}  // namespace tensors
