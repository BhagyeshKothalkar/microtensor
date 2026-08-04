#pragma once

#include "tensor.hpp"
namespace tensors {
namespace cpu_kernels {
void add(Tensor& a, const Tensor& b);
void elementwise_multiply(Tensor& a, const Tensor& b);
void scalar_multiply(Tensor& a, const float& s);
void naive_matmul(const Tensor& a, const Tensor& b, Tensor& res);
void relu(Tensor& x);
void softmax(Tensor& x);
};  // namespace cpu_kernels
}  // namespace tensors
