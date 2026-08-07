#pragma once

#include "tensor.hpp"
namespace tensors {
namespace cpu_kernels {
void add(Tensor& a, const Tensor& b);
void sub(Tensor& a, const Tensor& b);
void elementwise_multiply(Tensor& a, const Tensor& b);
void elementwise_div(Tensor& a, const Tensor& b);
void scalar_multiply(Tensor& a, const float& s);
void scalar_add(Tensor& a, const float& s);
void naive_matmul(const Tensor& a, const Tensor& b, Tensor& res);
void relu(Tensor& x);
void relu_backward(const Tensor& grad_out, const Tensor& input,
                   Tensor& grad_in);
void softmax(Tensor& x);
void sin_kernel(const Tensor& in, Tensor& out);
void cos_kernel(const Tensor& in, Tensor& out);
void sum_kernel(const Tensor& in, Tensor& out);
};  // namespace cpu_kernels
}  // namespace tensors
