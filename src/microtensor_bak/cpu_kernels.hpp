#pragma once

#include "tensor.hpp"

namespace tensors {
namespace cpu_kernels {

Tensor& add(Tensor& a, const Tensor& b);
Tensor& add(Tensor& a, const float b);
Tensor& sub(Tensor& a, const Tensor& b);
Tensor& sub(Tensor& a, const float b);
Tensor& mul(Tensor& a, const Tensor& b);
Tensor& mul(Tensor& a, const float s);
Tensor& div(Tensor& a, const Tensor& b);
Tensor& div(Tensor& a, const float s);

Tensor& neg(Tensor& a);
Tensor& reciprocal(Tensor& a);
Tensor& sin(Tensor& x);
Tensor& cos(Tensor& x);
Tensor& relu(Tensor& x);
Tensor& relu_backward(const Tensor& grad_out, const Tensor& input,
                      Tensor& grad_in);
Tensor& softmax(Tensor& x);
Tensor& sqrt(Tensor& x);

Tensor& naive_matmul(const Tensor& a, const Tensor& b, Tensor& res);
Tensor sum(const Tensor& src, const std::vector<index_t>& dims);
Tensor clone(const Tensor& a);

};  // namespace cpu_kernels
}  // namespace tensors
