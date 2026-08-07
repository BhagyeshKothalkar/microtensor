#include "microtensor/cpu_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "tensor.hpp"
#include "tensor_iterator.hpp"

namespace tensors {
namespace cpu_kernels {
void add(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float> itr(a, b);
  while (itr.has_next()) {
    auto&& [a_val, b_val] = itr.next();
    a_val += b_val;
  }
}

void sub(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float> itr(a, b);
  while (itr.has_next()) {
    auto&& [a_val, b_val] = itr.next();
    a_val -= b_val;
  }
}

void elementwise_multiply(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float> itr(a, b);
  while (itr.has_next()) {
    auto&& [a_val, b_val] = itr.next();
    a_val *= b_val;
  }
}

void elementwise_div(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float> itr(a, b);
  while (itr.has_next()) {
    auto&& [a_val, b_val] = itr.next();
    a_val /= b_val;
  }
}

void scalar_multiply(Tensor& a, const float& s) {
  TensorIterator<float> itr(a);
  while (itr.has_next()) {
    auto&& [a_val] = itr.next();
    a_val *= s;
  }
}

void scalar_add(Tensor& a, const float& s) {
  TensorIterator<float> itr(a);
  while (itr.has_next()) {
    auto&& [a_val] = itr.next();
    a_val += s;
  }
}

// i first have to create views in a particular manner for this to work.
void naive_matmul(const Tensor& a, const Tensor& b, Tensor& res) {
  TensorIterator<float, const float, const float> it(res, a, b);
  while (it.has_next()) {
    auto&& [res_val, a_val, b_val] = it.next();
    res_val += a_val * b_val;
  }
}

void relu(Tensor& x) {
  TensorIterator<float> it(x);
  while (it.has_next()) {
    auto&& [x_val] = it.next();
    x_val = std::max(x_val, 0.0f);
  }
}

void relu_backward(const Tensor& grad_out, const Tensor& input,
                   Tensor& grad_in) {
  TensorIterator<float, const float, const float> it(grad_in, grad_out, input);
  while (it.has_next()) {
    auto&& [gin_val, g_val, in_val] = it.next();
    gin_val = (in_val > 0.0f) ? g_val : 0.0f;
  }
}

void softmax(Tensor& x) {
  float sum = 0;

  TensorIterator<const float> it(x);
  while (it.has_next()) {
    auto&& [x_val] = it.next();
    sum += std::exp(x_val);
  }

  TensorIterator<float> it2(x);
  while (it2.has_next()) {
    auto&& [x_val] = it2.next();
    x_val = std::exp(x_val) / sum;
  }
}

void sin_kernel(const Tensor& in, Tensor& out) {
  TensorIterator<float, const float> it(out, in);
  while (it.has_next()) {
    auto&& [out_val, in_val] = it.next();
    out_val = std::sin(in_val);
  }
}

void cos_kernel(const Tensor& in, Tensor& out) {
  TensorIterator<float, const float> it(out, in);
  while (it.has_next()) {
    auto&& [out_val, in_val] = it.next();
    out_val = std::cos(in_val);
  }
}

void sum_kernel(const Tensor& in, Tensor& out) {
  float acc = 0.0f;
  Tensor mutable_in = in;
  TensorIterator<const float> it(mutable_in);
  while (it.has_next()) {
    auto&& [in_val] = it.next();
    acc += in_val;
  }
  out.data()[0] = acc;
}

};  // namespace cpu_kernels
};  // namespace tensors
