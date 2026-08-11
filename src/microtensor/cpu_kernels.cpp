#include "microtensor/cpu_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "tensor.hpp"
#include "tensor_iterator.hpp"

namespace tensors {
namespace cpu_kernels {

Tensor& add(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float>(a, b).for_each(
      [](float& x, const float& y) { x += y; });
  return a;
}

Tensor& add(Tensor& a, const float s) {
  TensorIterator<float>(a).for_each([&s](float& x) { x += s; });
  return a;
}

Tensor& sub(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float>(a, b).for_each(
      [](float& x, const float& y) { x -= y; });
  return a;
}

Tensor& sub(Tensor& a, const float s) {
  TensorIterator<float>(a).for_each([&s](float& x) { x -= s; });
  return a;
}

Tensor& mul(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float>(a, b).for_each(
      [](float& x, const float& y) { x *= y; });
  return a;
}

Tensor& mul(Tensor& a, const float s) {
  TensorIterator<float>(a).for_each([&s](float& x) { x *= s; });
  return a;
}

Tensor& div(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float>(a, b).for_each(
      [](float& x, const float& y) { x /= y; });
  return a;
}

Tensor& div(Tensor& a, const float s) {
  TensorIterator<float>(a).for_each([&s](float& x) { x /= s; });
  return a;
}

Tensor& neg(Tensor& x) {
  TensorIterator<float>(x).for_each([](float& value) { value = -value; });
  return x;
}

Tensor& reciprocal(Tensor& x) {
  TensorIterator<float>(x).for_each([](float& value) { value = 1 / value; });
  return x;
}

Tensor& sin(Tensor& x) {
  TensorIterator<float>(x).for_each(
      [](float& value) { value = std::sin(value); });
  return x;
}

Tensor& cos(Tensor& x) {
  TensorIterator<float>(x).for_each(
      [](float& value) { value = std::cos(value); });
  return x;
}

Tensor& relu(Tensor& x) {
  TensorIterator<float>(x).for_each(
      [](float& value) { value = std::max(value, 0.0f); });
  return x;
}

Tensor& softmax(Tensor& x) {
  float sum = 0;

  TensorIterator<const float>(x).for_each(
      [&sum](const float& val) { sum += std::exp(val); });

  TensorIterator<float>(x).for_each(
      [sum](float& val) { val = std::exp(val) / sum; });
  return x;
}

Tensor &sqrt(Tensor &x){
  TensorIterator<float>(x).for_each([](float&x_val){x_val = sqrtf(x_val);});
  return x;
}

Tensor& naive_matmul(const Tensor& a, const Tensor& b, Tensor& res) {
  TensorIterator<float, const float, const float>(res, a, b).for_each(
      [](float& res_val, const float& a_val, const float& b_val) {
        res_val += a_val * b_val;
      });
  return res;
}

Tensor& relu_backward(const Tensor& grad_out, const Tensor& input,
                      Tensor& grad_in) {
  TensorIterator<float, const float, const float> it(grad_in, grad_out, input);
  it.for_each([](float& gin_val, const float& g_val, const float& in_val) {
    gin_val = (in_val > 0.0f) ? g_val : 0.0f;
  });
  return grad_in;
}

Tensor sum(const Tensor& in) {
  float acc = 0.0f;
  TensorIterator<const float>(in).for_each(
      [&acc](const float& a_val) { acc += a_val; });
  return Tensor({1}, {acc});
}

Tensor clone(const Tensor& a) {
  Tensor res(a.shape());
  TensorIterator<float, const float>(res, a).for_each(
      [](float& dst, const float& src) { dst = src; });
  return res;
}

};  // namespace cpu_kernels
};  // namespace tensors
