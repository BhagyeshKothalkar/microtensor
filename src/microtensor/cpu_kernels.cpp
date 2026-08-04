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

void elementwise_multiply(Tensor& a, const Tensor& b) {
  TensorIterator<float, const float> itr(a, b);
  while (itr.has_next()) {
    auto&& [a_val, b_val] = itr.next();
    a_val *= b_val;
  }
}

void scalar_multiply(Tensor& a, const float& s) {
  TensorIterator<float> itr(a);
  while (itr.has_next()) {
    auto&& [a_val] = itr.next();
    a_val *= s;
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

void softmax(Tensor& x) {
  float sum = 0;

  TensorIterator<const float> it(x);
  while (it.has_next()) {
    auto&& [x_val] = it.next();
    sum += exp(x_val);
  }

  TensorIterator<float> it2(x);
  while (it2.has_next()) {
    auto&& [x_val] = it2.next();
    x_val = exp(x_val) / sum;
  }
}

};  // namespace cpu_kernels
};  // namespace tensors
