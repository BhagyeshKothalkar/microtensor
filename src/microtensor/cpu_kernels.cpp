#include "microtensor/cpu_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "microtensor/broadcasting.hpp"
#include "tensor.hpp"
#include "tensor_iterator.hpp"

namespace tensors {

Tensor add(const Tensor& a, const Tensor& b) {
  auto [res_shape, broadcasted_tensors] = broadcast_tensors(a, b);
  auto [broadcast_a, broadcast_b] = broadcasted_tensors;

  Tensor res(res_shape);

  TensorIterator<float, const float, const float> itr(res, broadcast_a,
                                                      broadcast_b);

  while (itr.has_next()) {
    auto&& [res_val, a_val, b_val] = itr.next();
    res_val = a_val + b_val;
  }
  return res;
}

Tensor elementwise_multiply(const Tensor& a, const Tensor& b) {
  auto [res_shape, broadcasted_tensors] = broadcast_tensors(a, b);
  auto [broadcast_a, broadcast_b] = broadcasted_tensors;

  Tensor res(res_shape);
  TensorIterator<float, const float, const float> itr(res, broadcast_a,
                                                      broadcast_b);

  while (itr.has_next()) {
    auto&& [res_val, a_val, b_val] = itr.next();
    res_val = a_val * b_val;
  }
  return res;
}

Tensor scalar_multiply(const Tensor& a, const float& s) {
  Tensor res(a.shape());
  TensorIterator<float, const float> itr(res, a);

  while (itr.has_next()) {
    auto&& [res_val, a_val] = itr.next();
    res_val = a_val * s;
  }
  return res;
}
Tensor naive_matmul(const Tensor& a, const Tensor& b) {
  std::cout << "matmul(" << a.shape()[0] << "," << a.shape()[1] << ") x ("
            << b.shape()[0] << "," << b.shape()[1] << ")\n";

  assert(a.ndim() == 2 && b.ndim() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  std::vector<size_t> a_shape = a.shape(), b_shape = b.shape();
  size_t m = a_shape[0], k = a_shape[1], n = b_shape[1];

  Tensor res({m, n});
  // res.fill(0);

  std::vector<size_t> iter_shape = {m, n, k};

  Tensor a_view(iter_shape, {a.stride()[0], 0, a.stride()[1]}, a.storage(),
                a.offset());

  Tensor b_view(iter_shape, {0, b.stride()[1], b.stride()[0]}, b.storage(),
                b.offset());

  Tensor res_view(iter_shape, {res.stride()[0], res.stride()[1], 0},
                  res.storage(), res.offset());

  TensorIterator<float, const float, const float> it(res_view, a_view, b_view);
  while (it.has_next()) {
    auto&& [res_val, a_val, b_val] = it.next();
    res_val += a_val * b_val;
  }

  return res;
}

// inplace
Tensor relu_(Tensor& x) {
  TensorIterator<float> it(x);
  while (it.has_next()) {
    auto&& [x_val] = it.next();
    x_val = std::max(x_val, 0.0f);
  }
  return x;
}

Tensor softmax_(Tensor& x) {
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
  return x;
  
}
}  // namespace tensors
