#include "tensor.hpp"
#include "tensor_iterator.hpp"
#include <cassert>
#include <cstddef>
#include <vector>

namespace tensors {

template <typename T> Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b) {
  auto [res_shape, broadcasted_tensors] = broadcast_tensors(a, b);
  auto [broadcast_a, broadcast_b] = broadcasted_tensors;

  Tensor<T> res(res_shape);

  TensorIterator<T, const T, const T> itr(res, broadcast_a, broadcast_b);

  while (itr.has_next()) {
    auto [res_val, a_val, b_val] = itr.next();
    res_val = a_val + b_val;
  }
  return res;
}

template <typename T>
Tensor<T> elementwise_multiply(const Tensor<T> &a, const Tensor<T> &b) {
  auto [res_shape, broadcasted_tensors] = broadcast_tensors(a, b);
  auto [broadcast_a, broadcast_b] = broadcasted_tensors;

  Tensor<T> res(res_shape);
  TensorIterator<T, const T, const T> itr(res, broadcast_a, broadcast_b);

  while (itr.has_next()) {
    auto [res_val, a_val, b_val] = itr.next();
    res_val = a_val * b_val;
  }
  return res;
}

template <typename T>
Tensor<T> scalar_multiply(const Tensor<T> &a, const T &s) {
  Tensor<T> res(a.shape());
  TensorIterator<T, const T> itr(res, a);

  while (itr.has_next()) {
    auto [res_val, a_val] = itr.next();
    res_val = a_val * s;
  }
  return res;
}

template <typename T>
Tensor<T> naive_matmul(const Tensor<T> &a, const Tensor<T> &b) {
  assert(a.ndim() == 2 && b.ndim() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  std::vector<size_t> a_shape = a.shape(), b_shape = b.shape();
  size_t m = a_shape[0], k = a_shape[1], n = b_shape[1];

  Tensor<T> res({m, n});
  // res.fill(0);

  std::vector<size_t> iter_shape = {m, n, k};

  Tensor<T> a_view(iter_shape, {a.stride()[0], 0, a.stride()[1]}, a.storage(),
                   a.offset());

  Tensor<T> b_view(iter_shape, {0, b.stride()[1], b.stride()[0]}, b.storage(),
                   b.offset());

  Tensor<T> res_view(iter_shape, {res.stride()[0], res.stride()[1], 0},
                     res.storage(), res.offset());

  TensorIterator<T, const T, const T> it(res_view, a_view, b_view);
  while (it.has_next()) {
    auto [res_val, a_val, b_val] = it.next();
    res_val += a_val * b_val;
  }

  return res;
}

} // namespace tensors