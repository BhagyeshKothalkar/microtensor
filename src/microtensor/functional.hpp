#include <cassert>
#include <stdexcept>
#include <string>

#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {
namespace functional {
// fix convention for the input tensor to be a, b, ...
inline Tensor& add_(Tensor& a, const Tensor& b) {
  auto&& [_, broadcasted_tensors] = broadcast_tensors(a, b);
  auto&& [a_broadcast, b_broadcast] = broadcasted_tensors;
  cpu_kernels::add(a_broadcast, b_broadcast);
  return a;
}
inline Tensor& elementwise_multiply_(Tensor& a, const Tensor& b) {
  auto&& [_, broadcasted_tensors] = broadcast_tensors(a, b);
  auto&& [a_broadcast, b_broadcast] = broadcasted_tensors;
  cpu_kernels::elementwise_multiply(a_broadcast, b_broadcast);
  return a;
}
inline Tensor& scalar_multiply_(Tensor& a, const float b) {
  cpu_kernels::scalar_multiply(a, b);
  return a;
}

inline Tensor naive_matmul(const Tensor& a, const Tensor& b) {
  if (!(a.ndim() == 2 && b.ndim() == 2 && a.shape()[1] == b.shape()[0])) {
    throw(std::runtime_error("" + std::to_string(a.ndim()) + " " +
                             std::to_string(b.ndim())));
  }
  size_t i = a.shape()[0], k = a.shape()[1], j = b.shape()[1];
  Tensor a_view = a.view({i, k, 1});
  Tensor b_view = b.view({1, k, j});

  auto [_, tensors] = broadcast_tensors(a_view, b_view);
  auto& [A, B] = tensors;

  Tensor res = Tensor::zeros({i, j});
  Tensor res_view = broadcast_to_shape(res.view({i, 1, j}), {i, k, j});

  cpu_kernels::naive_matmul(A, B, res_view);
  return res;
}

inline Tensor& relu_(Tensor& a) {
  cpu_kernels::relu(a);
  return a;
}
inline Tensor& softmax_(Tensor& a) {
  cpu_kernels::softmax(a);
  return a;
}
};  // namespace functional
};  // namespace tensors