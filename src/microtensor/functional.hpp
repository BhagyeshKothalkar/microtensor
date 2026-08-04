#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {
namespace functional {
// fix convention for the input tensor to be a, b, ...
inline Tensor& add_(Tensor& a, const Tensor& b) {
  cpu_kernels::add(a, b);
  return a;
}
inline Tensor& elementwise_multiply_(Tensor& a, const Tensor& b) {
  cpu_kernels::elementwise_multiply(a, b);
  return a;
}
inline Tensor& scalar_multiply_(Tensor& a, const float b) {
  cpu_kernels::scalar_multiply(a, b);
  return a;
}
inline Tensor& naive_matmul(Tensor& a, const Tensor& b) {
  Tensor res;
  // allocate res, broadcast if needded, manage the case of higher dimensions,
  // manage the views
  cpu_kernels::naive_matmul(a, b, res);
  return a;
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