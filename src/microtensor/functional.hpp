#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "microtensor/autograd.hpp"
#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {
namespace functional {

// Forward declarations of out-of-place differentiable operators
inline Tensor add(const Tensor& a, const Tensor& b);
inline Tensor add(const Tensor& a, float s);
inline Tensor add(float s, const Tensor& a);

inline Tensor sub(const Tensor& a, const Tensor& b);
inline Tensor sub(const Tensor& a, float s);
inline Tensor sub(float s, const Tensor& a);

inline Tensor mul(const Tensor& a, const Tensor& b);
inline Tensor mul(const Tensor& a, float s);
inline Tensor mul(float s, const Tensor& a);

inline Tensor div(const Tensor& a, const Tensor& b);
inline Tensor div(const Tensor& a, float s);
inline Tensor div(float s, const Tensor& a);

inline Tensor neg(const Tensor& a);
inline Tensor sin(const Tensor& a);
inline Tensor cos(const Tensor& a);
inline Tensor relu(const Tensor& a);
inline Tensor sum(const Tensor& a);
inline Tensor mean(const Tensor& a);
inline Tensor matmul(const Tensor& a, const Tensor& b);

inline Tensor reshape(const Tensor& a, const std::vector<size_t>& new_shape);
inline Tensor transpose(const Tensor& a, size_t dim0, size_t dim1);
inline Tensor broadcast_to(const Tensor& a,
                           const std::vector<size_t>& target_shape);

/* In-place mutation functions */
inline Tensor& add_(Tensor& a, const Tensor& b) {
  auto&& [_, broadcasted_tensors] = broadcast_tensors(a, b);
  auto&& [a_broadcast, b_broadcast] = broadcasted_tensors;
  cpu_kernels::add(a_broadcast, b_broadcast);
  return a;
}

inline Tensor& sub_(Tensor& a, const Tensor& b) {
  auto&& [_, broadcasted_tensors] = broadcast_tensors(a, b);
  auto&& [a_broadcast, b_broadcast] = broadcasted_tensors;
  cpu_kernels::sub(a_broadcast, b_broadcast);
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

inline Tensor& relu_(Tensor& a) {
  cpu_kernels::relu(a);
  return a;
}

inline Tensor& softmax_(Tensor& a) {
  cpu_kernels::softmax(a);
  return a;
}

/* Differentiable Functional Operations */

inline Tensor add(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::add(result, b_bc);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(grad);
      if (rhs.requires_grad()) rhs.add_grad(grad);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor add(const Tensor& a, float s) {
  Tensor result = a.clone();
  cpu_kernels::scalar_add(result, s);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(grad);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor add(float s, const Tensor& a) { return add(a, s); }

inline Tensor sub(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::sub(result, b_bc);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(grad);
      if (rhs.requires_grad()) rhs.add_grad(neg(grad));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor sub(const Tensor& a, float s) { return add(a, -s); }

inline Tensor sub(float s, const Tensor& a) { return add(neg(a), s); }

inline Tensor mul(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::elementwise_multiply(result, b_bc);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(mul(grad, rhs));
      if (rhs.requires_grad()) rhs.add_grad(mul(grad, lhs));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor mul(const Tensor& a, float s) {
  Tensor result = a.clone();
  cpu_kernels::scalar_multiply(result, s);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, s](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(mul(grad, s));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor mul(float s, const Tensor& a) { return mul(a, s); }

inline Tensor div(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::elementwise_div(result, b_bc);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(div(grad, rhs));
      if (rhs.requires_grad())
        rhs.add_grad(neg(div(mul(grad, lhs), mul(rhs, rhs))));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor div(const Tensor& a, float s) { return mul(a, 1.0f / s); }

inline Tensor div(float s, const Tensor& a) {
  Tensor result(a.shape());
  TensorIterator<float, const float> it(result, a);
  while (it.has_next()) {
    auto&& [out_val, in_val] = it.next();
    out_val = s / in_val;
  }

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, s](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(neg(mul(grad, div(s, mul(lhs, lhs)))));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor neg(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::scalar_multiply(result, -1.0f);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(neg(grad));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor sin(const Tensor& a) {
  Tensor result(a.shape());
  cpu_kernels::sin_kernel(a, result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(mul(grad, cos(lhs)));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor cos(const Tensor& a) {
  Tensor result(a.shape());
  cpu_kernels::cos_kernel(a, result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) lhs.add_grad(neg(mul(grad, sin(lhs))));
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor relu(const Tensor& a) {
  Tensor result = a.clone();
  relu_(result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        Tensor grad_input(lhs.shape());
        cpu_kernels::relu_backward(grad, lhs, grad_input);
        lhs.add_grad(grad_input);
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor sum(const Tensor& a) {
  Tensor result({1});
  cpu_kernels::sum_kernel(a, result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(broadcast_to_shape(grad, lhs.shape()));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor mean(const Tensor& a) {
  Tensor result = sum(a);
  float n = static_cast<float>(a.numel());
  return mul(result, 1.0f / n);
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

inline Tensor matmul(const Tensor& a, const Tensor& b) {
  Tensor result = naive_matmul(a, b);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        Tensor rhs_t = rhs.transpose(0, 1);
        lhs.add_grad(matmul(grad, rhs_t));
      }
      if (rhs.requires_grad()) {
        Tensor lhs_t = lhs.transpose(0, 1);
        rhs.add_grad(matmul(lhs_t, grad));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor reshape(const Tensor& a, const std::vector<size_t>& new_shape) {
  Tensor result = a.view(new_shape);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad.view(lhs.shape()));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor transpose(const Tensor& a, size_t dim0, size_t dim1) {
  Tensor result = a.transpose(dim0, dim1);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, dim0, dim1](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad.transpose(dim0, dim1));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

inline Tensor broadcast_to(const Tensor& a,
                           const std::vector<size_t>& target_shape) {
  Tensor result = broadcast_to_shape(a, target_shape);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad);
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

}  // namespace functional

/* Global Operator Overloads */
inline Tensor operator+(const Tensor& a, const Tensor& b) {
  return functional::add(a, b);
}
inline Tensor operator+(const Tensor& a, float s) {
  return functional::add(a, s);
}
inline Tensor operator+(float s, const Tensor& a) {
  return functional::add(s, a);
}

inline Tensor operator-(const Tensor& a, const Tensor& b) {
  return functional::sub(a, b);
}
inline Tensor operator-(const Tensor& a, float s) {
  return functional::sub(a, s);
}
inline Tensor operator-(float s, const Tensor& a) {
  return functional::sub(s, a);
}
inline Tensor operator-(const Tensor& a) { return functional::neg(a); }

inline Tensor operator*(const Tensor& a, const Tensor& b) {
  return functional::mul(a, b);
}
inline Tensor operator*(const Tensor& a, float s) {
  return functional::mul(a, s);
}
inline Tensor operator*(float s, const Tensor& a) {
  return functional::mul(s, a);
}

inline Tensor operator/(const Tensor& a, const Tensor& b) {
  return functional::div(a, b);
}
inline Tensor operator/(const Tensor& a, float s) {
  return functional::div(a, s);
}
inline Tensor operator/(float s, const Tensor& a) {
  return functional::div(s, a);
}

}  // namespace tensors