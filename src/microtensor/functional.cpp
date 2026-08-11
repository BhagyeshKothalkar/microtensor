#include "microtensor/functional.hpp"

#include <sched.h>

#include <algorithm>
#include <utility>

#include "microtensor/autograd.hpp"
#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {
namespace functional {

Tensor add(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);
  auto result = a.clone();

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

Tensor add(const Tensor& a, float s) {
  Tensor result = a.clone();
  cpu_kernels::add(result, s);

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

Tensor add(float s, const Tensor& a) { return add(a, s); }

Tensor sub(const Tensor& a, const Tensor& b) {
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

Tensor sub(const Tensor& a, float s) { return add(a, -s); }

Tensor sub(float s, const Tensor& a) { return add(neg(a), s); }

Tensor mul(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::mul(result, b_bc);

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

Tensor mul(const Tensor& a, float s) {
  Tensor result = a.clone();
  cpu_kernels::mul(result, s);

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

Tensor mul(float s, const Tensor& a) { return mul(a, s); }

Tensor div(const Tensor& a, const Tensor& b) {
  auto target_shape = get_broadcast_shape(a, b);
  Tensor result = Tensor::zeros(target_shape);
  auto [a_bc, b_bc] = broadcast_tensors(target_shape, a, b);

  TensorIterator<float, const float> it1(result, a_bc);
  while (it1.has_next()) {
    auto&& [res_val, a_val] = it1.next();
    res_val = a_val;
  }
  cpu_kernels::div(result, b_bc);

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

Tensor div(const Tensor& a, float s) { return mul(a, 1.0f / s); }

Tensor div(float s, const Tensor& a) {
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

Tensor neg(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::mul(result, -1.0f);

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

Tensor reciprocal(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::reciprocal(result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(neg(div(grad, mul(lhs, lhs))));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

Tensor sin(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::sin(result);

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

Tensor cos(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::cos(result);

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

Tensor relu(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::relu(result);

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
Tensor sqrt(const Tensor& a) {
  Tensor result = a.clone();
  cpu_kernels::sqrt(result);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(
            broadcast_to_shape(reciprocal(2 * out) * grad, lhs.shape()));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor sum(const Tensor& a) {
  Tensor result = cpu_kernels::sum(a);

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

Tensor mean(const Tensor& a) {
  Tensor result = sum(a);
  float n = static_cast<float>(a.numel());
  return mul(result, 1.0f / n);
}

static Tensor naive_matmul(const Tensor& a, const Tensor& b) {
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

Tensor matmul(const Tensor& a, const Tensor& b) {
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

}  // namespace functional
}  // namespace tensors
