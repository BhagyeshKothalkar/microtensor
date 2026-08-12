#include "microtensor/functional.hpp"

#include <sched.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
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
  Tensor result = Tensor::zeros(target_shape);

  TensorIterator<float, const float>(result, a_bc)
      .for_each([](float& dst, const float& src) { dst = src; });

  cpu_kernels::add(result, b_bc);

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);
    auto parents = make_parents(a, b);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad);
      }
      if (rhs.requires_grad()) {
        rhs.add_grad(grad);
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(grad);
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(grad);
      }
      if (rhs.requires_grad()) {
        rhs.add_grad(neg(grad));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(mul(grad, rhs));
      }
      if (rhs.requires_grad()) {
        rhs.add_grad(mul(grad, lhs));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(mul(grad, s));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(div(grad, rhs));
      }
      if (rhs.requires_grad()) {
        rhs.add_grad(neg(div(mul(grad, lhs), mul(rhs, rhs))));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(neg(grad));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(mul(grad, cos(lhs)));
      }
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
      if (lhs.requires_grad()) {
        lhs.add_grad(neg(mul(grad, sin(lhs))));
      }
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

Tensor gelu(const Tensor& a) {
  Tensor result(a.shape());

  TensorIterator<float, const float>(result, a).for_each(
      [](float& y, const float& x) {
        y = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
      });

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);

    auto backward = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [x] = parents;
      if (x.requires_grad()) {
        Tensor g(x.shape());
        TensorIterator<float, const float, const float>(g, out.grad(), x)
            .for_each([](float& dst, const float& dy, const float& v) {
              const float cdf = 0.5f * (1.0f + std::erf(v / std::sqrt(2.0f)));
              const float pdf = std::exp(-0.5f * v * v) /
                                std::sqrt(2.0f * std::numbers::pi_v<float>);
              dst = dy * (cdf + v * pdf);
            });
        x.add_grad(g);
      }
    };

    result.set_grad_fn(make_grad_node(std::move(parents), std::move(backward)));
  }
  return result;
}

namespace detail {
index_t normalize_idx(const Tensor& tensor, index_t idx, size_t) {
  if (idx < 0) {
    idx += static_cast<index_t>(tensor.ndim());
  }
  if (idx < 0 || idx >= static_cast<index_t>(tensor.ndim())) {
    throw std::out_of_range("dimension is out of range");
  }
  return idx;
}

std::vector<index_t> normalize_idx(const Tensor& tensor,
                                   const std::vector<index_t>& idx) {
  std::vector<index_t> out;

  for (auto dim : idx) {
    dim = normalize_idx(tensor, dim, 0);
    if (std::find(out.begin(), out.end(), dim) != out.end()) {
      throw std::invalid_argument("duplicate reduction dimension");
    }
    out.push_back(dim);
  }

  return out;
}

Tensor sum_to_shape(const Tensor& input, const std::vector<size_t>& target) {
  if (input.shape() == target) {
    return input;
  }
  if (target.size() > input.ndim()) {
    throw std::invalid_argument(
        "sum_to_shape(): target rank cannot exceed input rank");
  }

  std::vector<size_t> aligned = target;
  aligned.insert(aligned.begin(), input.ndim() - target.size(), 1);
  for (size_t i = 0; i < input.ndim(); ++i) {
    if (aligned[i] != 1 && aligned[i] != input.shape()[i]) {
      throw std::invalid_argument("sum_to_shape(): incompatible shapes");
    }
  }

  Tensor out = Tensor::zeros(aligned);
  TensorIterator<float, const float>(broadcast_to_shape(out, input.shape()),
                                     input)
      .for_each([](float& d, const float& s) { d += s; });
  return aligned == target ? out : out.view(target);
}
}  // namespace detail

Tensor sum(const Tensor& a) {
  std::vector<index_t> dims;
  for (size_t i = 0; i < a.ndim(); ++i) {
    dims.push_back(static_cast<index_t>(i));
  }
  return sum(a, dims);
}

Tensor sum(const Tensor& a, const std::vector<index_t>& dims) {
  const auto normalized = detail::normalize_idx(a, dims);
  std::vector<bool> reduce(a.ndim(), false);

  for (auto d : normalized) {
    reduce[d] = true;
  }

  std::vector<size_t> keep = a.shape();
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (reduce[i]) {
      keep[i] = 1;
    }
  }

  Tensor kept = Tensor::zeros(keep);
  TensorIterator<float, const float>(broadcast_to_shape(kept, a.shape()), a)
      .for_each([](float& d, const float& s) { d += s; });

  std::vector<size_t> shape;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (!reduce[i]) {
      shape.push_back(a.shape()[i]);
    }
  }

  Tensor result = kept.view(shape);

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, keep](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(broadcast_to_shape(grad.view(keep), lhs.shape()));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

Tensor mean(const Tensor& a) {
  std::vector<index_t> dims;
  for (size_t i = 0; i < a.ndim(); ++i) {
    dims.push_back(static_cast<index_t>(i));
  }
  return mean(a, dims);
}

Tensor mean(const Tensor& a, const std::vector<index_t>& dims) {
  if (a.empty()) {
    throw std::invalid_argument("mean(): empty tensors have no mean");
  }
  float n = 1.0f;
  for (auto dim : detail::normalize_idx(a, dims)) {
    n *= static_cast<float>(a.shape()[dim]);
  }
  return mul(sum(a, dims), 1.0f / n);
}

Tensor rmsnorm(const Tensor& a, const std::vector<index_t>& dims, float eps) {
  if (eps < 0) {
    throw std::invalid_argument("rmsnorm(): eps must be non-negative");
  }
  return div(a, sqrt(add(mean(mul(a, a), dims), eps)));
}

static Tensor make_matmul_view(const Tensor& t,
                               const std::vector<size_t>& domain,
                               const std::vector<size_t>& batch, bool lhs) {
  std::vector<size_t> s(domain.size(), 0);
  const size_t base = batch.size(), off = base - (t.ndim() - 2);

  for (size_t i = off; i < base; ++i) {
    s[i] = t.shape()[i - off] == 1 ? 0 : t.stride()[i - off];
  }

  if (lhs) {
    s[base] = t.stride()[t.ndim() - 2];
    s[base + 1] = t.stride().back();
  } else {
    s[base + 1] = t.stride()[t.ndim() - 2];
    s[base + 2] = t.stride().back();
  }

  return Tensor(domain, s, t.storage(), t.storage_size(), t.offset());
}

static Tensor naive_matmul(const Tensor& a, const Tensor& b) {
  if (a.ndim() < 2 || b.ndim() < 2 ||
      a.shape().back() != b.shape()[b.ndim() - 2]) {
    throw std::invalid_argument("matmul(): incompatible matrix shapes");
  }

  const size_t m = a.shape()[a.ndim() - 2], k = a.shape().back(),
               n = b.shape().back();
  std::vector<size_t> ab(a.shape().begin(), a.shape().end() - 2);
  std::vector<size_t> bb(b.shape().begin(), b.shape().end() - 2);
  Tensor aa(ab), bbv(bb);
  auto batch = get_broadcast_shape(aa, bbv);

  std::vector<size_t> out = batch;
  out.insert(out.end(), {m, n});
  Tensor result = Tensor::zeros(out);

  std::vector<size_t> domain = batch;
  domain.insert(domain.end(), {m, k, n});
  auto av = make_matmul_view(a, domain, batch, true);
  auto bv = make_matmul_view(b, domain, batch, false);
  auto rv = make_matmul_view(result, domain, batch, false);
  cpu_kernels::naive_matmul(av, bv, rv);
  return result;
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
        lhs.add_grad(detail::sum_to_shape(matmul(grad, rhs.transpose(-1, -2)),
                                          lhs.shape()));
      }
      if (rhs.requires_grad()) {
        rhs.add_grad(detail::sum_to_shape(matmul(lhs.transpose(-1, -2), grad),
                                          rhs.shape()));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

}  // namespace functional
}  // namespace tensors
