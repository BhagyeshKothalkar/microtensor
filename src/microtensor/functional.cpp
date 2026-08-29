#include "microtensor/functional.hpp"

#include <sched.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

namespace {

size_t tensor_offset(const Tensor& tensor, const std::vector<size_t>& indices) {
  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * tensor.stride()[i];
  }
  return offset;
}

std::vector<size_t> decode_index(size_t linear,
                                 const std::vector<size_t>& shape) {
  std::vector<size_t> indices(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    indices[i] = linear % shape[i];
    linear /= shape[i];
  }
  return indices;
}

size_t checked_index(const Tensor& indices, size_t linear, size_t upper_bound) {
  const float raw = indices.data()[linear];
  if (!std::isfinite(raw) || raw < 0.0f || raw != std::floor(raw) ||
      raw >= static_cast<float>(upper_bound)) {
    throw std::invalid_argument("indices must contain valid integer values");
  }
  return static_cast<size_t>(raw);
}

}  // namespace

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
  auto intermediate = broadcast_to_shape(out, input.shape());
  TensorIterator<float, const float>(intermediate, input)
      .for_each([](float& d, const float& s) { d += s; });
  return aligned == target ? out : out.view(target);
}
}  // namespace detail

Tensor sum(const Tensor& a, const std::vector<index_t>& dims, bool keepdims) {
  const auto normalized = detail::normalize_idx(a, dims);

  std::vector<int> reduce_dims;
  reduce_dims.reserve(normalized.size());

  std::vector<bool> reduce(a.ndim(), false);
  for (auto d : normalized) {
    reduce[d] = true;
    reduce_dims.push_back(static_cast<int>(d));
  }

  // Build output shape.
  std::vector<size_t> shape;
  if (keepdims) {
    shape = a.shape();

    for (auto d : reduce_dims) {
      shape[d] = 1;
    }
  } else {
    for (size_t i = 0; i < a.ndim(); ++i) {
      if (!reduce[i]) {
        shape.push_back(a.shape()[i]);
      }
    }
  }

  std::vector<size_t> reduced_shape;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (!reduce[i]) {
      reduced_shape.push_back(a.shape()[i]);
    }
  }

  Tensor reduced_result = Tensor::zeros(reduced_shape);

  ReductionIterator<float, float> iter(reduce_dims, reduced_result, a);

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float sum = 0.0f;

    reduce.for_each([&](const float& x) { sum += x; });

    out = sum;
  });

  Tensor result = keepdims ? reduced_result.view(shape) : reduced_result;

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);

    auto parents = make_parents(a);

    // Need keepdims shape for broadcasting backward.
    std::vector<size_t> keep = a.shape();
    for (auto d : reduce_dims) {
      keep[d] = 1;
    }

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

Tensor max(const Tensor& a, const std::vector<index_t>& dims, bool keepdims) {
  const auto normalized = detail::normalize_idx(a, dims);

  std::vector<int> reduce_dims;
  reduce_dims.reserve(normalized.size());
  std::vector<bool> reduce(a.ndim(), false);
  for (auto d : normalized) {
    reduce[d] = true;
    reduce_dims.push_back(static_cast<int>(d));
  }

  std::vector<size_t> shape;
  if (keepdims) {
    shape = a.shape();
    for (auto d : reduce_dims) {
      shape[d] = 1;
    }
  } else {
    for (size_t i = 0; i < a.ndim(); ++i) {
      if (!reduce[i]) {
        shape.push_back(a.shape()[i]);
      }
    }
  }

  std::vector<size_t> reduced_shape;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (!reduce[i]) {
      reduced_shape.push_back(a.shape()[i]);
    }
  }

  Tensor reduced_result = Tensor::zeros(reduced_shape);
  ReductionIterator<float, float> iter(reduce_dims, reduced_result, a);
  iter.for_each([](float& out, TensorIterator<const float> reduce_iter) {
    float value = -std::numeric_limits<float>::infinity();
    reduce_iter.for_each(
        [&value](const float& x) { value = std::max(value, x); });
    out = value;
  });

  Tensor result = keepdims ? reduced_result.view(shape) : reduced_result;

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    std::vector<size_t> keep = a.shape();
    for (auto d : reduce_dims) {
      keep[d] = 1;
    }

    auto backward_fn = [out = result, keep, reduce_dims](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      if (!lhs.requires_grad()) {
        return;
      }

      Tensor max_keep = out.view(keep);
      Tensor max_broadcast = broadcast_to_shape(max_keep, lhs.shape());
      Tensor grad_keep = out.grad().view(keep);
      Tensor grad_broadcast = broadcast_to_shape(grad_keep, lhs.shape());

      Tensor count_reduced = Tensor::zeros(out.shape());
      ReductionIterator<float, float, float> count_iter(
          reduce_dims, count_reduced, lhs, max_broadcast);
      count_iter.for_each(
          [](float& out_count, TensorIterator<const float, const float> it) {
            float count = 0.0f;
            it.for_each([&count](const float& value, const float& maximum) {
              count += value == maximum ? 1.0f : 0.0f;
            });
            out_count = count;
          });

      Tensor count_broadcast =
          broadcast_to_shape(count_reduced.view(keep), lhs.shape());
      Tensor grad_input(lhs.shape());
      TensorIterator<float, const float, const float, const float> grad_iter(
          grad_input, lhs, max_broadcast, grad_broadcast);
      grad_iter.for_each(
          [](float& dst, const float& value, const float& maximum,
             const float& grad) { dst = value == maximum ? grad : 0.0f; });
      TensorIterator<float, const float> normalize_iter(grad_input,
                                                        count_broadcast);
      normalize_iter.for_each(
          [](float& value, const float& count) { value /= count; });
      lhs.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

Tensor softmax(const Tensor& a, index_t dim) {
  const index_t normalized_dim = detail::normalize_idx(a, dim, a.ndim());
  Tensor maximum;
  {
    NoGradGuard guard;
    maximum = max(a, {normalized_dim}, true);
  }
  Tensor maximum_broadcast = broadcast_to_shape(maximum, a.shape());
  Tensor result(a.shape());
  TensorIterator<float, const float, const float> exp_iter(result, a,
                                                           maximum_broadcast);
  exp_iter.for_each([](float& out, const float& value, const float& maximum) {
    out = std::exp(value - maximum);
  });
  Tensor denominator = sum(result, {normalized_dim}, true);
  result = div(result, broadcast_to_shape(denominator, a.shape()));

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, normalized_dim](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      if (!lhs.requires_grad()) {
        return;
      }
      const Tensor& grad = out.grad();
      Tensor dot = sum(mul(grad, out), {normalized_dim}, true);
      Tensor grad_input =
          mul(out, sub(grad, broadcast_to_shape(dot, lhs.shape())));
      lhs.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor logsoftmax(const Tensor& a, index_t dim) {
  const index_t normalized_dim = detail::normalize_idx(a, dim, a.ndim());
  Tensor maximum;
  {
    NoGradGuard guard;
    maximum = max(a, {normalized_dim}, true);
  }
  Tensor maximum_broadcast = broadcast_to_shape(maximum, a.shape());
  Tensor exp_values(a.shape());
  TensorIterator<float, const float, const float> exp_iter(exp_values, a,
                                                           maximum_broadcast);
  exp_iter.for_each([](float& out, const float& value, const float& maximum) {
    out = std::exp(value - maximum);
  });
  Tensor log_denominator = sum(exp_values, {normalized_dim}, true);
  Tensor result(a.shape());
  Tensor log_denominator_broadcast =
      broadcast_to_shape(log_denominator, a.shape());
  TensorIterator<float, const float, const float> result_iter(
      result, a, maximum_broadcast);
  result_iter.for_each([&log_denominator_broadcast](
                           float& out, const float& value,
                           const float& maximum) { out = value - maximum; });
  TensorIterator<float, const float> subtract_log_iter(
      result, log_denominator_broadcast);
  subtract_log_iter.for_each(
      [](float& value, const float& log_sum) { value -= std::log(log_sum); });

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, normalized_dim](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      if (!lhs.requires_grad()) {
        return;
      }
      const Tensor& grad = out.grad();
      Tensor probabilities(out.shape());
      TensorIterator<float, const float> exp_iter(probabilities, out);
      exp_iter.for_each([](float& value, const float& log_value) {
        value = std::exp(log_value);
      });
      Tensor summed_grad = sum(grad, {normalized_dim}, true);
      Tensor grad_input =
          sub(grad,
              mul(probabilities, broadcast_to_shape(summed_grad, lhs.shape())));
      lhs.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor masked_fill(const Tensor& a, const Tensor& mask, float value) {
  Tensor mask_broadcast = broadcast_to_shape(mask, a.shape());
  Tensor result(a.shape());
  TensorIterator<float, const float, const float> iter(result, a,
                                                       mask_broadcast);
  iter.for_each(
      [value](float& out, const float& input, const float& mask_value) {
        out = mask_value != 0.0f ? value : input;
      });

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, mask_broadcast](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      if (!lhs.requires_grad()) {
        return;
      }
      Tensor grad_input(lhs.shape());
      TensorIterator<float, const float, const float> grad_iter(
          grad_input, out.grad(), mask_broadcast);
      grad_iter.for_each(
          [](float& dst, const float& grad, const float& mask_value) {
            dst = mask_value != 0.0f ? 0.0f : grad;
          });
      lhs.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor index_select(const Tensor& a, index_t dim, const Tensor& indices) {
  if (indices.ndim() != 1) {
    throw std::invalid_argument("index_select(): indices must be 1-D");
  }
  const index_t normalized_dim = detail::normalize_idx(a, dim, a.ndim());
  std::vector<size_t> output_shape = a.shape();
  output_shape[normalized_dim] = indices.shape()[0];
  Tensor result(output_shape);

  for (size_t linear = 0; linear < result.numel(); ++linear) {
    auto output_indices = decode_index(linear, output_shape);
    const size_t index = checked_index(indices, output_indices[normalized_dim],
                                       a.shape()[normalized_dim]);
    std::vector<size_t> input_indices = output_indices;
    input_indices[normalized_dim] = index;
    result.data()[tensor_offset(result, output_indices)] =
        a.data()[tensor_offset(a, input_indices)];
  }

  if (AutogradContext::is_enabled() && a.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(a);
    auto backward_fn = [out = result, indices,
                        normalized_dim](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      if (!lhs.requires_grad()) {
        return;
      }
      Tensor grad_input = Tensor::zeros(lhs.shape());
      for (size_t linear = 0; linear < out.numel(); ++linear) {
        auto output_indices = decode_index(linear, out.shape());
        const size_t index =
            checked_index(indices, output_indices[normalized_dim],
                          lhs.shape()[normalized_dim]);
        std::vector<size_t> input_indices = output_indices;
        input_indices[normalized_dim] = index;
        grad_input.data()[tensor_offset(grad_input, input_indices)] +=
            out.grad().data()[tensor_offset(out.grad(), output_indices)];
      }
      lhs.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor cat(const std::vector<Tensor>& tensors, index_t dim) {
  if (tensors.empty()) {
    throw std::invalid_argument("cat(): expected at least one tensor");
  }
  const index_t normalized_dim =
      detail::normalize_idx(tensors.front(), dim, tensors.front().ndim());
  std::vector<size_t> output_shape = tensors.front().shape();
  output_shape[normalized_dim] = 0;
  for (const Tensor& tensor : tensors) {
    if (tensor.ndim() != tensors.front().ndim()) {
      throw std::invalid_argument("cat(): tensor ranks must match");
    }
    for (size_t axis = 0; axis < tensor.ndim(); ++axis) {
      if (axis != static_cast<size_t>(normalized_dim) &&
          tensor.shape()[axis] != tensors.front().shape()[axis]) {
        throw std::invalid_argument(
            "cat(): non-concatenated shapes must match");
      }
    }
    output_shape[normalized_dim] += tensor.shape()[normalized_dim];
  }

  Tensor result(output_shape);
  size_t axis_offset = 0;
  for (const Tensor& tensor : tensors) {
    for (size_t linear = 0; linear < tensor.numel(); ++linear) {
      auto input_indices = decode_index(linear, tensor.shape());
      auto output_indices = input_indices;
      output_indices[normalized_dim] += axis_offset;
      result.data()[tensor_offset(result, output_indices)] =
          tensor.data()[tensor_offset(tensor, input_indices)];
    }
    axis_offset += tensor.shape()[normalized_dim];
  }

  if (AutogradContext::is_enabled()) {
    bool needs_grad = false;
    for (const Tensor& tensor : tensors) {
      needs_grad = needs_grad || tensor.requires_grad();
    }
    if (needs_grad) {
      result.set_requires_grad(true);
      auto parents = make_parents(tensors);
      auto backward_fn = [out = result, normalized_dim](const auto& parents) {
        NoGradGuard guard;
        const auto& [inputs] = parents;
        size_t axis_offset = 0;
        for (const Tensor& input : inputs) {
          if (input.requires_grad()) {
            Tensor input_grad(input.shape());
            for (size_t linear = 0; linear < input.numel(); ++linear) {
              auto input_indices = decode_index(linear, input.shape());
              auto output_indices = input_indices;
              output_indices[normalized_dim] += axis_offset;
              input_grad.data()[tensor_offset(input_grad, input_indices)] =
                  out.grad().data()[tensor_offset(out.grad(), output_indices)];
            }
            input.add_grad(input_grad);
          }
          axis_offset += input.shape()[normalized_dim];
        }
      };
      result.set_grad_fn(
          make_grad_node(std::move(parents), std::move(backward_fn)));
    }
  }
  return result;
}

Tensor layer_norm(const Tensor& a, const std::vector<size_t>& normalized_shape,
                  const Tensor& weight, const Tensor& bias, float eps) {
  if (normalized_shape.empty() || normalized_shape.size() > a.ndim()) {
    throw std::invalid_argument("layer_norm(): invalid normalized shape");
  }
  for (size_t i = 0; i < normalized_shape.size(); ++i) {
    if (normalized_shape[i] !=
        a.shape()[a.ndim() - normalized_shape.size() + i]) {
      throw std::invalid_argument("layer_norm(): normalized shape mismatch");
    }
  }
  if (weight.shape() != normalized_shape || bias.shape() != normalized_shape) {
    throw std::invalid_argument("layer_norm(): parameter shape mismatch");
  }
  if (eps < 0.0f) {
    throw std::invalid_argument("layer_norm(): eps must be non-negative");
  }
  std::vector<index_t> dims;
  for (size_t i = a.ndim() - normalized_shape.size(); i < a.ndim(); ++i) {
    dims.push_back(static_cast<index_t>(i));
  }
  Tensor mean_value = mean(a, dims, true);
  Tensor centered = sub(a, mean_value);
  Tensor variance = mean(mul(centered, centered), dims, true);
  Tensor normalized = div(centered, sqrt(add(variance, eps)));
  return add(mul(normalized, weight), bias);
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets, index_t dim) {
  const index_t normalized_dim =
      detail::normalize_idx(logits, dim, logits.ndim());
  if (static_cast<size_t>(normalized_dim) != logits.ndim() - 1 ||
      targets.shape() != std::vector<size_t>(logits.shape().begin(),
                                             logits.shape().end() - 1)) {
    throw std::invalid_argument("cross_entropy(): invalid target shape");
  }
  const size_t classes = logits.shape().back();
  const size_t rows = targets.numel();
  Tensor result({}, {0.0f});
  float total = 0.0f;
  for (size_t row = 0; row < rows; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t cls = 0; cls < classes; ++cls) {
      maximum = std::max(maximum, logits.data()[row * classes + cls]);
    }
    float denominator = 0.0f;
    for (size_t cls = 0; cls < classes; ++cls) {
      denominator += std::exp(logits.data()[row * classes + cls] - maximum);
    }
    const size_t target = checked_index(targets, row, classes);
    total += -(logits.data()[row * classes + target] - maximum -
               std::log(denominator));
  }
  result.data()[0] = total / static_cast<float>(rows);

  if (AutogradContext::is_enabled() && logits.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(logits);
    auto backward_fn = [out = result, targets, classes](const auto& parents) {
      NoGradGuard guard;
      const auto& [input] = parents;
      if (!input.requires_grad()) {
        return;
      }
      Tensor grad_input(input.shape());
      const float upstream =
          out.grad().data()[0] / static_cast<float>(targets.numel());
      for (size_t row = 0; row < targets.numel(); ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t cls = 0; cls < classes; ++cls) {
          maximum = std::max(maximum, input.data()[row * classes + cls]);
        }
        float denominator = 0.0f;
        for (size_t cls = 0; cls < classes; ++cls) {
          denominator += std::exp(input.data()[row * classes + cls] - maximum);
        }
        const size_t target = checked_index(targets, row, classes);
        for (size_t cls = 0; cls < classes; ++cls) {
          const float probability =
              std::exp(input.data()[row * classes + cls] - maximum) /
              denominator;
          grad_input.data()[row * classes + cls] =
              upstream * (probability - (cls == target ? 1.0f : 0.0f));
        }
      }
      input.add_grad(grad_input);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor mean(const Tensor& a, const std::vector<index_t>& dims, bool keepdims) {
  if (a.empty()) {
    throw std::invalid_argument("mean(): empty tensors have no mean");
  }
  float n = 1.0f;
  for (auto dim : detail::normalize_idx(a, dims)) {
    n *= static_cast<float>(a.shape()[dim]);
  }
  return mul(sum(a, dims, keepdims), 1.0f / n);
}

Tensor rmsnorm(const Tensor& a, const std::vector<index_t>& dims, float eps) {
  if (eps < 0) {
    throw std::invalid_argument("rmsnorm(): eps must be non-negative");
  }
  return div(a, sqrt(add(mean(mul(a, a), dims, true), eps)));
}

static Tensor make_matmul_view(const Tensor& t,
                               const std::vector<size_t>& domain,
                               const std::vector<size_t>& batch,
                               int zero_stride_dim) {
  if (zero_stride_dim < -3 || zero_stride_dim > -1) {
    throw std::invalid_argument("invalid matmul zero stride dimension");
  }

  std::vector<size_t> s(domain.size(), 0);
  const size_t base = batch.size();
  const size_t off = base - (t.ndim() - 2);

  for (size_t i = off; i < base; ++i) {
    s[i] = t.shape()[i - off] == 1 ? 0 : t.stride()[i - off];
  }

  const size_t m = base;
  const size_t k = base + 1;
  const size_t n = base + 2;

  switch (zero_stride_dim) {
    case -1:  // lhs: [..., m, k]
      s[m] = t.stride()[t.ndim() - 2];
      s[k] = t.stride().back();
      // s[n] remains 0
      break;

    case -3:  // rhs: [..., k, n]
      s[k] = t.stride()[t.ndim() - 2];
      s[n] = t.stride().back();
      // s[m] remains 0
      break;

    case -2:  // output: [..., m, n]
      s[m] = t.stride()[t.ndim() - 2];
      s[n] = t.stride().back();
      // s[k] remains 0
      break;
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
  auto av = make_matmul_view(a, domain, batch, -1);
  auto bv = make_matmul_view(b, domain, batch, -3);
  auto rv = make_matmul_view(result, domain, batch, -2);
  cpu_kernels::naive_matmul(av, bv, rv);
  return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  bool lhs_was_1d = (a.ndim() == 1);
  bool rhs_was_1d = (b.ndim() == 1);

  Tensor lhs = a;
  Tensor rhs = b;

  // Promote vectors to matrices.
  // (k,) -> (1,k)
  if (lhs_was_1d) {
    lhs = lhs.view({1, a.shape()[0]});
  }

  // (k,) -> (k,1)
  if (rhs_was_1d) {
    rhs = rhs.view({b.shape()[0], 1});
  }

  Tensor result = naive_matmul(lhs, rhs);

  // Restore matmul output rank.
  if (lhs_was_1d && rhs_was_1d) {
    // (1,1) -> ()
    result = result.view({});
  } else if (lhs_was_1d) {
    // (1,n) -> (n,)
    result = result.view({result.shape()[1]});
  } else if (rhs_was_1d) {
    // (m,1) -> (m,)
    result = result.view({result.shape()[0]});
  }

  if (AutogradContext::is_enabled() &&
      (a.requires_grad() || b.requires_grad())) {
    result.set_requires_grad(true);

    auto parents = make_parents(a, b);

    auto backward_fn = [out = result, lhs_was_1d,
                        rhs_was_1d](const auto& parents) {
      NoGradGuard guard;

      const auto& [lhs, rhs] = parents;
      const Tensor& grad = out.grad();

      Tensor grad2d = grad;

      // Undo output squeezing.
      //
      // scalar -> (1,1)
      // (n,) from lhs vector -> (1,n)
      // (m,) from rhs vector -> (m,1)

      if (lhs_was_1d && rhs_was_1d) {
        grad2d = grad.view({1, 1});
      } else if (lhs_was_1d) {
        grad2d = grad.view({1, grad.shape()[0]});
      } else if (rhs_was_1d) {
        grad2d = grad.view({grad.shape()[0], 1});
      }

      if (lhs.requires_grad()) {
        Tensor rhs_t = rhs.transpose(-1, -2);

        Tensor grad_lhs = matmul(grad2d, rhs_t);

        lhs.add_grad(detail::sum_to_shape(grad_lhs, lhs.shape()));
      }

      if (rhs.requires_grad()) {
        Tensor lhs_t = lhs.transpose(-1, -2);

        Tensor grad_rhs = matmul(lhs_t, grad2d);

        rhs.add_grad(detail::sum_to_shape(grad_rhs, rhs.shape()));
      }
    };

    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}

}  // namespace functional
}  // namespace tensors
