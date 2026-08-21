#include "functional.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "autograd.hpp"
#include "cpu.hpp"
#include "microtensor/tensor.hpp"
#include "shape.hpp"

namespace microtensor {

namespace {

template <class Forward, class Backward>
Tensor unary_op(const Tensor& input, Forward&& forward, Backward&& backward) {
  Tensor output = Tensor::zeros(input.shape());

  cpu::unary(output, input, std::forward<Forward>(forward));

  autograd::record(
      output,

      [&input,
       backward = std::forward<Backward>(backward)](const Tensor& grad) {
        autograd::accumulate(const_cast<Tensor&>(input), backward(grad));
      },

      const_cast<Tensor&>(input));

  return output;
}

template <class Forward, class BackwardLhs, class BackwardRhs>
Tensor binary_op(const Tensor& lhs, const Tensor& rhs, Forward&& forward,
                 BackwardLhs&& backward_lhs, BackwardRhs&& backward_rhs) {
  std::array<const Tensor*, 2> tensors{&lhs, &rhs};

  auto shape = broadcast_shape(tensors);

  Tensor output(shape);

  cpu::binary(output, lhs, rhs, std::forward<Forward>(forward));

  autograd::record(
      output,

      [&lhs, &rhs,

       backward_lhs = std::forward<BackwardLhs>(backward_lhs),

       backward_rhs =
           std::forward<BackwardRhs>(backward_rhs)](const Tensor& grad) {
        autograd::accumulate(const_cast<Tensor&>(lhs),
                             backward_lhs(grad, lhs, rhs));

        autograd::accumulate(const_cast<Tensor&>(rhs),
                             backward_rhs(grad, lhs, rhs));
      },

      const_cast<Tensor&>(lhs), const_cast<Tensor&>(rhs));

  return output;
}

}  // namespace

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  return binary_op(
      lhs, rhs,

      [](float a, float b) { return a + b; },

      [](const Tensor& grad, const Tensor&, const Tensor&) { return grad; },

      [](const Tensor& grad, const Tensor&, const Tensor&) { return grad; });
}

Tensor add(const Tensor& lhs, float rhs) {
  std::array<size_t, 0> shape{};

  Tensor scalar = Tensor::full(shape, rhs);

  return add(lhs, scalar);
}

Tensor sub(const Tensor& lhs, const Tensor& rhs) {
  return binary_op(
      lhs, rhs,

      [](float a, float b) { return a - b; },

      [](const Tensor& grad, const Tensor&, const Tensor&) { return grad; },

      [](const Tensor& grad, const Tensor&, const Tensor&) {
        return neg(grad);
      });
}

Tensor mul(const Tensor& lhs, const Tensor& rhs) {
  return binary_op(
      lhs, rhs,

      [](float a, float b) { return a * b; },

      [](const Tensor& grad, const Tensor&, const Tensor& rhs) {
        return mul(grad, rhs);
      },

      [](const Tensor& grad, const Tensor& lhs, const Tensor&) {
        return mul(grad, lhs);
      });
}

Tensor div(const Tensor& lhs, const Tensor& rhs) {
  return binary_op(
      lhs, rhs,

      [](float a, float b) { return a / b; },

      [](const Tensor& grad, const Tensor&, const Tensor& rhs) {
        return div(grad, rhs);
      },

      [](const Tensor& grad, const Tensor& lhs, const Tensor& rhs) {
        return neg(div(mul(grad, lhs), mul(rhs, rhs)));
      });
}

Tensor neg(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return -x; },

      [](const Tensor& grad) { return neg(grad); });
}

Tensor reciprocal(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return 1.0f / x; },

      [](const Tensor& grad) { return neg(grad); });
}

Tensor sin(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return std::sin(x); },

      [](const Tensor& grad) { return grad; });
}

Tensor cos(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return std::cos(x); },

      [](const Tensor& grad) { return grad; });
}

Tensor relu(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return x > 0 ? x : 0; },

      [](const Tensor& grad) { return grad; });
}

Tensor sqrt(const Tensor& input) {
  return unary_op(
      input,

      [](float x) { return std::sqrt(x); },

      [](const Tensor& grad) { return grad; });
}

Tensor gelu(const Tensor& input) {
  return unary_op(
      input,

      [](float x) {
        return 0.5f * x *
               (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
      },

      [](const Tensor& grad) { return grad; });
}

Tensor sum(const Tensor& input, std::span<const size_t> dims) {
  if (!dims.empty()) {
    throw std::runtime_error("dimension reduction pending");
  }

  Tensor output(std::array<size_t, 0>{});

  cpu::sum(output, input, dims);

  autograd::record(
      output,
      [&input](const Tensor& grad_output) {
        Tensor grad = Tensor::zeros(input.shape());

        for (size_t i = 0; i < grad.numel(); ++i) {
          grad.data()[i] = grad_output.data()[0];
        }

        autograd::accumulate(const_cast<Tensor&>(input), grad);
      },
      const_cast<Tensor&>(input));

  return output;
}

Tensor mean(const Tensor& input, std::span<const size_t> dims) {
  Tensor result = sum(input, dims);

  std::array<size_t, 0> shape{};

  Tensor scale = Tensor::full(shape, static_cast<float>(input.numel()));

  return div(result, scale);
}

Tensor rmsnorm(const Tensor& input, std::span<const size_t> dims) {
  Tensor squared = mul(input, input);

  Tensor mean_square = mean(squared, dims);

  return div(input, sqrt(mean_square));
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  std::array<size_t, 2> shape{lhs.shape()[0], rhs.shape()[1]};

  Tensor output(shape);

  cpu::matmul(output, lhs, rhs);

  return output;
}

}  // namespace microtensor
