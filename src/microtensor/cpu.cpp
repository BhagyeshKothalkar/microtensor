#include "cpu.hpp"

#include <stdexcept>
#include <vector>

namespace microtensor::cpu {

namespace {

template <class F>
void visit_tensor(Tensor& output, const Tensor& input, F&& fn) {
  auto out_shape = output.shape();

  auto in_stride = input.stride();

  size_t ndim = out_shape.size();

  if (ndim == 0) {
    output.data()[0] = fn(input.data()[0]);

    return;
  }

  std::vector<size_t> index(ndim);

  size_t total = output.numel();

  for (size_t linear = 0; linear < total; ++linear) {
    size_t tmp = linear;

    size_t in_offset = input.offset();

    for (size_t d = ndim; d-- > 0;) {
      size_t coord = tmp % out_shape[d];

      tmp /= out_shape[d];

      in_offset += coord * in_stride[d];
    }

    output.data()[linear] = fn(input.storage().get()[in_offset]);
  }
}

template <class F>
void visit_binary(Tensor& output, const Tensor& lhs, const Tensor& rhs,
                  F&& fn) {
  auto shape = output.shape();

  size_t ndim = shape.size();

  std::vector<size_t> index(ndim);

  for (size_t linear = 0; linear < output.numel(); ++linear) {
    size_t tmp = linear;

    size_t lhs_offset = lhs.offset();

    size_t rhs_offset = rhs.offset();

    for (size_t d = ndim; d-- > 0;) {
      size_t coord = tmp % shape[d];

      tmp /= shape[d];

      size_t lhs_dim = lhs.ndim() - (ndim - d);

      size_t rhs_dim = rhs.ndim() - (ndim - d);

      if (lhs.ndim() > d) {
        auto ld = lhs.shape()[lhs_dim];

        if (ld != 1) {
          lhs_offset += coord * lhs.stride()[lhs_dim];
        }
      }

      if (rhs.ndim() > d) {
        auto rd = rhs.shape()[rhs_dim];

        if (rd != 1) {
          rhs_offset += coord * rhs.stride()[rhs_dim];
        }
      }
    }

    output.data()[linear] = fn(lhs.storage().get()[lhs_offset],

                               rhs.storage().get()[rhs_offset]);
  }
}

}  // namespace

void sum(Tensor& output, const Tensor& input, std::span<const size_t> dims) {
  if (!dims.empty()) {
    throw std::runtime_error("dimension reductions not implemented yet");
  }

  if (output.numel() != 1) {
    throw std::runtime_error("sum output must be scalar");
  }

  float result = 0.0f;

  for (size_t i = 0; i < input.numel(); ++i) {
    result += input.data()[i];
  }

  output.data()[0] = result;
}

void matmul(Tensor& output, const Tensor& lhs, const Tensor& rhs) {
  if (lhs.ndim() != 2 || rhs.ndim() != 2) {
    throw std::runtime_error("only 2D matmul supported");
  }

  size_t M = lhs.shape()[0];

  size_t K = lhs.shape()[1];

  size_t N = rhs.shape()[1];

  if (rhs.shape()[0] != K) {
    throw std::runtime_error("matmul dimension mismatch");
  }

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float value = 0.0f;

      for (size_t k = 0; k < K; ++k) {
        value += lhs.data()[i * K + k] * rhs.data()[k * N + j];
      }

      output.data()[i * N + j] = value;
    }
  }
}

}  // namespace microtensor::cpu
