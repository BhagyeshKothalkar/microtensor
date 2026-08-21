#pragma once

#include <concepts>
#include <vector>

#include "tensor.hpp"

namespace microtensor::cpu {

template <class F>
  requires std::invocable<F&, float>
void unary(Tensor& output, const Tensor& input, F&& operation) {
  for (size_t i = 0; i < input.numel(); ++i) {
    output.data()[i] = operation(input.data()[i]);
  }
}

template <class F>
  requires std::invocable<F&, float, float>
void binary(Tensor& output, const Tensor& lhs, const Tensor& rhs,
            F&& operation) {
  for (size_t i = 0; i < output.numel(); ++i) {
    output.data()[i] = operation(lhs.data()[i], rhs.data()[i]);
  }
}

void sum(Tensor& output, const Tensor& input, std::span<const size_t> dims);

void matmul(Tensor& output, const Tensor& lhs, const Tensor& rhs);

void copy(Tensor& output, const Tensor& input);

}  // namespace microtensor::cpu
