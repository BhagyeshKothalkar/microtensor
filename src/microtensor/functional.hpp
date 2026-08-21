#pragma once

#include <concepts>
#include <span>

#include "tensor.hpp"

namespace microtensor {

Tensor add(const Tensor& lhs, const Tensor& rhs);

Tensor add(const Tensor& lhs, float rhs);

Tensor sub(const Tensor& lhs, const Tensor& rhs);

Tensor mul(const Tensor& lhs, const Tensor& rhs);

Tensor div(const Tensor& lhs, const Tensor& rhs);

Tensor neg(const Tensor& input);

Tensor reciprocal(const Tensor& input);

Tensor sin(const Tensor& input);

Tensor cos(const Tensor& input);

Tensor relu(const Tensor& input);

Tensor sqrt(const Tensor& input);

Tensor gelu(const Tensor& input);

Tensor sum(const Tensor& input, std::span<const size_t> dims = {});

Tensor mean(const Tensor& input, std::span<const size_t> dims = {});

Tensor rmsnorm(const Tensor& input, std::span<const size_t> dims = {});

Tensor matmul(const Tensor& lhs, const Tensor& rhs);

}  // namespace microtensor
