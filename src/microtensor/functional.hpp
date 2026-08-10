#pragma once

#include "microtensor/autograd.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {
namespace functional {

Tensor add(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const float b);
Tensor add(const float s, const Tensor& t);

Tensor sub(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const float b);
Tensor sub(const float s, const Tensor& t);

Tensor mul(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const float b);
Tensor mul(const float s, const Tensor& t);

Tensor div(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const float b);
Tensor div(const float s, const Tensor& t);

Tensor neg(const Tensor& a);
Tensor reciprocal(const Tensor& a);
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor relu(const Tensor& a);

Tensor sum(const Tensor& a);
Tensor mean(const Tensor& a);
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace functional

namespace F = functional;

inline Tensor operator+(const Tensor& a, const Tensor& b) {
  return F::add(a, b);
}
inline Tensor operator+(const Tensor& a, float s) { return F::add(a, s); }
inline Tensor operator+(float s, const Tensor& a) { return F::add(s, a); }

inline Tensor operator-(const Tensor& a, const Tensor& b) {
  return F::sub(a, b);
}
inline Tensor operator-(const Tensor& a, float s) { return F::sub(a, s); }
inline Tensor operator-(float s, const Tensor& a) { return F::sub(s, a); }

inline Tensor operator*(const Tensor& a, const Tensor& b) {
  return F::mul(a, b);
}
inline Tensor operator*(const Tensor& a, float s) { return F::mul(a, s); }
inline Tensor operator*(float s, const Tensor& a) { return F::mul(s, a); }

inline Tensor operator/(const Tensor& a, const Tensor& b) {
  return F::div(a, b);
}
inline Tensor operator/(const Tensor& a, float s) { return F::div(a, s); }
inline Tensor operator/(float s, const Tensor& a) { return F::div(s, a); }

}  // namespace tensors
