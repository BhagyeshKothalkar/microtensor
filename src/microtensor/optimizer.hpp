#pragma once

#include <cstddef>
#include <vector>

#include "microtensor/autograd.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {
namespace optim {

class Optimizer {
 protected:
  std::vector<Tensor*> parameters_;

  explicit Optimizer(const std::vector<Tensor*>& tensors)
      : parameters_(tensors) {}

 public:
  virtual ~Optimizer() = default;

  Optimizer(const Optimizer&) = default;
  Optimizer& operator=(const Optimizer&) = default;

  virtual void step() = 0;

  void zero_grad() {
    NoGradGuard guard;

    for (Tensor* p : parameters_) {
      if (p == nullptr || !p->requires_grad()) {
        continue;
      }

      Tensor& grad = p->mutable_grad();

      auto it = TensorIterator<float>(grad);

      while (it.has_next()) {
        auto [v] = it.next();
        v = 0.0f;
      }
    }
  }
};

class SGD : public Optimizer {
 private:
  float lr_;

 public:
  SGD(const std::vector<Tensor*>& tensors, float lr)
      : Optimizer(tensors), lr_(lr) {}

  void step() override {
    NoGradGuard guard;

    for (Tensor* p : parameters_) {
      if (p == nullptr || !p->requires_grad()) {
        continue;
      }

      Tensor scaled_grad = functional::mul(p->grad(), lr_);

      cpu_kernels::sub(*p, scaled_grad);
    }
  }
};

class Adam : public Optimizer {
 private:
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;

  std::vector<Tensor> momentum_1_;
  std::vector<Tensor> momentum_2_;

  float beta1_pow_ = 1.0f;
  float beta2_pow_ = 1.0f;

 public:
  Adam(const std::vector<Tensor*>& tensors, float lr, float beta1, float beta2,
       float eps = 1e-8f)
      : Optimizer(tensors),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        momentum_1_(tensors.size()),
        momentum_2_(tensors.size()) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      Tensor* p = parameters_[i];

      if (p == nullptr) {
        continue;
      }

      momentum_1_[i] = Tensor::zeros_like(*p);
      momentum_2_[i] = Tensor::zeros_like(*p);
    }
  }

  void step() override {
    NoGradGuard guard;

    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;

    const float bias_correction_1 = 1.0f - beta1_pow_;
    const float bias_correction_2 = 1.0f - beta2_pow_;

    for (size_t i = 0; i < parameters_.size(); ++i) {
      Tensor* p = parameters_[i];

      if (p == nullptr || !p->requires_grad()) {
        continue;
      }
      Tensor& m1 = momentum_1_[i];
      Tensor& m2 = momentum_2_[i];
      const Tensor& grad = p->grad();

      cpu_kernels::mul(m1, beta1_);
      Tensor grad_m1 = functional::mul(grad, 1.0f - beta1_);
      cpu_kernels::add(m1, grad_m1);
      cpu_kernels::mul(m2, beta2_);
      Tensor grad_squared = functional::mul(grad, grad);
      Tensor grad_m2 = functional::mul(grad_squared, 1.0f - beta2_);
      cpu_kernels::add(m2, grad_m2);
      Tensor m1_corr = functional::div(m1, bias_correction_1);
      Tensor m2_corr = functional::div(m2, bias_correction_2);
      Tensor denominator = functional::add(m2_corr, eps_);
      Tensor update = functional::div(m1_corr, denominator);
      Tensor scaled_update = functional::mul(update, lr_);
      cpu_kernels::sub(*p, scaled_update);
    }
  }
};

}  // namespace optim
}  // namespace tensors
