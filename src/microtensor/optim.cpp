#include "optim.hpp"

#include <cmath>

namespace microtensor {

Optimizer::Optimizer(std::vector<Tensor*> parameters)
    : parameters_(std::move(parameters)) {}

void Optimizer::zero_grad() {
  for (auto* parameter : parameters_) {
    auto* grad = parameter->grad();

    if (!grad) {
      continue;
    }

    for (size_t i = 0; i < grad->numel(); ++i) {
      const_cast<Tensor*>(grad)->data()[i] = 0.0f;
    }
  }
}

// ---------------- SGD ----------------

SGD::SGD(std::vector<Tensor*> parameters, float learning_rate, float momentum)
    : Optimizer(std::move(parameters)),
      lr_(learning_rate),
      momentum_(momentum) {}

void SGD::step() {
  for (auto* parameter : parameters_) {
    auto* grad = parameter->grad();

    if (!grad) {
      continue;
    }

    Tensor update = Tensor::zeros(parameter->shape());

    if (momentum_ != 0.0f) {
      auto& velocity = velocity_[parameter];

      if (!velocity.storage()) {
        velocity = Tensor::zeros(parameter->shape());
      }

      for (size_t i = 0; i < grad->numel(); ++i) {
        velocity.data()[i] = momentum_ * velocity.data()[i] + grad->data()[i];

        update.data()[i] = velocity.data()[i];
      }
    } else {
      for (size_t i = 0; i < grad->numel(); ++i) {
        update.data()[i] = grad->data()[i];
      }
    }

    for (size_t i = 0; i < parameter->numel(); ++i) {
      parameter->data()[i] -= lr_ * update.data()[i];
    }
  }
}

// ---------------- Adam ----------------

Adam::Adam(std::vector<Tensor*> parameters, float learning_rate, float beta1,
           float beta2, float eps)
    : Optimizer(std::move(parameters)),
      lr_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps) {}

void Adam::step() {
  step_++;

  for (auto* parameter : parameters_) {
    auto* grad = parameter->grad();

    if (!grad) {
      continue;
    }

    auto& m = m_[parameter];

    auto& v = v_[parameter];

    if (!m.storage()) {
      m = Tensor::zeros(parameter->shape());

      v = Tensor::zeros(parameter->shape());
    }

    for (size_t i = 0; i < parameter->numel(); ++i) {
      float g = grad->data()[i];

      m.data()[i] = beta1_ * m.data()[i] + (1 - beta1_) * g;

      v.data()[i] = beta2_ * v.data()[i] + (1 - beta2_) * g * g;

      float m_hat = m.data()[i] / (1 - std::pow(beta1_, step_));

      float v_hat = v.data()[i] / (1 - std::pow(beta2_, step_));

      parameter->data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
    }
  }
}

}  // namespace microtensor
