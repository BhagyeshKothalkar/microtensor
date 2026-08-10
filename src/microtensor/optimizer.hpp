#include <ranges>
#include <vector>

#include "microtensor/autograd.hpp"
#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {
namespace optim {
class Optimizer {
 protected:
  std::vector<Tensor> parameters_;
  Optimizer(std::vector<Tensor>& tensors)
      : parameters_(tensors.begin(), tensors.end()) {}

 public:
  virtual void step() = 0;
  void zero_grad() {
    for (auto& p : parameters_) {
      auto& grad = p.mutable_grad();

      // for the lack of anything at all that does what i wanted.
      // maybe for now
      // TODO: this is just running away form implementing a method. this has to
      // be fixed in tensor.
      auto it = TensorIterator<float>(grad);
      while (it.has_next()) {
        auto [v] = it.next();
        v = 0;
      }
    }
  }
};

class sgd : public Optimizer {
  float lr_;

 public:
  sgd(std::vector<Tensor>& tensors, float lr) : Optimizer(tensors), lr_(lr) {}
  void step() {
    for (Tensor& t : parameters_) {
      NoGradGuard guard;
      if (t.requires_grad()) {
        functional::sub_(t, (t.grad()) * lr_);
      }
    }
  }
};

class adam : public Optimizer {
  float lr_, beta1_, beta2_, eps_;
  std::vector<Tensor> momentum_1_, momentum_2_;
  float beta1_pow = 1, beta2_pow = 1;

 public:
  adam(std::vector<Tensor>& tensors, float lr, float beta1, float beta2,
       float eps = 1e-4)
      : Optimizer(tensors),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        momentum_1_(tensors.size()),
        momentum_2_(tensors.size()) {
    auto v = std::views::zip(std::views::as_const(parameters_), momentum_1_,
                             momentum_2_);
    for (auto&& [t, m1, m2] : v) {
      m1 = Tensor::zeros_like(t);
      m2 = Tensor::zeros_like(t);
    }
  }

  void step() {
    namespace F = functional;
    NoGradGuard guard;

    beta1_pow *= beta1_;
    beta2_pow *= beta2_;

    auto v = std::views::zip(parameters_, momentum_1_, momentum_2_);
    for (auto&& [t, m1, m2] : v) {
      if (t.requires_grad()) {
        F::add_(F::scalar_multiply_(m1, beta1_), (1 - beta1_) * t.grad());
        F::add_(F::scalar_multiply_(m2, beta2_), (1 - beta2_) * t.grad());
        auto m1_corr = m1 / (1 - beta1_pow);
        auto m2_corr = m2 / (1 - beta2_pow);
        F::sub_(t, lr_ * (m1_corr / (/*F::sqrt*/ (m2_corr) + eps_)));
      }
    }
  }
};

}  // namespace optim
}  // namespace tensors
