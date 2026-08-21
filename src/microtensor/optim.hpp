#pragma once

#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace microtensor {

class Optimizer {
 public:
  explicit Optimizer(std::vector<Tensor*> parameters);

  virtual ~Optimizer() = default;

  virtual void step() = 0;

  void zero_grad();

 protected:
  std::vector<Tensor*> parameters_;
};

class SGD final : public Optimizer {
 public:
  SGD(std::vector<Tensor*> parameters, float learning_rate,
      float momentum = 0.0f);

  void step() override;

 private:
  float lr_;

  float momentum_;

  std::unordered_map<Tensor*, Tensor> velocity_;
};

class Adam final : public Optimizer {
 public:
  Adam(std::vector<Tensor*> parameters, float learning_rate = 1e-3f,
       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

  void step() override;

 private:
  float lr_;

  float beta1_;

  float beta2_;

  float eps_;

  size_t step_ = 0;

  std::unordered_map<Tensor*, Tensor> m_;

  std::unordered_map<Tensor*, Tensor> v_;
};

}  // namespace microtensor
