#include "nn.hpp"

#include "functional.hpp"

namespace microtensor {

Parameter::Parameter(Tensor tensor) : tensor_(std::move(tensor)) {
  tensor_.requires_grad(true);
}

Tensor& Parameter::tensor() { return tensor_; }

const Tensor& Parameter::tensor() const { return tensor_; }

Tensor Module::operator()(const Tensor& input) { return forward(input); }

void Module::register_parameter(const std::string& name, Parameter& parameter) {
  parameters_[name] = &parameter;
}

void Module::register_module(const std::string& name, Module& module) {
  modules_[name] = &module;
}

std::vector<Tensor*> Module::parameters() {
  std::vector<Tensor*> result;

  for (auto& [_, parameter] : parameters_) {
    result.push_back(&parameter->tensor());
  }

  for (auto& [_, module] : modules_) {
    auto child = module->parameters();

    result.insert(result.end(), child.begin(), child.end());
  }

  return result;
}

void Module::train(bool mode) {
  training_ = mode;

  for (auto& [_, module] : modules_) {
    module->train(mode);
  }
}

void Module::eval() { train(false); }

bool Module::training() const { return training_; }

// ---------------- Linear ----------------

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {
  std::array<size_t, 2> weight_shape{out_features, in_features};

  Tensor weight = Tensor::zeros(weight_shape);

  weight_ = Parameter(std::move(weight));

  register_parameter("weight", weight_);

  if (bias) {
    std::array<size_t, 1> bias_shape{out_features};

    Tensor b = Tensor::zeros(bias_shape);

    bias_ = Parameter(std::move(b));

    register_parameter("bias", bias_);
  }
}

Tensor Linear::forward(const Tensor& input) {
  Tensor output = matmul(input, weight_.tensor());

  if (has_bias_) {
    output = add(output, bias_.tensor());
  }

  return output;
}

// ---------------- Activations ----------------

Tensor ReLU::forward(const Tensor& input) { return relu(input); }

Tensor GELU::forward(const Tensor& input) { return gelu(input); }

}  // namespace microtensor
