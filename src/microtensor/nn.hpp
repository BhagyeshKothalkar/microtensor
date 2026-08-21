#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace microtensor {

class Parameter {
 public:
  Parameter() = default;

  explicit Parameter(Tensor tensor);

  Tensor& tensor();

  const Tensor& tensor() const;

 private:
  Tensor tensor_;
};

class Module {
 public:
  virtual ~Module() = default;

  virtual Tensor forward(const Tensor& input) = 0;

  Tensor operator()(const Tensor& input);

  void register_parameter(const std::string& name, Parameter& parameter);

  void register_module(const std::string& name, Module& module);

  std::vector<Tensor*> parameters();

  void train(bool mode = true);

  void eval();

  bool training() const;

 protected:
  std::unordered_map<std::string, Parameter*> parameters_;

  std::unordered_map<std::string, Module*> modules_;

  bool training_ = true;
};

class Linear final : public Module {
 public:
  Linear(size_t in_features, size_t out_features, bool bias = true);

  Tensor forward(const Tensor& input) override;

 private:
  size_t in_features_;

  size_t out_features_;

  Parameter weight_;

  Parameter bias_;

  bool has_bias_;
};

class ReLU final : public Module {
 public:
  Tensor forward(const Tensor& input) override;
};

class GELU final : public Module {
 public:
  Tensor forward(const Tensor& input) override;
};

}  // namespace microtensor
