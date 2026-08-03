#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {
namespace nn {

class Module {
 private:
  std::vector<std::pair<std::string, Module*>> named_children_;
  std::vector<std::pair<std::string, Tensor*>> named_params_;

 public:
  virtual ~Module() = default;

  void register_parameters(
      std::initializer_list<std::pair<std::string_view, Tensor*>> pairs);

  void register_children(
      std::initializer_list<std::pair<std::string_view, Module*>> pairs);

  const std::vector<std::pair<std::string, Module*>>& children() const;
  const std::vector<std::pair<std::string, Tensor*>>& parameters() const;

  virtual Tensor forward(const Tensor& x) = 0;
};

class Linear : public Module {
 private:
  Tensor weight;
  Tensor bias;

 public:
  Linear(size_t in_dim, size_t out_dim);

  Tensor forward(const Tensor& x) override;
};

class Sequential : public Module {
 private:
  std::vector<std::pair<std::string, Module*>> modules_;

 public:
  Sequential(std::initializer_list<std::pair<std::string_view, Module*>> list);

  Tensor forward(const Tensor& x) override;
};

inline void Module::register_parameters(
    std::initializer_list<std::pair<std::string_view, Tensor*>> pairs) {
  for (const auto& [name, param] : pairs) {
    named_params_.emplace_back(std::string(name), param);
  }
}

inline void Module::register_children(
    std::initializer_list<std::pair<std::string_view, Module*>> pairs) {
  for (const auto& [name, child] : pairs) {
    named_children_.emplace_back(std::string(name), child);
  }
}

inline const std::vector<std::pair<std::string, Module*>>& Module::children()
    const {
  return named_children_;
}

inline const std::vector<std::pair<std::string, Tensor*>>& Module::parameters()
    const {
  return named_params_;
}

inline Linear::Linear(size_t in_dim, size_t out_dim)
    : weight({out_dim, in_dim}), bias({out_dim}) {
  register_parameters({{"weight", &weight}, {"bias", &bias}});
}

inline Tensor Linear::forward(const Tensor& x) {
  return add(naive_matmul(weight, x), bias);
}

// construct sequential, the children are already registered in order
inline Sequential::Sequential(
    std::initializer_list<std::pair<std::string_view, Module*>> list) {
  this->register_children(list);
}

// call forward from each children in order
inline Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;
  for (auto& [_, module] : this->children()) {
    out = module->forward(out);
  }
  return out;
}

}  // namespace nn
}  // namespace tensors