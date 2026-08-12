#pragma once

#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {
namespace nn {

class Module {
 private:
  std::vector<std::pair<std::string, Module*>> named_children_;
  std::vector<std::pair<std::string, Tensor*>> named_params_;

  void collect_named_parameters(
      const std::string& prefix,
      std::vector<std::pair<std::string, Tensor*>>& result) const;

 protected:
  void register_parameters(
      std::initializer_list<std::pair<std::string_view, Tensor*>> pairs);

  void register_children(
      std::initializer_list<std::pair<std::string_view, Module*>> pairs);

 public:
  Module() = default;

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;

  virtual ~Module() = default;

  const std::vector<std::pair<std::string, Module*>>& children() const noexcept;
  const std::vector<std::pair<std::string, Tensor*>>& parameters()
      const noexcept;

  // Depth-first, registration-order traversal.
  std::vector<std::pair<std::string, Tensor*>> named_parameters_recursive()
      const;

  std::vector<Tensor*> parameters_recursive() const;

  virtual Tensor forward(const Tensor& x) = 0;
};

class Linear : public Module {
 private:
  Tensor weight_;
  Tensor bias_;

 public:
  Linear(size_t in_dim, size_t out_dim);

  Tensor forward(const Tensor& x) override;

  Tensor& weight() noexcept { return weight_; }
  const Tensor& weight() const noexcept { return weight_; }

  Tensor& bias() noexcept { return bias_; }
  const Tensor& bias() const noexcept { return bias_; }
};

class ReLU : public Module {
 public:
  Tensor forward(const Tensor& x) override { return functional::relu(x); }
};

class GELU : public Module {
 public:
  Tensor forward(const Tensor& x) override { return functional::gelu(x); }
};

class ModuleHolder {
 public:
  std::shared_ptr<Module> ptr;

  template <typename T, typename = std::enable_if_t<
                            std::is_base_of_v<Module, std::decay_t<T>> &&
                            !std::is_same_v<Module, std::decay_t<T>>>>
  explicit ModuleHolder(T&& module)
      : ptr(std::make_shared<std::decay_t<T>>(std::forward<T>(module))) {}

  explicit ModuleHolder(std::shared_ptr<Module> module)
      : ptr(std::move(module)) {}

  ModuleHolder(const ModuleHolder&) = default;
  ModuleHolder& operator=(const ModuleHolder&) = default;

  ModuleHolder(ModuleHolder&&) noexcept = default;
  ModuleHolder& operator=(ModuleHolder&&) noexcept = default;

  Module& operator*() noexcept { return *ptr; }
  const Module& operator*() const noexcept { return *ptr; }

  Module* operator->() noexcept { return ptr.get(); }
  const Module* operator->() const noexcept { return ptr.get(); }

  explicit operator bool() const noexcept { return static_cast<bool>(ptr); }
};

class Sequential : public Module {
 private:
  std::vector<std::pair<std::string, ModuleHolder>> modules_;

 public:
  Sequential(std::initializer_list<ModuleHolder> modules);

  Sequential(const Sequential&) = delete;
  Sequential& operator=(const Sequential&) = delete;

  Sequential(Sequential&&) = delete;
  Sequential& operator=(Sequential&&) = delete;

  Tensor forward(const Tensor& x) override;

  const std::vector<std::pair<std::string, ModuleHolder>>& modules()
      const noexcept {
    return modules_;
  }
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

inline void Module::collect_named_parameters(
    const std::string& prefix,
    std::vector<std::pair<std::string, Tensor*>>& result) const {
  // Parameters belonging directly to this module.
  for (const auto& [name, param] : named_params_) {
    std::string full_name;

    if (prefix.empty()) {
      full_name = name;
    } else {
      full_name = prefix + "." + name;
    }

    result.emplace_back(std::move(full_name), param);
  }

  // Then recursively visit children in registration order.
  for (const auto& [name, child] : named_children_) {
    if (child == nullptr) {
      continue;
    }

    std::string child_prefix;

    if (prefix.empty()) {
      child_prefix = name;
    } else {
      child_prefix = prefix + "." + name;
    }

    child->collect_named_parameters(child_prefix, result);
  }
}

inline std::vector<std::pair<std::string, Tensor*>>
Module::named_parameters_recursive() const {
  std::vector<std::pair<std::string, Tensor*>> result;

  collect_named_parameters("", result);

  return result;
}

inline std::vector<Tensor*> Module::parameters_recursive() const {
  std::vector<Tensor*> result;

  for (const auto& [_, param] : named_parameters_recursive()) {
    result.push_back(param);
  }

  return result;
}

inline const std::vector<std::pair<std::string, Module*>>& Module::children()
    const noexcept {
  return named_children_;
}

inline const std::vector<std::pair<std::string, Tensor*>>& Module::parameters()
    const noexcept {
  return named_params_;
}

inline Linear::Linear(size_t in_dim, size_t out_dim)
    : weight_(Tensor::zeros({in_dim, out_dim})), bias_(Tensor::zeros({out_dim})) {
  register_parameters({
      {"weight", &weight_},
      {"bias", &bias_},
  });
}

inline Tensor Linear::forward(const Tensor& x) {
  if (x.ndim() == 0 || x.shape().back() != weight_.shape()[0]) {
    throw std::invalid_argument("Linear::forward(): input feature dimension mismatch");
  }
  return functional::add(functional::matmul(x, weight_), bias_);
}

inline Sequential::Sequential(std::initializer_list<ModuleHolder> modules) {
  size_t index = 0;

  for (const auto& holder : modules) {
    if (!holder.ptr) {
      continue;
    }

    std::string name = std::to_string(index++);

    modules_.emplace_back(name, holder);

    register_children({
        {name, modules_.back().second.ptr.get()},
    });
  }
}

inline Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;

  for (auto& [_, holder] : modules_) {
    out = holder->forward(out);
  }

  return out;
}

}  // namespace nn
}  // namespace tensors
