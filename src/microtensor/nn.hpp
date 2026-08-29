#pragma once

#include <initializer_list>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
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
  bool training_ = true;

  void collect_named_parameters(
      const std::string& prefix,
      std::vector<std::pair<std::string, Tensor*>>& result) const;

 protected:
  template <typename... Pairs>
    requires((std::is_same_v<Pairs, std::pair<std::string_view, Tensor*>> &&
              ...))
  void register_parameters(Pairs&&... pairs);

  template <typename... Pairs>
    requires((std::is_same_v<Pairs, std::pair<std::string_view, Module*>> &&
              ...))
  void register_children(Pairs&&... pairs);

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

  void train(bool mode = true) noexcept;
  void eval() noexcept;

  bool is_training() const noexcept { return training_; }

  virtual Tensor forward(const Tensor& x) = 0;
};

using namedparam = std::pair<std::string_view, Tensor*>;
using namedchild = std::pair<std::string_view, Module*>;

class Sequential : public Module {
 private:
  std::vector<std::unique_ptr<Module>> modules_;

 public:
  template <typename... Modules>
    requires(std::derived_from<Modules, Module> && ...)
  Sequential(std::unique_ptr<Modules>&&... ptrs);

  template <typename T, typename... Args>
    requires(std::derived_from<T, Module>)
  T& emplace(Args&&... args);

  inline Tensor forward(const Tensor& x) override;

  inline size_t size() const noexcept;
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

class Dropout : public Module {
 private:
  float probability_;

 public:
  explicit Dropout(float probability = 0.5f);
  Tensor forward(const Tensor& x) override;
};

class Embedding : public Module {
 private:
  Tensor weight_;

 public:
  Embedding(size_t num_embeddings, size_t embedding_dim);
  Tensor forward(const Tensor& x) override;

  Tensor& weight() noexcept { return weight_; }

  const Tensor& weight() const noexcept { return weight_; }
};

class LayerNorm : public Module {
 private:
  std::vector<size_t> normalized_shape_;
  float eps_;
  Tensor weight_;
  Tensor bias_;

 public:
  explicit LayerNorm(std::vector<size_t> normalized_shape, float eps = 1e-5f);
  Tensor forward(const Tensor& x) override;

  Tensor& weight() noexcept { return weight_; }

  const Tensor& weight() const noexcept { return weight_; }

  Tensor& bias() noexcept { return bias_; }

  const Tensor& bias() const noexcept { return bias_; }
};

class MultiHeadAttention : public Module {
 private:
  size_t embed_dim_;
  size_t num_heads_;
  size_t head_dim_;
  Linear query_projection_;
  Linear key_projection_;
  Linear value_projection_;
  Linear output_projection_;
  Dropout dropout_;

  Tensor forward_impl(const Tensor& query, const Tensor& context,
                      const Tensor& mask);

 public:
  MultiHeadAttention(size_t embed_dim, size_t num_heads, float dropout = 0.0f);

  Tensor forward(const Tensor& x) override;
  Tensor forward(const Tensor& query, const Tensor& context,
                 const Tensor& mask);
};

template <typename... Pairs>
  requires((std::is_same_v<Pairs, std::pair<std::string_view, Tensor*>> && ...))
void Module::register_parameters(Pairs&&... pairs) {
  (named_params_.emplace_back(pairs), ...);
}

template <typename... Pairs>
  requires((std::is_same_v<Pairs, std::pair<std::string_view, Module*>> && ...))
void Module::register_children(Pairs&&... pairs) {
  (named_children_.emplace_back(pairs), ...);
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

inline void Module::train(bool mode) noexcept {
  training_ = mode;
  for (const auto& [_, child] : named_children_) {
    if (child != nullptr) {
      child->train(mode);
    }
  }
}

inline void Module::eval() noexcept { train(false); }

inline const std::vector<std::pair<std::string, Module*>>& Module::children()
    const noexcept {
  return named_children_;
}

inline const std::vector<std::pair<std::string, Tensor*>>& Module::parameters()
    const noexcept {
  return named_params_;
}

inline Linear::Linear(size_t in_dim, size_t out_dim)
    : weight_(Tensor::zeros({in_dim, out_dim})),
      bias_(Tensor::zeros({out_dim})) {
  register_parameters(namedparam("weight", &weight_),
                      namedparam({"bias", &bias_}));
}

inline Tensor Linear::forward(const Tensor& x) {
  if (x.ndim() == 0 || x.shape().back() != weight_.shape()[0]) {
    throw std::invalid_argument(
        "Linear::forward(): input feature dimension mismatch");
  }
  return functional::add(functional::matmul(x, weight_), bias_);
}

template <typename... Modules>
  requires(std::derived_from<Modules, Module> && ...)
Sequential::Sequential(std::unique_ptr<Modules>&&... ptrs) {
  modules_.reserve(sizeof...(ptrs));
  (modules_.push_back(std::move(ptrs)), ...);
  for (auto&& [i, module] : std::ranges::views::enumerate(modules_)) {
    register_children(namedchild({std::to_string(i), module.get()}));
  }
}

template <typename T, typename... Args>
  requires(std::derived_from<T, Module>)
T& Sequential::emplace(Args&&... args) {
  auto module = std::make_unique<T>(std::forward<Args>(args)...);

  T* ptr = module.get();

  std::string name = std::to_string(modules_.size());

  modules_.push_back(std::move(module));

  register_children(namedchild({name, ptr}));

  return *ptr;
}

inline Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;

  for (const auto& module : modules_) {
    out = module->forward(out);
  }

  return out;
}

inline size_t Sequential::size() const noexcept { return modules_.size(); }

}  // namespace nn
}  // namespace tensors
