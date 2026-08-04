/**
 * @file nn.hpp
 * @brief Minimal neural network module framework.
 * This file provides a lightweight abstraction for building neural network
 * layers and composing them into larger models.
 * Every module exposes a forward() function and may register:
 *  - trainable parameters (Tensor objects),
 *  - child modules.
 * Parameter and child registration enables recursive traversal of a model
 * hierarchy without requiring RTTI or manual bookkeeping.
 */

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

/**
 * @brief Base class for all neural network modules.
 * A Module represents any differentiable computation, such as a linear layer,
 * activation function or container.
 * Derived classes should:
 *  - register their trainable parameters,
 *  - register child modules (if any),
 *  - implement forward().
 * Example:
 * @code
 * class ReLU : public Module {
 * public:
 *     Tensor forward(const Tensor& x) override;
 * };
 * @endcode
 */
class Module {
 private:
  /* Named child modules. */
  std::vector<std::pair<std::string, Module*>> named_children_;

  /* Named trainable parameters. */
  std::vector<std::pair<std::string, Tensor*>> named_params_;

 public:
  /**
   * @brief Virtual destructor.
   */
  virtual ~Module() = default;

  /**
   * @brief Registers trainable parameters.
   * Registration does not transfer ownership. The caller is responsible for
   * ensuring the lifetime of every Tensor exceeds that of the module.
   * Example:
   * @code
   * register_parameters({
   *     {"weight",&weight},
   *     {"bias",&bias}
   * });
   * @endcode
   * @param pairs Parameter name/pointer pairs.
   */
  void register_parameters(
      std::initializer_list<std::pair<std::string_view, Tensor*>> pairs);

  /**
   * @brief Registers child modules.
   * Registration preserves insertion order, allowing containers such as
   * Sequential to execute children deterministically.
   * Ownership is not transferred.
   * @param pairs Child name/pointer pairs.
   */
  void register_children(
      std::initializer_list<std::pair<std::string_view, Module*>> pairs);

  /**
   * @brief Returns the registered child modules.
   * @return Immutable list of named child modules.
   */
  const std::vector<std::pair<std::string, Module*>>& children() const;

  /**
   * @brief Returns the registered trainable parameters.
   * @return Immutable list of named parameters.
   */
  const std::vector<std::pair<std::string, Tensor*>>& parameters() const;

  /**
   * @brief Computes the module output.
   * Every derived module must implement this function.
   * @param x Input tensor.
   * @return Output tensor.
   */
  virtual Tensor forward(const Tensor& x) = 0;
};

/**
 * @brief Fully connected affine layer.
 * Computes
 *     y = Wx + b
 * where W has shape (out_dim, in_dim) and b has shape (out_dim).
 * The weight and bias tensors are automatically registered as trainable
 * parameters.
 */
class Linear : public Module {
 private:
  /* Weight matrix. */
  Tensor weight;

  /* Bias vector. */
  Tensor bias;

 public:
  /**
   * @brief Constructs a linear layer.
   * Storage for the weight matrix and bias vector is allocated but left
   * uninitialized.
   * @param in_dim Number of input features.
   * @param out_dim Number of output features.
   */
  Linear(size_t in_dim, size_t out_dim);

  /**
   * @brief Applies the affine transformation.
   * Computes
   *     weight × x + bias
   * @param x Input tensor.
   * @return Layer output.
   */
  Tensor forward(const Tensor& x) override;
};

class ModuleHolder {
 public:
  std::shared_ptr<Module> ptr;

  // Template constructor that accepts ANY object derived from Module by
  // value/rvalue. We use SFINAE to ensure it only captures Module derivatives
  // and doesn't hijack copy constructors.
  template <typename T, typename = std::enable_if_t<
                            std::is_base_of_v<Module, std::decay_t<T>> &&
                            !std::is_same_v<std::decay_t<T>, ModuleHolder>>>
  ModuleHolder(T&& module)
      : ptr(std::make_shared<std::decay_t<T>>(std::forward<T>(module))) {}
};

/**
 * @brief Sequential container of modules.
 * Executes each registered child module in insertion order, passing the
 * output of one module as the input to the next.
 * Example:
 * @code
 * Sequential model({
 *     {"fc1",&fc1},
 *     {"relu",&relu},
 *     {"fc2",&fc2}
 * });
 * @endcode
 */
class Sequential : public Module {
 private:
  // We store the holders here to maintain ownership and keep the modules alive.
  std::vector<std::pair<std::string, ModuleHolder>> modules_;

 public:
  /**
   * @brief Constructs a sequential container from newly constructed modules.

   * @param list Ordered sequence of named modules.
   */
  Sequential(std::initializer_list<ModuleHolder> list);

  /**
   * @brief Applies every child module in sequence.
   * Equivalent to
   * @code
   * y = m_n(...m_2(m_1(x)))
   * @endcode
   * @param x Input tensor.
   * @return Output of the final module.
   */
  inline Tensor forward(const Tensor& x);
};

/* implementations */

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

inline Sequential::Sequential(std::initializer_list<ModuleHolder> list) {
  size_t index = 0;
  for (const auto& holder : list) {
    std::string name = std::to_string(index++);

    /* Store the holder to manage the lifecycle */
    modules_.emplace_back(name, holder);

    /* Register the raw pointer with the base Module class */
    this->register_children({{name, holder.ptr.get()}});
  }
}

inline Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;
  /* Iterates through the registered children just like your original code */
  for (auto& [_, module] : this->children()) {
    out = module->forward(out);
  }
  return out;
}

}  // namespace nn
}  // namespace tensors