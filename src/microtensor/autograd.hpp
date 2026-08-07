#pragma once

#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensors {

class Tensor;

/**
 * @brief Thread-local state for controlling autograd graph construction.
 */
class AutogradContext {
 private:
  inline static thread_local bool enabled_ = true;

 public:
  static bool is_enabled() noexcept { return enabled_; }
  static void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
};

/**
 * @brief RAII guard to temporarily disable autograd computation graph building.
 */
class NoGradGuard {
 private:
  bool prev_state_;

 public:
  NoGradGuard() noexcept : prev_state_(AutogradContext::is_enabled()) {
    AutogradContext::set_enabled(false);
  }

  ~NoGradGuard() { AutogradContext::set_enabled(prev_state_); }

  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;
  NoGradGuard(NoGradGuard&&) = delete;
  NoGradGuard& operator=(NoGradGuard&&) = delete;
};

/**
 * @brief Abstract base class for execution graph nodes in automatic
 * differentiation.
 */
class IGradNode {
 public:
  virtual ~IGradNode() = default;

  /**
   * @brief Executes the backward pass closure for this node.
   */
  virtual void backward() = 0;

  /**
   * @brief Returns parent tensors captured by this node during the forward
   * pass.
   */
  virtual std::vector<Tensor> get_parents() const = 0;
};

namespace detail {

template <typename T>
void collect_tensor_parents(const T& item, std::vector<Tensor>& out) {
  if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
    out.push_back(item);
  }
}

template <typename Tuple, std::size_t... Is>
void collect_parents_from_tuple(const Tuple& t, std::index_sequence<Is...>,
                                std::vector<Tensor>& out) {
  (collect_tensor_parents(std::get<Is>(t), out), ...);
}

}  // namespace detail

/**
 * @brief Concrete execution node holding saved forward state and inlined
 * backward closure.
 */
template <typename Parents, typename BackwardFn>
class GradNode : public IGradNode {
 public:
  const Parents parents;
  BackwardFn backward_fn;

  GradNode(Parents p, BackwardFn fn)
      : parents(std::move(p)), backward_fn(std::move(fn)) {}

  void backward() override { backward_fn(parents); }

  std::vector<Tensor> get_parents() const override {
    std::vector<Tensor> result;
    detail::collect_parents_from_tuple(
        parents, std::make_index_sequence<std::tuple_size_v<Parents>>{},
        result);
    return result;
  }
};

/**
 * @brief Helper to construct a tuple of decay-copied parent arguments.
 */
template <typename... Args>
auto make_parents(Args&&... args) {
  return std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
}

/**
 * @brief Automatic type-deduction factory for creating shared IGradNode
 * instances.
 */
template <typename Parents, typename BackwardFn>
std::shared_ptr<IGradNode> make_grad_node(Parents&& parents,
                                          BackwardFn&& backward_fn) {
  using ParentsDecayed = std::decay_t<Parents>;
  using BackwardFnDecayed = std::decay_t<BackwardFn>;

  return std::make_shared<GradNode<ParentsDecayed, BackwardFnDecayed>>(
      std::forward<Parents>(parents), std::forward<BackwardFn>(backward_fn));
}

/**
 * @brief Autograd metadata structure associated with a differentiable Tensor.
 */
struct AutogradMeta {
  std::unique_ptr<Tensor> grad_;
  std::shared_ptr<IGradNode> grad_fn_;
  bool requires_grad_ = false;
  bool is_leaf_ = true;
};

}  // namespace tensors
