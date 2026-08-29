#pragma once

#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensors {

class Tensor;

class AutogradContext {
 private:
  inline static thread_local bool enabled_ = true;

 public:
  static bool is_enabled() noexcept { return enabled_; }

  static void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
};

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

class IGradNode {
 public:
  virtual ~IGradNode() = default;

  virtual void backward() = 0;

  virtual void for_each_parent(
      const std::function<void(const Tensor&)>& callback) const = 0;
};

namespace detail {

template <typename T>
void for_each_tensor_parent(
    const T& item, const std::function<void(const Tensor&)>& callback) {
  if constexpr (std::is_same_v<std::decay_t<T>, Tensor>) {
    callback(item);
  } else if constexpr (std::is_same_v<std::decay_t<T>, std::vector<Tensor>>) {
    for (const Tensor& tensor : item) {
      callback(tensor);
    }
  }
}

template <typename Tuple, std::size_t... Is>
void for_each_parent_in_tuple(
    const Tuple& parents, std::index_sequence<Is...>,
    const std::function<void(const Tensor&)>& callback) {
  (for_each_tensor_parent(std::get<Is>(parents), callback), ...);
}

}  // namespace detail

template <typename Parents, typename BackwardFn>
class GradNode : public IGradNode {
 public:
  const Parents parents;
  BackwardFn backward_fn;

  GradNode(Parents p, BackwardFn fn)
      : parents(std::move(p)), backward_fn(std::move(fn)) {}

  void backward() override { backward_fn(parents); }

  void for_each_parent(
      const std::function<void(const Tensor&)>& callback) const override {
    detail::for_each_parent_in_tuple(
        parents, std::make_index_sequence<std::tuple_size_v<Parents>>{},
        callback);
  }
};

template <typename... Args>
auto make_parents(Args&&... args) {
  return std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...);
}

template <typename Parents, typename BackwardFn>
std::shared_ptr<IGradNode> make_grad_node(Parents&& parents,
                                          BackwardFn&& backward_fn) {
  using ParentsDecayed = std::decay_t<Parents>;
  using BackwardFnDecayed = std::decay_t<BackwardFn>;

  return std::make_shared<GradNode<ParentsDecayed, BackwardFnDecayed>>(
      std::forward<Parents>(parents), std::forward<BackwardFn>(backward_fn));
}

struct AutogradMeta {
  std::unique_ptr<Tensor> grad_;
  std::shared_ptr<IGradNode> grad_fn_;
  bool requires_grad_ = false;
  bool is_leaf_ = true;
};

}  // namespace tensors
