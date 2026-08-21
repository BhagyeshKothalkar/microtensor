#pragma once

#include <memory>
#include <span>
#include <tuple>
#include <vector>

#include "tensor.hpp"

namespace microtensor {

namespace autograd {

class AutogradMeta {
 public:
  bool requires_grad = false;

  Tensor gradient;

  std::shared_ptr<class GradNode> grad_fn;

  bool is_leaf = true;
};

class AutogradContext {
 public:
  static bool enabled();

  static void set_enabled(bool value);

 private:
  static bool enabled_;
};

class NoGradGuard {
 public:
  NoGradGuard();

  ~NoGradGuard();

  NoGradGuard(const NoGradGuard&) = delete;

  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool previous_;
};

class GradNode {
 public:
  virtual ~GradNode() = default;

  virtual std::vector<Tensor*> parents() = 0;

  virtual void backward(const Tensor& gradient) = 0;
};

template <class ParentTuple, class Backward>
class GradNodeImpl final : public GradNode {
 public:
  GradNodeImpl(ParentTuple parents, Backward backward)
      : parents_(std::move(parents)), backward_(std::move(backward)) {}

  std::vector<Tensor*> parents() override {
    std::vector<Tensor*> result;

    std::apply([&](auto&... p) { (result.push_back(&p), ...); }, parents_);

    return result;
  }

  void backward(const Tensor& gradient) override { backward_(gradient); }

 private:
  ParentTuple parents_;

  Backward backward_;
};

template <typename... T>
auto make_parents(T&... tensors) {
  return std::tuple<T&...>(tensors...);
}

template <class Backward, class... Parents>
void record(Tensor& output, Backward&& backward, Parents&... parents);

void accumulate(Tensor& destination, const Tensor& gradient);

void backward(Tensor& root);

Tensor sum_to_shape(const Tensor& input, std::span<const size_t> shape);

template <class Backward, class... Parents>
void record(Tensor& output, Backward&& backward, Parents&... parents) {
  if (!AutogradContext::enabled()) {
    return;
  }

  bool needs_grad = false;

  ((needs_grad |= parents.requires_grad()), ...);

  if (!needs_grad) {
    return;
  }

  if (!output.autograd_meta()) {
    output.autograd_meta() = std::make_shared<AutogradMeta>();
  }

  output.autograd_meta()->requires_grad = true;
  output.autograd_meta()->is_leaf = false;

  using ParentTuple = decltype(make_parents(parents...));

  using Node = GradNodeImpl<ParentTuple, std::decay_t<Backward> >;

  output.autograd_meta()->grad_fn = std::make_shared<Node>(
      make_parents(parents...), std::forward<Backward>(backward));
}

}  // namespace autograd

}  // namespace microtensor
