#include "autograd.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace microtensor {

namespace autograd {

bool AutogradContext::enabled_ = true;

bool AutogradContext::enabled() { return enabled_; }

void AutogradContext::set_enabled(bool value) { enabled_ = value; }

NoGradGuard::NoGradGuard() {
  previous_ = AutogradContext::enabled();

  AutogradContext::set_enabled(false);
}

NoGradGuard::~NoGradGuard() { AutogradContext::set_enabled(previous_); }

void accumulate(Tensor& destination, const Tensor& gradient) {
  if (!destination.autograd_meta()) {
    destination.autograd_meta() = std::make_shared<AutogradMeta>();
  }

  if (!destination.autograd_meta()->gradient.storage()) {
    destination.autograd_meta()->gradient = gradient.clone();

    return;
  }

  for (size_t i = 0; i < gradient.numel(); ++i) {
    destination.autograd_meta()->gradient.data()[i] += gradient.data()[i];
  }
}

namespace {

void build_graph(Tensor& tensor, std::vector<Tensor*>& order,
                 std::unordered_set<Tensor*>& visited) {
  if (visited.contains(&tensor)) {
    return;
  }

  visited.insert(&tensor);

  if (!tensor.autograd_meta() || !tensor.autograd_meta()->grad_fn) {
    return;
  }

  for (auto* parent : tensor.autograd_meta()->grad_fn->parents()) {
    build_graph(*parent, order, visited);
  }

  order.push_back(&tensor);
}

}  // namespace

void backward(Tensor& root) {
  if (!root.autograd_meta()) {
    return;
  }

  if (!root.autograd_meta()->gradient.storage()) {
    root.autograd_meta()->gradient = Tensor::ones(root.shape());
  }

  std::vector<Tensor*> order;

  std::unordered_set<Tensor*> visited;

  build_graph(root, order, visited);

  std::reverse(order.begin(), order.end());

  for (auto* tensor : order) {
    if (!tensor->autograd_meta()->grad_fn) {
      continue;
    }

    tensor->autograd_meta()->grad_fn->backward(
        tensor->autograd_meta()->gradient);
  }
}

}  // namespace autograd

}  // namespace microtensor
