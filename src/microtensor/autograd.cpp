#include "microtensor/autograd.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {

namespace {

Tensor reduce_sum_to(const Tensor& input,
                     const std::vector<size_t>& target_shape) {
  if (input.shape() == target_shape) {
    return input.clone();
  }

  /* Pad target shape with leading 1s to align ranks. */
  std::vector<size_t> aligned_target = target_shape;
  while (aligned_target.size() < input.ndim()) {
    aligned_target.insert(aligned_target.begin(), 1);
  }

  Tensor output = Tensor::zeros(aligned_target);
  Tensor output_view = broadcast_to_shape(output, input.shape());

  TensorIterator<float, const float> it(output_view, input);
  while (it.has_next()) {
    auto&& [out_val, in_val] = it.next();
    out_val += in_val;
  }

  if (aligned_target == target_shape) {
    return output;
  }
  return output.view(target_shape);
}

}  // namespace

void Tensor::add_grad(const Tensor& g) const {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }

  Tensor effective_g = (g.shape() == shape_) ? g : reduce_sum_to(g, shape_);

  if (!autograd_meta_->grad_) {
    autograd_meta_->grad_ = std::make_unique<Tensor>(effective_g.clone());
  } else {
    functional::add_(*autograd_meta_->grad_, effective_g);
  }
}

void Tensor::backward() const {
  if (!requires_grad() && !grad_fn()) {
    return;
  }

  /* Seed gradient if uninitialized */
  if (!autograd_meta_ || !autograd_meta_->grad_) {
    if (!autograd_meta_) {
      autograd_meta_ = std::make_shared<AutogradMeta>();
    }
    autograd_meta_->grad_ = std::make_unique<Tensor>(Tensor::ones(shape_));
  }

  /* Phase 1: Build post-order topological sort of execution nodes */
  std::vector<IGradNode*> topo_order;
  std::unordered_set<IGradNode*> visited;

  std::function<void(IGradNode*)> build_topo = [&](IGradNode* node) {
    if (!node || visited.contains(node)) {
      return;
    }
    visited.insert(node);

    for (const auto& parent : node->get_parents()) {
      if (parent.grad_fn()) {
        build_topo(parent.grad_fn().get());
      }
    }

    topo_order.push_back(node);
  };

  if (grad_fn()) {
    build_topo(grad_fn().get());
  }

  /* Phase 2: Execute backward pass in reverse topological order */
  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    (*it)->backward();
  }
}

}  // namespace tensors
