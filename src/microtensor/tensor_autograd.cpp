#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "microtensor/autograd.hpp"
#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {

namespace {

Tensor reduce_sum_to(const Tensor& input,
                     const std::vector<std::size_t>& target_shape) {
  if (input.shape() == target_shape) {
    return input.clone();
  }

  std::vector<std::size_t> aligned_target = target_shape;
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

bool Tensor::requires_grad() const noexcept {
  return autograd_meta_ ? autograd_meta_->requires_grad_ : false;
}

void Tensor::set_requires_grad(bool requires_grad) const {
  if (requires_grad && !autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }
  if (autograd_meta_) {
    autograd_meta_->requires_grad_ = requires_grad;
    if (!requires_grad) {
      autograd_meta_->grad_.reset();
      autograd_meta_->grad_fn_.reset();
      autograd_meta_->is_leaf_ = true;
    }
  }
}

bool Tensor::is_leaf() const noexcept {
  return autograd_meta_ ? autograd_meta_->is_leaf_ : true;
}

void Tensor::set_is_leaf(bool is_leaf) const {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }
  autograd_meta_->is_leaf_ = is_leaf;
}

std::shared_ptr<IGradNode> Tensor::grad_fn() const noexcept {
  return autograd_meta_ ? autograd_meta_->grad_fn_ : nullptr;
}

void Tensor::set_grad_fn(std::shared_ptr<IGradNode> node) const {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }
  autograd_meta_->grad_fn_ = std::move(node);
  autograd_meta_->requires_grad_ = true;
  autograd_meta_->is_leaf_ = false;
}

const Tensor& Tensor::grad() const {
  static const Tensor empty_tensor;
  if (autograd_meta_ && autograd_meta_->grad_) {
    return *autograd_meta_->grad_;
  }
  return empty_tensor;
}

bool Tensor::has_grad() const noexcept {
  return autograd_meta_ && autograd_meta_->grad_;
}

Tensor& Tensor::mutable_grad() const {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }
  if (!autograd_meta_->grad_) {
    autograd_meta_->grad_ = std::make_unique<Tensor>(Tensor::zeros(shape_));
  }
  return *autograd_meta_->grad_;
}

void Tensor::zero_grad() const {
  if (autograd_meta_ && autograd_meta_->grad_) {
    std::fill_n(autograd_meta_->grad_->data(), autograd_meta_->grad_->numel(),
                0.0f);
  }
}

void Tensor::add_grad(const Tensor& g) const {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }

  Tensor effective_g = (g.shape() == shape_) ? g : reduce_sum_to(g, shape_);

  if (!autograd_meta_->grad_) {
    autograd_meta_->grad_ = std::make_unique<Tensor>(effective_g.clone());
  } else {
    cpu_kernels::add(*autograd_meta_->grad_, effective_g);
  }
}

void Tensor::backward() const {
  if (!requires_grad() && !grad_fn()) {
    return;
  }

  if (!autograd_meta_ || !autograd_meta_->grad_) {
    if (!autograd_meta_) {
      autograd_meta_ = std::make_shared<AutogradMeta>();
    }

    autograd_meta_->grad_ = std::make_unique<Tensor>(Tensor::ones(shape_));
  }

  std::vector<IGradNode*> topo_order;
  std::unordered_set<IGradNode*> visited;

  std::function<void(IGradNode*)> build_topo = [&](IGradNode* node) {
    if (!node || visited.contains(node)) {
      return;
    }

    visited.insert(node);

    node->for_each_parent([&](const Tensor& parent) {
      if (parent.grad_fn()) {
        build_topo(parent.grad_fn().get());
      }
    });

    topo_order.push_back(node);
  };

  if (grad_fn()) {
    build_topo(grad_fn().get());
  }

  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    (*it)->backward();
  }
}

}  // namespace tensors
