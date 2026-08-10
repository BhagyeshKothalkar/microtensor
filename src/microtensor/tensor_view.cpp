#include <stdexcept>

#include "microtensor/autograd.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {

Tensor Tensor::clone() const {
  auto result = cpu_kernels::clone(*this);

  if (AutogradContext::is_enabled() && this->requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(*this);
    auto backward_fn = [out = result](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad);
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor Tensor::as_strided(const std::vector<size_t>& shape,
                          const std::vector<size_t>& stride,
                          size_t offset) const {
  return Tensor(shape, stride, this->storage(), offset);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
  std::vector<size_t> new_shape = this->shape();
  std::vector<size_t> new_stride = this->stride();
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_stride[dim0], new_stride[dim1]);
  auto result = this->as_strided(new_shape, new_stride, this->offset());

  if (AutogradContext::is_enabled() && this->requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(*this);
    auto backward_fn = [out = result, dim0, dim1](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad.transpose(dim0, dim1));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

namespace {

std::vector<size_t> scalar_view_stride(const std::vector<size_t>& shape) {
  return std::vector<size_t>(shape.size(), 1);
}

std::vector<size_t> empty_view_stride(const std::vector<size_t>& shape) {
  std::vector<size_t> stride(shape.size(), 0);
  if (shape.empty()) {
    return stride;
  }
  stride.back() = 1;
  for (size_t d = shape.size() - 1; d > 0; --d) {
    stride[d - 1] = std::max<size_t>(shape[d], 1) * stride[d];
  }
  return stride;
}

std::vector<size_t> compute_view_stride(const std::vector<size_t>& old_shape,
                                        const std::vector<size_t>& old_stride,
                                        const std::vector<size_t>& new_shape) {
  std::vector<size_t> new_stride(new_shape.size(), 0);

  ssize_t view_d = static_cast<ssize_t>(new_shape.size()) - 1;
  size_t chunk_stride = old_stride.back();
  size_t tensor_numel = 1;
  size_t view_numel = 1;

  for (ssize_t tensor_d = static_cast<ssize_t>(old_shape.size()) - 1;
       tensor_d >= 0; --tensor_d) {
    tensor_numel *= old_shape[tensor_d];

    const bool chunk_end = tensor_d == 0 || (old_shape[tensor_d - 1] != 1 &&
                                             old_stride[tensor_d - 1] !=
                                                 tensor_numel * chunk_stride);

    if (!chunk_end) {
      continue;
    }

    while (view_d >= 0 &&
           (view_numel < tensor_numel || new_shape[view_d] == 1)) {
      new_stride[view_d] = view_numel * chunk_stride;
      view_numel *= new_shape[view_d];
      --view_d;
    }

    if (view_numel != tensor_numel) {
      throw std::runtime_error("view(): incompatible shape");
    }

    if (tensor_d > 0) {
      chunk_stride = old_stride[tensor_d - 1];
      tensor_numel = 1;
      view_numel = 1;
    }
  }

  if (view_d != -1) {
    throw std::runtime_error("view(): incompatible shape");
  }

  return new_stride;
}

};  // namespace

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
  assert(compute_size(new_shape) == numel());

  Tensor result;

  if (shape_.empty()) {
    result = as_strided(new_shape, scalar_view_stride(new_shape), offset());
  } else if (numel() == 0) {
    if (shape_ == new_shape) {
      result = *this;
    } else {
      result = as_strided(new_shape, empty_view_stride(new_shape), offset());
    }
  } else {
    result = as_strided(
        new_shape, compute_view_stride(shape_, stride_, new_shape), offset());
  }

  if (AutogradContext::is_enabled() && requires_grad()) {
    auto parents = make_parents(*this);
    auto backward = [out = result,
                     original_shape = shape_](const auto& parents) {
      NoGradGuard guard;

      const auto& [input] = parents;
      const Tensor grad = out.grad();

      if (input.requires_grad()) {
        input.add_grad(grad.view(original_shape));
      }
    };

    result.set_requires_grad(true);
    result.set_grad_fn(make_grad_node(std::move(parents), backward));
  }

  return result;
}

}  // namespace tensors