#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>

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
  if (shape.size() != stride.size()) {
    throw std::invalid_argument("as_strided(): shape and stride ranks differ");
  }
  if (offset > max_size_) {
    throw std::out_of_range("as_strided(): offset exceeds storage");
  }
  if (compute_size(shape) != 0) {
    if (!storage_) {
      throw std::runtime_error("as_strided(): tensor has no storage");
    }
    size_t last_index = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
      const size_t extent = shape[i] - 1;
      if (extent != 0 && stride[i] > (max_size_ - last_index) / extent) {
        throw std::out_of_range("as_strided(): view exceeds storage");
      }
      last_index += extent * stride[i];
    }
    if (last_index >= max_size_) {
      throw std::out_of_range("as_strided(): view exceeds storage");
    }
  }
  return Tensor(shape, stride, storage_, max_size_, offset);
}

Tensor Tensor::transpose(index_t dim0, index_t dim1) const {
  auto normalize_dim = [this](index_t dim) { if (dim < 0) dim += static_cast<index_t>(ndim()); if (dim < 0 || dim >= static_cast<index_t>(ndim())) throw std::out_of_range("transpose(): dimension is out of range"); return static_cast<size_t>(dim); };
  const size_t first = normalize_dim(dim0);
  const size_t second = normalize_dim(dim1);
  std::vector<size_t> new_shape = this->shape();
  std::vector<size_t> new_stride = this->stride();
  std::swap(new_shape[first], new_shape[second]);
  std::swap(new_stride[first], new_stride[second]);
  auto result = this->as_strided(new_shape, new_stride, this->offset());

  if (AutogradContext::is_enabled() && this->requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(*this);
    auto backward_fn = [out = result, first, second](const auto& parents) {
      NoGradGuard guard;
      const auto& [lhs] = parents;
      const Tensor& grad = out.grad();
      if (lhs.requires_grad()) {
        lhs.add_grad(grad.transpose(static_cast<index_t>(first), static_cast<index_t>(second)));
      }
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Tensor Tensor::permute(const std::vector<index_t>& dims) const {
  if (dims.size() != ndim()) throw std::invalid_argument("permute(): rank mismatch");
  std::vector<size_t> order; order.reserve(ndim());
  std::vector<bool> seen(ndim(), false);
  for (auto dim : dims) { if (dim < 0) dim += static_cast<index_t>(ndim()); if (dim < 0 || dim >= static_cast<index_t>(ndim())) throw std::out_of_range("permute(): dimension is out of range"); const size_t normalized = static_cast<size_t>(dim); if (seen[normalized]) throw std::invalid_argument("permute(): repeated dimension"); seen[normalized] = true; order.push_back(normalized); }
  std::vector<size_t> shape(ndim()), stride(ndim());
  for (size_t i = 0; i < ndim(); ++i) { shape[i] = shape_[order[i]]; stride[i] = stride_[order[i]]; }
  Tensor result = as_strided(shape, stride, offset_);
  if (AutogradContext::is_enabled() && requires_grad()) {
    result.set_requires_grad(true); auto parents = make_parents(*this);
    auto backward = [out = result, order](const auto& parents) { NoGradGuard guard; const auto& [input] = parents; if (input.requires_grad()) { std::vector<index_t> inverse(order.size()); for (size_t i = 0; i < order.size(); ++i) inverse[order[i]] = static_cast<index_t>(i); input.add_grad(out.grad().permute(inverse)); } };
    result.set_grad_fn(make_grad_node(std::move(parents), std::move(backward)));
  }
  return result;
}

Tensor Tensor::contiguous() const { return is_contiguous() ? *this : clone(); }

std::vector<Tensor> Tensor::split(const std::vector<size_t>& sizes, index_t axis) const {
  if (axis < 0) axis += static_cast<index_t>(ndim()); if (axis < 0 || axis >= static_cast<index_t>(ndim())) throw std::out_of_range("split(): dimension is out of range"); const size_t dim = static_cast<size_t>(axis); size_t total = 0; for (auto size : sizes) total += size;
  if (total != shape_[dim]) throw std::invalid_argument("split(): sizes do not match dimension");
  std::vector<Tensor> result; result.reserve(sizes.size()); size_t start = 0;
  for (auto size : sizes) { auto shape = shape_; shape[dim] = size; result.push_back(as_strided(shape, stride_, offset_ + start * stride_[dim])); start += size; }
  return result;
}

std::vector<Tensor> Tensor::chunk(size_t chunks, index_t axis) const {
  if (chunks == 0) throw std::invalid_argument("chunk(): chunks must be positive"); if (axis < 0) axis += static_cast<index_t>(ndim()); if (axis < 0 || axis >= static_cast<index_t>(ndim())) throw std::out_of_range("chunk(): dimension is out of range"); const size_t dim = static_cast<size_t>(axis); const size_t base = shape_[dim] / chunks, remainder = shape_[dim] % chunks;
  std::vector<size_t> sizes; sizes.reserve(chunks); for (size_t i = 0; i < chunks; ++i) sizes.push_back(base + (i < remainder)); return split(sizes, static_cast<index_t>(dim));
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
  if (compute_size(new_shape) != numel()) {
    for (auto i : new_shape) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
    throw std::runtime_error("size mismatch, " +
                             std::to_string(compute_size(new_shape)) + " " +
                             std::to_string(numel()));
  }

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
    result.set_requires_grad(true);

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

    result.set_grad_fn(make_grad_node(std::move(parents), backward));
  }

  return result;
}

}  // namespace tensors
