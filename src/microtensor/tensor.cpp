#include "tensor.hpp"

#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

#include "autograd.hpp"

namespace microtensor {

namespace {

size_t max_offset(std::span<const size_t> shape, std::span<const size_t> stride,
                  size_t offset) {
  if (shape.empty()) {
    return offset;
  }

  size_t result = offset;

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 0) {
      result += (shape[i] - 1) * stride[i];
    }
  }

  return result;
}

}  // namespace

Tensor::Tensor() : storage_(nullptr), offset_(0), storage_size_(0) {}

Tensor::Tensor(std::span<const size_t> shape)
    : shape_(shape.begin(), shape.end()),
      stride_(contiguous_strides(shape)),
      offset_(0),
      storage_size_(checked_numel(shape)) {
  if (storage_size_) {
    storage_ = std::shared_ptr<float[]>(new float[storage_size_]());
  }
}

Tensor::Tensor(std::shared_ptr<float[]> storage, std::vector<size_t> shape,
               std::vector<size_t> stride, size_t offset, size_t storage_size)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      storage_size_(storage_size) {}

size_t Tensor::checked_numel(std::span<const size_t> shape) {
  if (shape.empty()) {
    return 1;
  }

  size_t result = 1;

  for (auto dim : shape) {
    if (dim == 0) {
      return 0;
    }

    if (result > SIZE_MAX / dim) {
      throw std::overflow_error("Tensor numel overflow");
    }

    result *= dim;
  }

  return result;
}

std::vector<size_t> Tensor::contiguous_strides(std::span<const size_t> shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<size_t> stride(shape.size());

  size_t current = 1;

  for (size_t i = shape.size(); i-- > 0;) {
    stride[i] = current;
    current *= shape[i];
  }

  return stride;
}

Tensor Tensor::zeros(std::span<const size_t> shape) { return Tensor(shape); }

Tensor Tensor::ones(std::span<const size_t> shape) {
  Tensor result(shape);

  std::fill(result.data(), result.data() + result.numel(), 1.0f);

  return result;
}

Tensor Tensor::full(std::span<const size_t> shape, float value) {
  Tensor result(shape);

  std::fill(result.data(), result.data() + result.numel(), value);

  return result;
}

Tensor Tensor::linspace(float start, float end, size_t steps) {
  Tensor result(std::array<size_t, 1>{steps});

  if (steps == 0) {
    return result;
  }

  if (steps == 1) {
    result[0] = start;
    return result;
  }

  float step = (end - start) / static_cast<float>(steps - 1);

  for (size_t i = 0; i < steps; ++i) {
    result[i] = start + step * i;
  }

  return result;
}

Tensor Tensor::rand(std::span<const size_t> shape) {
  Tensor result(shape);

  std::mt19937 rng(std::random_device{}());

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (size_t i = 0; i < result.numel(); ++i) {
    result.data()[i] = dist(rng);
  }

  return result;
}

Tensor Tensor::zeros_like(const Tensor& other) { return zeros(other.shape()); }

Tensor Tensor::ones_like(const Tensor& other) { return ones(other.shape()); }

std::span<const size_t> Tensor::shape() const noexcept { return shape_; }

std::span<const size_t> Tensor::stride() const noexcept { return stride_; }

size_t Tensor::ndim() const noexcept { return shape_.size(); }

size_t Tensor::numel() const noexcept { return checked_numel(shape_); }

size_t Tensor::storage_size() const noexcept { return storage_size_; }

size_t Tensor::offset() const noexcept { return offset_; }

bool Tensor::empty() const noexcept { return numel() == 0; }

bool Tensor::is_contiguous() const noexcept {
  return stride_ == contiguous_strides(shape_);
}

float* Tensor::data() noexcept { return storage_.get() + offset_; }

const float* Tensor::data() const noexcept { return storage_.get() + offset_; }

std::shared_ptr<float[]> Tensor::storage() const noexcept { return storage_; }

Tensor Tensor::as_strided(std::span<const size_t> shape,
                          std::span<const size_t> stride, size_t offset) const {
  validate_view_bounds(shape, stride, offset);

  return Tensor(storage_, std::vector<size_t>(shape.begin(), shape.end()),
                std::vector<size_t>(stride.begin(), stride.end()),
                offset_ + offset, storage_size_);
}

Tensor Tensor::view(std::span<const size_t> shape) const {
  if (!is_contiguous()) {
    throw std::runtime_error("view requires contiguous tensor");
  }

  if (checked_numel(shape) != numel()) {
    throw std::runtime_error("invalid view size");
  }

  return Tensor(storage_, std::vector<size_t>(shape.begin(), shape.end()),
                contiguous_strides(shape), offset_, storage_size_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
  if (dim0 >= ndim() || dim1 >= ndim()) {
    throw std::out_of_range("transpose dimension");
  }

  auto shape = shape_;
  auto stride = stride_;

  std::swap(shape[dim0], shape[dim1]);

  std::swap(stride[dim0], stride[dim1]);

  return Tensor(storage_, shape, stride, offset_, storage_size_);
}

Tensor Tensor::permute(std::span<const size_t> dims) const {
  if (dims.size() != ndim()) {
    throw std::runtime_error("invalid permutation");
  }

  std::vector<bool> seen(ndim());

  std::vector<size_t> shape(ndim());
  std::vector<size_t> stride(ndim());

  for (size_t i = 0; i < ndim(); ++i) {
    if (dims[i] >= ndim() || seen[dims[i]]) {
      throw std::runtime_error("invalid permutation");
    }

    seen[dims[i]] = true;

    shape[i] = shape_[dims[i]];
    stride[i] = stride_[dims[i]];
  }

  return Tensor(storage_, shape, stride, offset_, storage_size_);
}

Tensor Tensor::clone() const {
  Tensor result(shape_);

  for (size_t i = 0; i < numel(); ++i) {
    result.data()[i] = data()[i];
  }

  return result;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }

  return clone();
}

std::vector<Tensor> Tensor::split(size_t sections, size_t dim) const {
  if (shape_[dim] % sections != 0) {
    throw std::runtime_error("split not divisible");
  }

  size_t size = shape_[dim] / sections;

  std::vector<Tensor> result;

  for (size_t i = 0; i < sections; ++i) {
    auto shape = shape_;
    shape[dim] = size;

    size_t offset = i * size * stride_[dim];

    result.push_back(as_strided(shape, stride_, offset));
  }

  return result;
}

std::vector<Tensor> Tensor::chunk(size_t chunks, size_t dim) const {
  size_t size = (shape_[dim] + chunks - 1) / chunks;

  std::vector<Tensor> result;

  for (size_t start = 0; start < shape_[dim]; start += size) {
    auto shape = shape_;

    shape[dim] = std::min(size, shape_[dim] - start);

    result.push_back(as_strided(shape, stride_, start * stride_[dim]));
  }

  return result;
}

void Tensor::validate_view_bounds(std::span<const size_t> shape,
                                  std::span<const size_t> stride,
                                  size_t offset) const {
  if (shape.size() != stride.size()) {
    throw std::runtime_error("shape/stride mismatch");
  }

  if (max_offset(shape, stride, offset) >= storage_size_) {
    throw std::out_of_range("view exceeds storage");
  }
}

std::shared_ptr<autograd::AutogradMeta>& Tensor::autograd_meta() {
  return autograd_;
}

const std::shared_ptr<autograd::AutogradMeta>& Tensor::autograd_meta() const {
  return autograd_;
}

bool Tensor::requires_grad() const {
  return autograd_ && autograd_->requires_grad;
}

void Tensor::requires_grad(bool enabled) {
  if (!autograd_) {
    autograd_ = std::make_shared<autograd::AutogradMeta>();
  }

  autograd_->requires_grad = enabled;
}

const Tensor* Tensor::grad() const {
  if (!autograd_) {
    return nullptr;
  }

  if (!autograd_->gradient.storage()) {
    return nullptr;
  }

  return &autograd_->gradient;
}

void Tensor::backward() { autograd::backward(*this); }

}  // namespace microtensor
