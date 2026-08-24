#include <execution>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "microtensor/tensor.hpp"

namespace tensors {

std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<size_t> strides(shape.size());
  strides.back() = 1;

  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }

  return strides;
}

size_t compute_size(const std::vector<size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

Tensor::Tensor()
    : shape_({0}),
      stride_({0}),
      offset_(0),
      storage_(nullptr),
      max_size_(0),
      data_(nullptr) {}

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(shape),
      stride_(compute_strides(shape)),
      offset_(0),
      storage_(std::make_shared_for_overwrite<float[]>(compute_size(shape))),
      max_size_(compute_size(shape)),
      data_(storage_.get()) {}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
               std::shared_ptr<float[]> storage, size_t max_size, size_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      storage_(std::move(storage)),
      max_size_(max_size),
      data_(nullptr) {
  if (shape_.size() != stride_.size()) {
    throw std::invalid_argument("Tensor shape and stride ranks differ");
  }
  if (offset_ > max_size_) {
    throw std::out_of_range("Tensor offset exceeds storage");
  }
  if (numel() != 0) {
    if (!storage_ || (offset_ >= max_size_)) {
      throw std::out_of_range("Tensor view exceeds storage");
    }
    size_t last_index = offset_;
    for (size_t i = 0; i < shape_.size(); ++i) {
      const size_t extent = shape_[i] - 1;
      if (extent != 0 && stride_[i] > (max_size_ - last_index) / extent) {
        throw std::out_of_range("Tensor view exceeds storage");
      }
      last_index += extent * stride_[i];
    }
    if (last_index >= max_size_) {
      throw std::out_of_range("Tensor view exceeds storage");
    }
  }
  data_ = storage_ ? storage_.get() + offset_ : nullptr;
}

Tensor::Tensor(std::vector<size_t> shape, std::initializer_list<float> list)
    : Tensor(shape) {
  if (list.size() != compute_size(shape)) {
    throw std::invalid_argument("Tensor initializer size does not match shape");
  }

  std::copy(std::execution::unseq, list.begin(), list.end(), data_);
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), 0);
  return ret;
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), 1);
  return ret;
}

Tensor Tensor::full(const std::vector<size_t>& shape, float value) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), value);
  return ret;
}

Tensor Tensor::linspace(float start, float end, size_t num) {
  if (num == 0) {
    return Tensor({0});
  }
  Tensor ret({num});
  if (num == 1) {
    ret.data()[0] = start;
    return ret;
  }
  float step = (end - start) / (num - 1);
  std::generate_n(ret.storage().get(), num, [start, step, i = 0]() mutable {
    return start + step * (i++);
  });
  return ret;
}

size_t Tensor::get_flat_index(std::span<const size_t> indices) const {
  if (indices.size() != shape_.size()) {
    throw std::out_of_range("Tensor index rank does not match tensor rank");
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Tensor index is out of bounds");
    }
  }

  auto zipped = std::views::zip(indices, stride_);

  return std::ranges::fold_left(zipped, 0uz, [](size_t acc, const auto& pair) {
    return acc + (std::get<0>(pair) * std::get<1>(pair));
  });
}

size_t Tensor::normalize_index(index_t index, size_t dim) const {
  if (dim >= ndim()) {
    throw std::out_of_range("Tensor dimension is out of bounds");
  }
  const index_t extent = static_cast<index_t>(shape_[dim]);
  if (index < 0) {
    index += extent;
  }
  if (index < 0 || index >= extent) {
    throw std::out_of_range("Tensor index is out of bounds");
  }
  return static_cast<size_t>(index);
}

std::vector<size_t> Tensor::normalize_indices(
    std::span<const index_t> indices) const {
  if (indices.size() != ndim()) {
    throw std::out_of_range("Tensor index rank does not match tensor rank");
  }
  std::vector<size_t> result;
  result.reserve(indices.size());
  for (size_t dim = 0; dim < indices.size(); ++dim) {
    result.push_back(normalize_index(indices[dim], dim));
  }
  return result;
}

const std::vector<size_t>& Tensor::shape() const noexcept { return shape_; }

const std::vector<size_t>& Tensor::stride() const noexcept { return stride_; }

size_t Tensor::offset() const noexcept { return offset_; }

size_t Tensor::ndim() const noexcept { return shape_.size(); }

size_t Tensor::numel() const noexcept { return compute_size(shape_); }

bool Tensor::empty() const noexcept { return numel() == 0; }

size_t Tensor::storage_size() const noexcept { return max_size_; }

const float* Tensor::data() const noexcept { return data_; }

float* Tensor::data() noexcept { return data_; }

std::shared_ptr<float[]> Tensor::storage() const noexcept { return storage_; }

bool Tensor::is_contiguous() const noexcept {
  if (empty()) {
    return true;
  }
  size_t expected_stride = 1;

  for (size_t i = ndim(); i > 0; --i) {
    if (stride_[i - 1] != expected_stride) {
      return false;
    }
    expected_stride *= shape_[i - 1];
  }
  return true;
}
};  // namespace tensors
