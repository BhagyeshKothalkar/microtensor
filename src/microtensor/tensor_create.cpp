#include <execution>
#include <stdexcept>
#include <utility>

#include "microtensor/tensor.hpp"

namespace tensors {

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
    if (!storage_ || offset_ >= max_size_) {
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
};  // namespace tensors
