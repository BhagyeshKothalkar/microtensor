#include <execution>

#include "microtensor/tensor.hpp"

namespace tensors {

Tensor::Tensor()
    : shape_({0}),
      stride_({0}),
      offset_(0),
      storage_(nullptr),
      data_(nullptr) {}

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(shape),
      stride_(compute_strides(shape)),
      offset_(0),
      storage_(std::make_shared_for_overwrite<float[]>(compute_size(shape))),
      data_(storage_.get()) {}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
               std::shared_ptr<float[]> storage, size_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      storage_(std::move(storage)),

      data_(storage_.get() + offset_) {}

Tensor::Tensor(std::vector<size_t> shape, std::initializer_list<float> list)
    : Tensor(shape) {
  assert(list.size() == compute_size(shape));

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
  Tensor ret({num});
  float step = (end - start) / (num - 1);
  std::generate_n(ret.storage().get(), num, [start, step, i = 0]() mutable {
    return start + step * (i++);
  });
  return ret;
}
};  // namespace tensors