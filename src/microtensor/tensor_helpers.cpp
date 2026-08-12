#include <ranges>

#include "microtensor/tensor.hpp"

namespace tensors {
std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
  if (shape.empty()) return {};

  std::vector<size_t> strides(shape.size());
  strides.back() = 1;

  for (size_t i = shape.size() - 1; i > 0; --i)
    strides[i - 1] = strides[i] * shape[i];

  return strides;
}

size_t compute_size(const std::vector<size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
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
  if (dim >= ndim()) throw std::out_of_range("Tensor dimension is out of bounds");
  const index_t extent = static_cast<index_t>(shape_[dim]);
  if (index < 0) index += extent;
  if (index < 0 || index >= extent) throw std::out_of_range("Tensor index is out of bounds");
  return static_cast<size_t>(index);
}

std::vector<size_t> Tensor::normalize_indices(std::span<const index_t> indices) const {
  if (indices.size() != ndim()) throw std::out_of_range("Tensor index rank does not match tensor rank");
  std::vector<size_t> result;
  result.reserve(indices.size());
  for (size_t dim = 0; dim < indices.size(); ++dim) result.push_back(normalize_index(indices[dim], dim));
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
  if (empty()) return true;

  size_t expected_stride = 1;

  for (size_t i = ndim(); i > 0; --i) {
    if (stride_[i - 1] != expected_stride) return false;

    expected_stride *= shape_[i - 1];
  }

  return true;
}

}  // namespace tensors
