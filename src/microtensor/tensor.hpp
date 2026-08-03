#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace tensors {

template <typename Dest, typename... Src>
class TensorIterator;

inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
  if (shape.empty()) return {};
  std::vector<size_t> strides(shape.size());
  strides.back() = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

inline size_t compute_size(const std::vector<size_t>& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

class Tensor {
 private:
  std::vector<size_t> shape_, stride_;
  size_t offset_ = 0;
  std::shared_ptr<float[]> storage_;
  float* data_ = nullptr;

  size_t get_flat_index(std::span<const size_t> indices) const noexcept;

 public:
  // Constructors
  Tensor();
  explicit Tensor(std::vector<size_t> shape);
  Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
         std::shared_ptr<float[]> storage, size_t offset = 0);
  Tensor(std::vector<size_t> shape, std::initializer_list<float> list);

  // Subscript Operators
  template <typename... Indices>
    requires(std::same_as<std::decay_t<Indices>, size_t> && ...)
  float& operator[](Indices... indices) {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    return data_[get_flat_index(idx_arr)];
  }

  template <typename... Indices>
    requires(std::same_as<std::decay_t<Indices>, size_t> && ...)
  const float& operator[](Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    return data_[get_flat_index(idx_arr)];
  }

  // Operations
  // Tensor clone();

  // Metadata Getters
  const std::vector<size_t>& shape() const noexcept;
  const std::vector<size_t>& stride() const noexcept;
  size_t offset() const noexcept;
  size_t ndim() const noexcept;
  size_t numel() const noexcept;
  bool empty() const noexcept;

  // Raw Pointers / Storage Accessors
  const float* data() const noexcept;
  float* data() noexcept;
  std::shared_ptr<float[]> storage() const noexcept;

  // Layout Helpers
  bool is_contiguous() const noexcept;
  std::span<float> as_span();
  std::span<const float> as_span() const;
};

inline size_t Tensor::get_flat_index(
    std::span<const size_t> indices) const noexcept {
  auto zipped = std::views::zip(indices, stride_);
  return std::ranges::fold_left(zipped, 0uz, [](size_t acc, const auto& pair) {
    return acc + (std::get<0>(pair) * std::get<1>(pair));
  });
}

inline Tensor::Tensor()
    : shape_({0}),
      stride_({0}),
      offset_(0),
      storage_(nullptr),
      data_(nullptr) {}

inline Tensor::Tensor(std::vector<size_t> shape)
    : shape_(shape),
      stride_(compute_strides(shape)),
      offset_(0),
      storage_(std::make_shared_for_overwrite<float[]>(compute_size(shape))),
      data_(storage_.get()) {}

inline Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
                      std::shared_ptr<float[]> storage, size_t offset)
    : shape_(std::move(shape)),
      stride_(std::move(stride)),
      offset_(offset),
      storage_(std::move(storage)),
      data_(storage_.get() + offset_) {}

inline Tensor::Tensor(std::vector<size_t> shape,
                      std::initializer_list<float> list)
    : Tensor(shape) {
  assert(list.size() == compute_size(shape));
  std::copy(list.begin(), list.end(), data_);
}

// inline Tensor Tensor::clone() {
//   Tensor ret(
//       this->shape_, this->stride_,
//       std::make_shared_for_overwrite<float[]>(compute_size(this->shape_)),
//       this->offset_);
//   TensorIterator<float, const float> it(ret, *this);
//   while (it.has_next()) {
//     auto&& [ret_val, this_val] = it.next();
//     ret_val = this_val;
//   }
//   return ret;
// }

inline const std::vector<size_t>& Tensor::shape() const noexcept {
  return shape_;
}
inline const std::vector<size_t>& Tensor::stride() const noexcept {
  return stride_;
}
inline size_t Tensor::offset() const noexcept { return offset_; }
inline size_t Tensor::ndim() const noexcept { return shape_.size(); }
inline size_t Tensor::numel() const noexcept { return compute_size(shape_); }
inline bool Tensor::empty() const noexcept { return numel() == 0; }

inline const float* Tensor::data() const noexcept { return data_; }
inline float* Tensor::data() noexcept { return data_; }
inline std::shared_ptr<float[]> Tensor::storage() const noexcept {
  return storage_;
}

inline bool Tensor::is_contiguous() const noexcept {
  if (empty()) return true;
  size_t expected_stride = 1;
  for (size_t i = ndim(); i > 0; --i) {
    if (stride_[i - 1] != expected_stride) return false;
    expected_stride *= shape_[i - 1];
  }
  return true;
}

inline std::span<float> Tensor::as_span() {
  assert(is_contiguous() &&
         "Cannot export non-contiguous tensor as a flat span");
  return std::span<float>(data_, numel());
}

inline std::span<const float> Tensor::as_span() const {
  assert(is_contiguous() &&
         "Cannot export non-contiguous tensor as a flat span");
  return std::span<const float>(data_, numel());
}

}  // namespace tensors