#pragma once

#include "broadcasting.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <ranges>
#include <vector>

namespace tensors {

static std::vector<size_t> compute_strides(const std::vector<size_t> &shape) {
  if (shape.empty())
    return {};

  std::vector<size_t> strides(shape.size());
  strides.back() = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

static size_t compute_size(const std::vector<size_t> &shape) {
  if (shape.empty())
    return 0;
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

// every Tensor in just a view of the correspondimg Storage.
template <typename T> class Tensor {

private:
  std::vector<size_t> shape_, stride_;
  std::shared_ptr<T[]> storage_;
  size_t begin_offset = 0;
  T *data_;
  // a non owning pointer for cleaner access. must point to
  // storage_->data()+begin_offset.

  size_t get_flat_index(std::span<const size_t> indices) const noexcept {
    // Zip indices and precomputed strides, then accumulate the dot product
    auto zipped = std::views::zip(indices, stride_);

    return std::ranges::fold_left(
        zipped, 0uz, [](size_t acc, const auto &pair) {
          // pair is a std::tuple containing the index and the stride
          return acc + (std::get<0>(pair) * std::get<1>(pair));
        });
  }

public:
  Tensor()
      : shape_({0}), stride_({0}), storage_(nullptr), begin_offset(0),
        data_(nullptr) {}

  explicit Tensor(std::vector<size_t> shape)
      : shape_(shape), stride_(compute_strides(shape)),
        storage_(std::make_shared_for_overwrite<T[]>(compute_size(shape))),
        data_(storage_.get()) {}

  Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
         std::shared_ptr<T[]> storage, size_t offset = 0)
      : shape_(std::move(shape)), stride_(std::move(stride)),
        storage_(std::move(storage)), begin_offset(offset),
        data_(storage_.get() + begin_offset) {}

  Tensor(std::vector<size_t> shape, std::initializer_list<T> list)
      : Tensor(shape) {
    assert(list.size() == compute_size(shape));
    std::copy(list.begin(), list.end(), data_);
  }

  template <typename... Indices> T &operator[](Indices... indices) {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    return data_[get_flat_index(idx_arr)];
  }

  template <typename... Indices> const T &operator[](Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");
    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};
    return data_[get_flat_index(idx_arr)];
  }

  // getters for the members
  const T *data() const noexcept { return data_; }
  T *data() noexcept { return data_; }
  const std::vector<size_t> &shape() const noexcept { return shape_; }
  const std::vector<size_t> &stride() const noexcept { return stride_; }
  std::shared_ptr<T[]> storage() const noexcept { return storage_; }
  size_t offset() const noexcept { return begin_offset; }

  // some metadata methods
  size_t ndim() const noexcept { return shape_.size(); }
  size_t numel() const noexcept { return compute_size(shape_); }
  bool empty() const noexcept { return numel() == 0; }

  // to implement:
  /*
  is_contigous
  slicing
  export as span and mdspan
  */

  Tensor<T> broadcast_to(const std::vector<size_t> &target_shape) const {
    return tensors::broadcast_to_shape(*this, target_shape);
  }
};

} // namespace tensors
