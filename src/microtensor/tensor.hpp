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

/**
 * @brief Computes row-major strides for a tensor shape.
 *
 * Given a tensor shape, returns the stride (measured in elements)
 * corresponding to a contiguous row-major layout.
 *
 * Example:
 * Shape = {2, 3, 4}
 * Strides = {12, 4, 1}
 *
 * @param shape Tensor dimensions.
 * @return Row-major strides for the given shape.
 */
inline std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
  if (shape.empty()) return {};

  std::vector<size_t> strides(shape.size());

  /* Last dimension is contiguous. */
  strides.back() = 1;

  /* Compute remaining strides from back to front. */
  for (size_t i = shape.size() - 1; i > 0; --i)
    strides[i - 1] = strides[i] * shape[i];

  return strides;
}

/**
 * @brief Computes the total number of elements in a tensor.
 *
 * The size is the product of all dimensions.
 *
 * @param shape Tensor dimensions.
 * @return Total number of elements.
 */
inline size_t compute_size(const std::vector<size_t>& shape) {
  if (shape.empty()) return 0;

  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

/**
 * @brief Dense tensor view with shared storage.
 *
 * Tensor stores only metadata (shape, stride and offset) together with
 * shared ownership of the underlying storage. Multiple Tensor objects
 * may therefore refer to the same allocation while exposing different
 * layouts or views.
 *
 * Memory layout is row-major by default.
 *
 * Example:
 * @code
 * Tensor A({2,3});
 * A[1,2] = 5.f;
 *
 * Tensor B({2,2}, stride, A.storage(), offset);
 * @endcode
 */
class Tensor {
 private:
  /* Tensor dimensions. */
  std::vector<size_t> shape_;

  /* Strides measured in elements. */
  std::vector<size_t> stride_;

  /* Offset from the beginning of storage. */
  size_t offset_ = 0;

  /* Shared ownership of underlying allocation. */
  std::shared_ptr<float[]> storage_;

  /* Pointer to the first accessible element. */
  float* data_ = nullptr;

  /**
   * @brief Converts multidimensional indices into a flat index.
   *
   * The returned index is relative to data_, meaning the tensor offset
   * has already been accounted for.
   *
   * @param indices Tensor indices.
   * @return Flat index into data_.
   */
  size_t get_flat_index(std::span<const size_t> indices) const noexcept;

 public:
  /**
   * @brief Constructs an empty tensor.
   */
  Tensor();

  /**
   * @brief Allocates a contiguous tensor.
   *
   * Storage is allocated but left uninitialized.
   *
   * @param shape Tensor dimensions.
   */
  explicit Tensor(std::vector<size_t> shape);

  /**
   * @brief Constructs a tensor view over existing storage.
   *
   * This constructor does not allocate memory. It allows multiple
   * Tensor objects to share storage while exposing different layouts.
   *
   * @param shape Tensor dimensions.
   * @param stride Tensor strides.
   * @param storage Shared storage.
   * @param offset Offset into storage (in elements).
   */
  Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
         std::shared_ptr<float[]> storage, size_t offset = 0);

  /**
   * @brief Constructs a tensor initialized from a list.
   *
   * The number of supplied values must exactly equal the tensor size.
   *
   * Example:
   * @code
   * Tensor t({2,2}, {1.f,2.f,3.f,4.f});
   * @endcode
   *
   * @param shape Tensor dimensions.
   * @param list Initial values.
   */
  Tensor(std::vector<size_t> shape, std::initializer_list<float> list);

  /**
   * @brief Mutable element access.
   *
   * Number of supplied indices must equal the tensor rank.
   *
   * Example:
   * @code
   * tensor[1,2] = 5.f;
   * @endcode
   */
  template <typename... Indices>
    requires(std::same_as<std::decay_t<Indices>, size_t> && ...)
  float& operator[](Indices... indices) {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};

    return data_[get_flat_index(idx_arr)];
  }

  /**
   * @brief Read-only element access.
   *
   * Number of supplied indices must equal the tensor rank.
   */
  template <typename... Indices>
    requires(std::same_as<std::decay_t<Indices>, size_t> && ...)
  const float& operator[](Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};

    return data_[get_flat_index(idx_arr)];
  }

  /* Operations */

  // Tensor clone();

  /**
   * @brief Returns the tensor shape.
   */
  const std::vector<size_t>& shape() const noexcept;

  /**
   * @brief Returns the tensor strides.
   */
  const std::vector<size_t>& stride() const noexcept;

  /**
   * @brief Returns the storage offset.
   */
  size_t offset() const noexcept;

  /**
   * @brief Returns the number of dimensions.
   */
  size_t ndim() const noexcept;

  /**
   * @brief Returns the total number of elements.
   */
  size_t numel() const noexcept;

  /**
   * @brief Checks whether the tensor has zero elements.
   */
  bool empty() const noexcept;

  /**
   * @brief Returns a const pointer to tensor data.
   */
  const float* data() const noexcept;

  /**
   * @brief Returns a mutable pointer to tensor data.
   */
  float* data() noexcept;

  /**
   * @brief Returns shared ownership of the underlying storage.
   */
  std::shared_ptr<float[]> storage() const noexcept;

  /**
   * @brief Checks whether the tensor is stored contiguously.
   *
   * A tensor is contiguous if its strides correspond to a
   * row-major memory layout.
   */
  bool is_contiguous() const noexcept;
};

inline size_t Tensor::get_flat_index(
    std::span<const size_t> indices) const noexcept {
  /* Pair every index with its corresponding stride. */
  auto zipped = std::views::zip(indices, stride_);

  /* Compute Σ(index_i × stride_i). */
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
      /* Point directly to the first accessible element. */
      data_(storage_.get() + offset_) {}

inline Tensor::Tensor(std::vector<size_t> shape,
                      std::initializer_list<float> list)
    : Tensor(shape) {
  assert(list.size() == compute_size(shape));

  /* Copy initializer values into storage. */
  std::copy(list.begin(), list.end(), data_);
}

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

  /* Verify strides from innermost to outermost dimension. */
  for (size_t i = ndim(); i > 0; --i) {
    if (stride_[i - 1] != expected_stride) return false;

    expected_stride *= shape_[i - 1];
  }

  return true;
}

} /* namespace tensors */