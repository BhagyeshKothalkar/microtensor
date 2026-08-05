#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <execution>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace tensors {

/**
 * @brief Computes row-major strides for a tensor shape.
 * Given a tensor shape, returns the stride (measured in elements)
 * corresponding to a contiguous row-major layout.
 * Example:
 * Shape = {2, 3, 4}
 * Strides = {12, 4, 1}
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
 * The size is the product of all dimensions.
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
 * Tensor stores only metadata (shape, stride and offset) together with
 * shared ownership of the underlying storage. Multiple Tensor objects
 * may therefore refer to the same allocation while exposing different
 * layouts or views.
 * Memory layout is row-major by default.
 * Example:
 * @code
 * Tensor A({2,3});
 * A[1,2] = 5.f;
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
   * The returned index is relative to data_, meaning the tensor offset
   * has already been accounted for.
   * @param indices Tensor indices.
   * @return Flat index into data_.
   */
  size_t get_flat_index(std::span<const size_t> indices) const noexcept;

  /**
   * @brief constructs a view of given shape, stride and offset.
   */
  inline Tensor as_strided(const std::vector<size_t>& shape,
                           const std::vector<size_t>& stride,
                           size_t offset) const;

 public:
  /**
   * @brief Constructs an empty tensor.
   */
  Tensor();

  /**
   * @brief Allocates a contiguous tensor.
   * Storage is allocated but left uninitialized.
   * @param shape Tensor dimensions.
   */
  explicit Tensor(std::vector<size_t> shape);

  /**
   * @brief Constructs a tensor view over existing storage.
   * This constructor does not allocate memory. It allows multiple
   * Tensor objects to share storage while exposing different layouts.
   * @param shape Tensor dimensions.
   * @param stride Tensor strides.
   * @param storage Shared storage.
   * @param offset Offset into storage (in elements).
   */
  Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
         std::shared_ptr<float[]> storage, size_t offset = 0);

  /**
   * @brief Constructs a tensor initialized from a list.
   * The number of supplied values must exactly equal the tensor size.
   * Example:
   * @code
   * Tensor t({2,2}, {1.f,2.f,3.f,4.f});
   * @endcode
   * @param shape Tensor dimensions.
   * @param list Initial values.
   */
  Tensor(std::vector<size_t> shape, std::initializer_list<float> list);

  /**
   * @brief Mutable element access.
   * Number of supplied indices must equal the tensor rank.
   * Example:
   * @code
   * tensor[1,2] = 5.f;
   * @endcode
   */
  template <typename... Indices>
    requires(std::convertible_to<std::decay_t<Indices>, size_t> && ...)
  float& operator[](Indices... indices) {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};

    return data_[get_flat_index(idx_arr)];
  }

  /**
   * @brief Read-only element access.
   * Number of supplied indices must equal the tensor rank.
   */
  template <typename... Indices>
    requires(std::convertible_to<std::decay_t<Indices>, size_t> && ...)
  const float& operator[](Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<size_t, sizeof...(Indices)> idx_arr{
        static_cast<size_t>(indices)...};

    return data_[get_flat_index(idx_arr)];
  }

  /* Fatory/Creation Methods */

  /**
   * @brief Creates a tensor filled with zeros.
   */
  static inline Tensor zeros(const std::vector<size_t>& shape);

  /**
   * @brief Creates a tensor filled with ones.
   */
  static inline Tensor ones(const std::vector<size_t>& shape);

  /**
   * @brief Creates a tensor filled with a specific scalar value.
   */
  static inline Tensor full(const std::vector<size_t>& shape, float value);

  /**
   * @brief Creates a 1D tensor with evenly spaced values over a specified
   * range.
   */
  static inline Tensor linspace(float start, float end, size_t num);

  /**
   * @brief Creates a tensor of given shape filled with random floats in [0, 1)
   */
  template <std::uniform_random_bit_generator Gen>
  static inline Tensor rand(const std::vector<size_t>& shape,
                            Gen& rng) {  // Notice the '&'
    Tensor ret(shape);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate_n(ret.storage().get(), ret.numel(),
                    [&]() { return dist(rng); });
    return ret;
  }

  /* Operations */
  inline Tensor clone() const;

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
   * once created, no operations are allowed to change numel for a tensor.
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
   * A tensor is contiguous if its strides correspond to a
   * row-major memory layout.
   */
  bool is_contiguous() const noexcept;

  // view ops
  inline Tensor view(const std::vector<size_t>& shape) const;
  // inline Tensor reshape(const std::vector<size_t>& shape) const;
  inline Tensor transpose(size_t dim0, size_t dim1) const;
  // inline Tensor permute(const std::vector<size_t>& dims) const;
  // inline Tensor narrow(size_t dim, size_t start, size_t length) const;
  // inline Tensor slice(size_t dim, size_t start, size_t stop,
  //                     size_t step = 1) const;
  // inline Tensor select(size_t dim, size_t index) const;
  // inline Tensor squeeze(std::optional<size_t> dim = {}) const;
  // inline Tensor unsqueeze(size_t dim) const;
  // inline Tensor expand(const std::vector<size_t>& shape) const;
  // inline Tensor flatten(size_t start_dim = 0, size_t end_dim = -1) const;
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
  std::copy(std::execution::unseq, list.begin(), list.end(), data_);
}

inline Tensor Tensor::zeros(const std::vector<size_t>& shape) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), 0);
  return ret;
}

inline Tensor Tensor::ones(const std::vector<size_t>& shape) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), 1);
  return ret;
}

inline Tensor Tensor::full(const std::vector<size_t>& shape, float value) {
  Tensor ret(shape);
  std::fill_n(std::execution::unseq, ret.storage().get(), ret.numel(), value);
  return ret;
}

inline Tensor Tensor::linspace(float start, float end, size_t num) {
  Tensor ret({num});
  float step = (end - start) / (num - 1);
  std::generate_n(ret.storage().get(), num, [start, step, i = 0]() mutable {
    return start + step * (i++);
  });
  return ret;
}

inline Tensor Tensor::clone() const {
  size_t num_elem = this->numel();
  std::shared_ptr<float[]> new_storage(
      std::make_shared_for_overwrite<float[]>(num_elem));
  std::copy_n(std::execution::unseq, this->storage().get(), num_elem,
              new_storage.get());
  return Tensor(this->shape_, this->stride_, new_storage, this->offset_);
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

inline Tensor Tensor::as_strided(const std::vector<size_t>& shape,
                                 const std::vector<size_t>& stride,
                                 size_t offset) const {
  return Tensor(shape, stride, this->storage(), offset);
}

inline Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
  assert(compute_size(new_shape) == numel());

  const auto& old_shape = shape_;
  const auto& old_stride = stride_;

  // 1. Scalar case (0-dim tensor)
  if (old_shape.empty()) {
    std::vector<size_t> new_stride(new_shape.size(), 1);
    return as_strided(new_shape, new_stride, this->offset());
  }

  size_t total_numel = numel();
  bool zero_numel = (total_numel == 0);

  // 2. 0-numel tensor with identical shape
  if (zero_numel && old_shape == new_shape) {
    return *this;
  }

  std::vector<size_t> new_stride(new_shape.size(), 0);

  // 3. 0-numel tensor stride computation
  if (zero_numel) {
    if (!new_shape.empty()) {
      for (size_t i = new_shape.size(); i > 0; --i) {
        size_t view_d = i - 1;
        if (view_d == new_shape.size() - 1) {
          new_stride[view_d] = 1;
        } else {
          new_stride[view_d] = std::max<size_t>(new_shape[view_d + 1], 1) * new_stride[view_d + 1];
        }
      }
    }
    return as_strided(new_shape, new_stride, this->offset());
  }

  // 4. General case block matching algorithm
  ssize_t view_d = static_cast<ssize_t>(new_shape.size()) - 1;
  size_t chunk_base_stride = old_stride.back();
  size_t tensor_numel = 1;
  size_t view_numel = 1;

  for (ssize_t tensor_d = static_cast<ssize_t>(old_shape.size()) - 1; tensor_d >= 0; --tensor_d) {
    tensor_numel *= old_shape[tensor_d];

    if (tensor_d == 0 ||
        (old_shape[tensor_d - 1] != 1 &&
         old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
      
      while (view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d] == 1)) {
        new_stride[view_d] = view_numel * chunk_base_stride;
        view_numel *= new_shape[view_d];
        --view_d;
      }

      if (view_numel != tensor_numel) {
        throw std::runtime_error("view(): incompatible shape");
      }

      if (tensor_d > 0) {
        chunk_base_stride = old_stride[tensor_d - 1];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }

  if (view_d != -1) {
    throw std::runtime_error("view(): incompatible shape");
  }

  return as_strided(new_shape, new_stride, this->offset());
}

inline Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
  std::vector<size_t> new_shape = this->shape();
  std::vector<size_t> new_stride = this->stride();
  std::swap(new_shape[dim0], new_shape[dim1]);
  std::swap(new_stride[dim0], new_stride[dim1]);
  return this->as_strided(new_shape, new_stride, this->offset());
}

} /* namespace tensors */