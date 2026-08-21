#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "microtensor/autograd.hpp"

namespace tensors {

using index_t = std::ptrdiff_t;

std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
size_t compute_size(const std::vector<size_t>& shape);

class Tensor {
 private:
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;
  size_t offset_ = 0;
  std::shared_ptr<float[]> storage_;
  size_t max_size_;
  float* data_ = nullptr;
  mutable std::shared_ptr<AutogradMeta> autograd_meta_ = nullptr;

  size_t get_flat_index(std::span<const size_t> indices) const;
  size_t normalize_index(index_t index, size_t dim) const;
  std::vector<size_t> normalize_indices(std::span<const index_t> indices) const;

  Tensor as_strided(const std::vector<size_t>& shape,
                    const std::vector<size_t>& stride, size_t offset) const;

 public:
  Tensor();

  explicit Tensor(std::vector<size_t> shape);

  Tensor(std::vector<size_t> shape, std::vector<size_t> stride,
         std::shared_ptr<float[]> storage, size_t max_size_, size_t offset = 0);

  Tensor(std::vector<size_t> shape, std::initializer_list<float> list);

  template <typename... Indices>
    requires(std::integral<std::decay_t<Indices>> && ...)
  float& operator[](Indices... indices) {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<index_t, sizeof...(Indices)> raw{
        static_cast<index_t>(indices)...};
    auto idx_arr = normalize_indices(raw);

    return data_[get_flat_index(idx_arr)];
  }

  template <typename... Indices>
    requires(std::integral<std::decay_t<Indices>> && ...)
  const float& operator[](Indices... indices) const {
    static_assert(sizeof...(indices) > 0, "Number of indices cannot be zero!");

    std::array<index_t, sizeof...(Indices)> raw{
        static_cast<index_t>(indices)...};
    auto idx_arr = normalize_indices(raw);

    return data_[get_flat_index(idx_arr)];
  }

  static Tensor zeros(const std::vector<size_t>& shape);
  static Tensor ones(const std::vector<size_t>& shape);
  static Tensor full(const std::vector<size_t>& shape, float value);
  static Tensor linspace(float start, float end, size_t num);

  template <std::uniform_random_bit_generator Gen>
  static Tensor rand(const std::vector<size_t>& shape, Gen& rng) {
    Tensor ret(shape);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::generate_n(ret.storage().get(), ret.numel(),
                    [&]() { return dist(rng); });
    return ret;
  }

  Tensor clone() const;

  static inline Tensor zeros_like(const Tensor& t) {
    return Tensor::zeros(t.shape());
  }

  static inline Tensor ones_like(const Tensor& t) {
    return Tensor::ones(t.shape());
  }

  const std::vector<size_t>& shape() const noexcept;
  const std::vector<size_t>& stride() const noexcept;
  size_t offset() const noexcept;
  size_t ndim() const noexcept;
  size_t numel() const noexcept;
  bool empty() const noexcept;
  size_t storage_size() const noexcept;
  bool has_grad() const noexcept;
  const float* data() const noexcept;
  float* data() noexcept;
  std::shared_ptr<float[]> storage() const noexcept;
  bool is_contiguous() const noexcept;

  Tensor view(const std::vector<size_t>& shape) const;
  Tensor transpose(index_t dim0, index_t dim1) const;
  Tensor permute(const std::vector<index_t>& dims) const;
  Tensor contiguous() const;
  std::vector<Tensor> split(const std::vector<size_t>& split_sizes,
                            index_t axis = 0) const;
  std::vector<Tensor> chunk(size_t chunks, index_t axis = 0) const;

  bool requires_grad() const noexcept;
  void set_requires_grad(bool requires_grad) const;
  bool is_leaf() const noexcept;
  void set_is_leaf(bool is_leaf) const;
  std::shared_ptr<IGradNode> grad_fn() const noexcept;
  void set_grad_fn(std::shared_ptr<IGradNode> node) const;
  const Tensor& grad() const;
  Tensor& mutable_grad() const;
  void zero_grad() const;
  void add_grad(const Tensor& g) const;
  void backward() const;
};
}  // namespace tensors
