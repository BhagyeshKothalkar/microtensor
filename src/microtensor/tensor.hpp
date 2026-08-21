#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>
namespace microtensor {

class Tensor;

namespace autograd {
class AutogradMeta;

class GradNode;
void backward(Tensor&);
void accumulate(Tensor&, const Tensor&);
}  // namespace autograd

class Tensor {
 public:
  Tensor();

  explicit Tensor(std::span<const size_t> shape);

  Tensor(std::shared_ptr<float[]> storage, std::vector<size_t> shape,
         std::vector<size_t> stride, size_t offset, size_t storage_size);

  // ---- creation ----

  static Tensor zeros(std::span<const size_t> shape);

  static Tensor ones(std::span<const size_t> shape);

  static Tensor full(std::span<const size_t> shape, float value);

  static Tensor linspace(float start, float end, size_t steps);

  static Tensor rand(std::span<const size_t> shape);

  static Tensor zeros_like(const Tensor& other);

  static Tensor ones_like(const Tensor& other);

  // ---- metadata ----

  std::span<const size_t> shape() const noexcept;

  std::span<const size_t> stride() const noexcept;

  size_t ndim() const noexcept;

  size_t numel() const noexcept;

  size_t storage_size() const noexcept;

  size_t offset() const noexcept;

  bool empty() const noexcept;

  bool is_contiguous() const noexcept;

  // ---- storage ----

  float* data() noexcept;

  const float* data() const noexcept;

  std::shared_ptr<float[]> storage() const noexcept;

  // ---- indexing ----

  template <class Index>
  float& operator[](Index index) {
    return data()[index];
  }

  template <class Index>
  const float& operator[](Index index) const {
    return data()[index];
  }

  // ---- views ----

  Tensor as_strided(std::span<const size_t> shape,
                    std::span<const size_t> stride, size_t offset = 0) const;

  Tensor view(std::span<const size_t> shape) const;

  Tensor transpose(size_t dim0, size_t dim1) const;

  Tensor permute(std::span<const size_t> dims) const;

  Tensor contiguous() const;

  Tensor clone() const;

  std::vector<Tensor> split(size_t sections, size_t dim) const;

  std::vector<Tensor> chunk(size_t chunks, size_t dim) const;

  // ---- autograd bridge ----

  bool requires_grad() const;

  void requires_grad(bool enabled);

  const Tensor* grad() const;

  void backward();
  friend class autograd::AutogradMeta;
  friend class GradNode;
  friend void autograd::backward(Tensor&);
  friend void autograd::accumulate(Tensor&, const Tensor&);

  std::shared_ptr<autograd::AutogradMeta>& autograd_meta();

  const std::shared_ptr<autograd::AutogradMeta>& autograd_meta() const;

 private:
  std::shared_ptr<float[]> storage_;

  std::vector<size_t> shape_;

  std::vector<size_t> stride_;

  size_t offset_;

  size_t storage_size_;

  std::shared_ptr<autograd::AutogradMeta> autograd_;

 private:
  static size_t checked_numel(std::span<const size_t> shape);

  static std::vector<size_t> contiguous_strides(std::span<const size_t> shape);

  size_t flat_index(std::span<const size_t> indices) const;

  void validate_view_bounds(std::span<const size_t> shape,
                            std::span<const size_t> stride,
                            size_t offset) const;
};

}  // namespace microtensor
