#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "microtensor/tensor.hpp"

namespace tensors {

template <typename Dest, typename... Src>
class TensorIterator {
 private:
  static constexpr std::size_t n_terms = sizeof...(Src) + 1;

  std::tuple<Dest*, Src*...> data_;
  std::vector<std::size_t> shapes_;
  std::array<std::vector<std::size_t>, n_terms> strides_;
  std::vector<std::size_t> idx_;
  std::array<std::size_t, n_terms> offsets_{};
  bool has_next_flag_;

  template <std::size_t... Is>
  auto dereference(std::index_sequence<Is...>);

 public:
  template <typename... TensorTypes>
  TensorIterator(Tensor& dest, const TensorTypes&... srcs);

  template <typename... TensorTypes>
  TensorIterator(const Tensor& dest, const TensorTypes&... srcs);

  void set_base_offsets(const std::array<std::size_t, n_terms>& offsets);

  // Restrict iteration to a set of dimensions of the operands.  The stride
  // vectors are dimension-major, just like the shape passed here.
  void set_iteration(const std::vector<std::size_t>& shape,
                     const std::vector<std::vector<std::size_t>>& strides);

  bool has_next() const;
  auto next();

  void for_each(std::invocable<Dest&, Src&...> auto fn);
};

template <typename Dest, typename... Src>
class ReductionIterator {
 private:
  static constexpr std::size_t n_srcs = sizeof...(Src);

  Dest* dest_;

  // Tensor copies retain the shared storage and keep temporary sources alive
  // for the lifetime of the reduction iterator.
  std::vector<Tensor> source_tensors_;

  std::vector<std::size_t> output_shape_;
  std::vector<std::size_t> output_idx_;

  std::vector<std::vector<std::size_t>> output_strides_;

  std::vector<std::size_t> reduce_shape_;
  std::vector<std::vector<std::size_t>> reduce_strides_;

  std::vector<int> reduce_dims_;

  std::vector<std::size_t> output_strides_dest_;
  std::size_t output_offset_;

  bool has_next_;

  TensorIterator<const Src...> reduce_iter_;

  std::array<std::size_t, n_srcs> current_reduce_offsets_{};

  void advance_output();

  static TensorIterator<const Src...> make_reduce_iterator(
      const std::vector<Tensor>& sources);

  template <std::size_t... Is>
  static TensorIterator<const Src...> make_reduce_iterator_impl(
      const std::vector<Tensor>& sources, std::index_sequence<Is...>);

 public:
  template <typename... TensorTypes>
  ReductionIterator(const std::vector<int>& reduce_dims, Tensor& dest,
                    const TensorTypes&... srcs);

  bool has_next() const;

  auto next();

  void for_each(std::invocable<Dest&, TensorIterator<const Src...>> auto fn);
};

template <typename Dest, typename... Src>
template <std::size_t... Is>
auto TensorIterator<Dest, Src...>::dereference(std::index_sequence<Is...>) {
  return std::tuple<Dest&, Src&...>{std::get<Is>(data_)[offsets_[Is]]...};
}

template <typename Dest, typename... Src>
template <typename... TensorTypes>
TensorIterator<Dest, Src...>::TensorIterator(Tensor& dest,
                                             const TensorTypes&... srcs)
    : data_(std::make_tuple(dest.data(), srcs.data()...)),
      shapes_(dest.shape()),
      strides_({dest.stride(), srcs.stride()...}),
      idx_(dest.ndim(), 0),
      offsets_{},
      has_next_flag_(!dest.empty()) {
  auto validate = [&dest](const auto& tensor) {
    if (tensor.shape() != dest.shape() ||
        tensor.stride().size() != dest.ndim()) {
      throw std::invalid_argument(
          "TensorIterator operands must match the destination shape and rank");
    }
  };
  if (dest.stride().size() != dest.ndim()) {
    throw std::invalid_argument(
        "TensorIterator destination has invalid strides");
  }
  (validate(srcs), ...);
  offsets_.fill(0);
}

template <typename Dest, typename... Src>
template <typename... TensorTypes>
TensorIterator<Dest, Src...>::TensorIterator(const Tensor& dest,
                                             const TensorTypes&... srcs)
    : data_(std::make_tuple(dest.data(), srcs.data()...)),
      shapes_(dest.shape()),
      strides_({dest.stride(), srcs.stride()...}),
      idx_(dest.ndim(), 0),
      offsets_{},
      has_next_flag_(!dest.empty()) {
  auto validate = [&dest](const auto& tensor) {
    if (tensor.shape() != dest.shape() ||
        tensor.stride().size() != dest.ndim()) {
      throw std::invalid_argument(
          "TensorIterator operands must match the destination shape and rank");
    }
  };
  if (dest.stride().size() != dest.ndim()) {
    throw std::invalid_argument(
        "TensorIterator destination has invalid strides");
  }
  (validate(srcs), ...);
  offsets_.fill(0);
}

template <typename Dest, typename... Src>
bool TensorIterator<Dest, Src...>::has_next() const {
  return has_next_flag_;
}

template <typename Dest, typename... Src>
auto TensorIterator<Dest, Src...>::next() {
  assert(has_next_flag_);

  auto curr = dereference(std::make_index_sequence<n_terms>{});

  if (idx_.empty()) {
    has_next_flag_ = false;
    return curr;
  }

  bool advanced = false;

  for (std::size_t dim = idx_.size(); dim-- > 0;) {
    ++idx_[dim];

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] += strides_[Is][dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    if (idx_[dim] < shapes_[dim]) {
      advanced = true;
      break;
    }

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((offsets_[Is] -= strides_[Is][dim] * idx_[dim]), ...);
    }(std::make_index_sequence<n_terms>{});

    idx_[dim] = 0;
  }

  if (!advanced) {
    has_next_flag_ = false;
  }

  return curr;
}

template <typename Dest, typename... Src>
void TensorIterator<Dest, Src...>::set_base_offsets(
    const std::array<std::size_t, n_terms>& offsets) {
  offsets_ = offsets;
  idx_.assign(shapes_.size(), 0);
  has_next_flag_ = true;
  for (auto extent : shapes_) {
    if (extent == 0) {
      has_next_flag_ = false;
      break;
    }
  }
}

template <typename Dest, typename... Src>
void TensorIterator<Dest, Src...>::set_iteration(
    const std::vector<std::size_t>& shape,
    const std::vector<std::vector<std::size_t>>& strides) {
  if (strides.size() != shape.size()) {
    throw std::invalid_argument("TensorIterator shape and stride ranks differ");
  }

  for (const auto& dimension_strides : strides) {
    if (dimension_strides.size() != n_terms) {
      throw std::invalid_argument("TensorIterator stride count is invalid");
    }
  }

  shapes_ = shape;
  for (std::size_t term = 0; term < n_terms; ++term) {
    strides_[term].resize(shape.size());
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
      strides_[term][dim] = strides[dim][term];
    }
  }
  idx_.assign(shape.size(), 0);
}

template <typename Dest, typename... Src>
void TensorIterator<Dest, Src...>::for_each(
    std::invocable<Dest&, Src&...> auto fn) {
  while (has_next()) {
    auto values = next();
    std::apply(fn, values);
  }
}

template <typename Dest, typename... Src>
template <typename... TensorTypes>
ReductionIterator<Dest, Src...>::ReductionIterator(
    const std::vector<int>& reduce_dims, Tensor& dest,
    const TensorTypes&... srcs)
    : dest_(dest.data()),
      source_tensors_{srcs...},
      output_idx_(),
      output_offset_(0),
      has_next_(!dest.empty()),
      reduce_iter_(make_reduce_iterator(source_tensors_)) {
  static_assert(n_srcs > 0, "ReductionIterator requires a source tensor");

  const auto& first_src = source_tensors_.front();
  std::vector<bool> reduced(first_src.ndim(), false);

  reduce_dims_.clear();
  for (auto raw_dim : reduce_dims) {
    auto dim = raw_dim;
    if (dim < 0) {
      dim += static_cast<int>(first_src.ndim());
    }
    if (dim < 0 || dim >= static_cast<int>(first_src.ndim())) {
      throw std::out_of_range("reduction dimension is out of range");
    }
    if (reduced[dim]) {
      throw std::invalid_argument("duplicate reduction dimension");
    }
    reduced[dim] = true;
    reduce_dims_.push_back(dim);
  }

  if (dest.ndim() != first_src.ndim() - reduce_dims_.size()) {
    throw std::invalid_argument(
        "ReductionIterator destination rank does not match the reduction "
        "output");
  }

  for (std::size_t i = 0; i < first_src.ndim(); ++i) {
    if (!reduced[i]) {
      output_shape_.push_back(first_src.shape()[i]);

      output_strides_.push_back(
          [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::vector<std::size_t>{source_tensors_[Is].stride()[i]...};
          }(std::make_index_sequence<n_srcs>{}));

      output_strides_dest_.push_back(
          dest.stride()[output_strides_dest_.size()]);
    }
  }

  for (auto d : reduce_dims_) {
    reduce_shape_.push_back(first_src.shape()[d]);

    reduce_strides_.push_back(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::vector<std::size_t>{source_tensors_[Is].stride()[d]...};
        }(std::make_index_sequence<n_srcs>{}));
  }

  output_idx_.assign(output_shape_.size(), 0);

  if (dest.shape() != output_shape_) {
    throw std::invalid_argument(
        "ReductionIterator destination shape does not match the reduction "
        "output");
  }

  current_reduce_offsets_.fill(0);

  reduce_iter_.set_iteration(reduce_shape_, reduce_strides_);
  reduce_iter_.set_base_offsets(current_reduce_offsets_);
}

template <typename Dest, typename... Src>
TensorIterator<const Src...>
ReductionIterator<Dest, Src...>::make_reduce_iterator(
    const std::vector<Tensor>& sources) {
  return make_reduce_iterator_impl(sources, std::make_index_sequence<n_srcs>{});
}

template <typename Dest, typename... Src>
template <std::size_t... Is>
TensorIterator<const Src...>
ReductionIterator<Dest, Src...>::make_reduce_iterator_impl(
    const std::vector<Tensor>& sources, std::index_sequence<Is...>) {
  return TensorIterator<const Src...>(sources[Is]...);
}

template <typename Dest, typename... Src>
bool ReductionIterator<Dest, Src...>::has_next() const {
  return has_next_;
}

template <typename Dest, typename... Src>
auto ReductionIterator<Dest, Src...>::next() {
  assert(has_next_);

  auto inner = reduce_iter_;

  auto result = std::tuple<Dest&, TensorIterator<const Src...>>{
      *(dest_ + output_offset_), inner};

  advance_output();

  return result;
}

template <typename Dest, typename... Src>
void ReductionIterator<Dest, Src...>::advance_output() {
  if (output_idx_.empty()) {
    has_next_ = false;
    return;
  }

  bool advanced = false;

  for (std::size_t dim = output_idx_.size(); dim-- > 0;) {
    ++output_idx_[dim];

    if (output_idx_[dim] < output_shape_[dim]) {
      advanced = true;
      break;
    }

    output_idx_[dim] = 0;
  }

  if (!advanced) {
    has_next_ = false;
    return;
  }

  output_offset_ = 0;

  for (std::size_t i = 0; i < output_idx_.size(); ++i) {
    output_offset_ += output_idx_[i] * output_strides_dest_[i];
  }

  current_reduce_offsets_.fill(0);

  for (std::size_t dim = 0; dim < output_idx_.size(); ++dim) {
    auto idx = output_idx_[dim];

    for (std::size_t t = 0; t < n_srcs; ++t) {
      current_reduce_offsets_[t] += idx * output_strides_[dim][t];
    }
  }

  reduce_iter_.set_base_offsets(current_reduce_offsets_);
}

template <typename Dest, typename... Src>
void ReductionIterator<Dest, Src...>::for_each(
    std::invocable<Dest&, TensorIterator<const Src...>> auto fn) {
  while (has_next()) {
    auto [dst, src] = next();

    fn(dst, src);
  }
}

}  // namespace tensors
