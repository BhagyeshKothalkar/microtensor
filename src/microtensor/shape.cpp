#include "shape.hpp"

#include <algorithm>
#include <stdexcept>

namespace microtensor {

namespace {

size_t aligned_dimension(const Tensor& tensor, size_t output_dim) {
  auto rank = tensor.ndim();

  if (output_dim < rank) {
    return tensor.shape()[rank - output_dim - 1];
  }

  return 1;
}

std::vector<size_t> broadcast_strides(const Tensor& input,
                                      std::span<const size_t> target) {
  std::vector<size_t> stride(target.size());

  size_t in_rank = input.ndim();

  size_t offset = target.size() - in_rank;

  for (size_t i = 0; i < target.size(); ++i) {
    if (i < offset) {
      stride[i] = 0;
      continue;
    }

    size_t in_dim = i - offset;

    if (input.shape()[in_dim] == 1 && target[i] != 1) {
      stride[i] = 0;
    } else {
      stride[i] = input.stride()[in_dim];
    }
  }

  return stride;
}

}  // namespace

std::vector<size_t> broadcast_shape(std::span<const Tensor* const> tensors) {
  size_t rank = 0;

  for (auto* tensor : tensors) {
    rank = std::max(rank, tensor->ndim());
  }

  std::vector<size_t> result(rank, 1);

  for (auto* tensor : tensors) {
    size_t offset = rank - tensor->ndim();

    for (size_t i = 0; i < tensor->ndim(); ++i) {
      size_t out_dim = offset + i;

      size_t current = result[out_dim];

      size_t incoming = tensor->shape()[i];

      if (current == 1) {
        result[out_dim] = incoming;
      } else if (incoming == 1 || incoming == current) {
        continue;
      } else {
        throw std::runtime_error("incompatible broadcast dimensions");
      }
    }
  }

  return result;
}

Tensor broadcast_to(const Tensor& input, std::span<const size_t> target_shape) {
  auto input_shape = input.shape();

  size_t target_rank = target_shape.size();

  if (input.ndim() > target_rank) {
    throw std::runtime_error("cannot broadcast to lower rank");
  }

  auto stride = broadcast_strides(input, target_shape);

  return input.as_strided(target_shape, stride, 0);
}

Tensor sum_to_shape(const Tensor& input, std::span<const size_t> target_shape) {
  if (std::ranges::equal(input.shape(), target_shape)) {
    return input;
  }

  Tensor result = Tensor::zeros(target_shape);

  auto out_shape = result.shape();

  size_t input_rank = input.ndim();

  size_t target_rank = result.ndim();

  if (target_rank > input_rank) {
    throw std::runtime_error("invalid reduction shape");
  }

  size_t offset = input_rank - target_rank;

  for (size_t linear = 0; linear < input.numel(); ++linear) {
    size_t tmp = linear;

    size_t out_offset = 0;

    for (size_t dim = input_rank; dim-- > 0;) {
      size_t coord = tmp % input.shape()[dim];

      tmp /= input.shape()[dim];

      if (dim >= offset) {
        size_t out_dim = dim - offset;

        if (out_shape[out_dim] != 1) {
          out_offset += coord * result.stride()[out_dim];
        }
      }
    }

    result.data()[out_offset] += input.data()[linear];
  }

  return result;
}

}  // namespace microtensor
