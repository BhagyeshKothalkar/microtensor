#include "microtensor/functional.hpp"
#include "microtensor/tensor_iterator.hpp"

namespace tensors {

[[maybe_unused]] static Tensor legacy_sum_to_shape(const Tensor& input,
                                  const std::vector<size_t>& target_shape) {
  const auto& input_shape = input.shape();

  if (input_shape == target_shape) {
    return input;
  }

  const size_t input_rank = input_shape.size();
  const size_t target_rank = target_shape.size();

  if (target_rank > input_rank) {
    throw std::runtime_error(
        "sum_to_shape(): target rank cannot exceed input rank");
  }

  for (size_t i = 0; i < target_rank; ++i) {
    const size_t input_dim = input_shape[input_rank - 1 - i];
    const size_t target_dim = target_shape[target_rank - 1 - i];

    if (target_dim != 1 && target_dim != input_dim) {
      throw std::runtime_error("sum_to_shape(): incompatible shapes");
    }
  }

  Tensor result(target_shape);
  TensorIterator<float>(result).for_each([](auto& value) { value = 0; });
  std::vector<size_t> broadcast_strides(input_rank, 0);
  const auto& result_strides = result.stride();

  for (size_t i = 0; i < input_rank; ++i) {
    const size_t input_dim = input_rank - 1 - i;

    if (i < target_rank) {
      const size_t target_dim = target_rank - 1 - i;

      if (target_shape[target_dim] == input_shape[input_dim]) {
        broadcast_strides[input_dim] = result_strides[target_dim];
      } else {
        broadcast_strides[input_dim] = 0;
      }
    } else {
      broadcast_strides[input_dim] = 0;
    }
  }

  Tensor broadcast_result(input.shape(), broadcast_strides, result.storage(),
                          result.storage_size(), result.offset());

  TensorIterator<float, const float>(broadcast_result, input)
      .for_each([](auto& dst, const auto& src) { dst += src; });

  return result;
}

Tensor broadcast_to_shape(const Tensor& in,
                          const std::vector<size_t>& target_shape) {
  std::vector<size_t> new_strides(target_shape.size(), 0);

  const auto& curr_shape = in.shape();
  const auto& curr_strides = in.stride();

  const size_t target_rank = target_shape.size();
  const size_t curr_rank = curr_shape.size();

  if (curr_rank > target_rank) {
    throw std::runtime_error("broadcast_to_shape(): incompatible shapes");
  }

  for (size_t i = 0; i < target_rank; ++i) {
    if (i < curr_rank) {
      const size_t curr_dim_size = curr_shape[curr_rank - 1 - i];
      const size_t target_dim_size = target_shape[target_rank - 1 - i];

      if (curr_dim_size == target_dim_size) {
        new_strides[target_rank - 1 - i] = curr_strides[curr_rank - 1 - i];
      } else if (curr_dim_size == 1) {
        new_strides[target_rank - 1 - i] = 0;
      } else {
        throw std::runtime_error("broadcast_to_shape(): incompatible shapes");
      }
    } else {
      new_strides[target_rank - 1 - i] = 0;
    }
  }

  Tensor result(target_shape, new_strides, in.storage(), in.storage_size(),
                in.offset());

  if (AutogradContext::is_enabled() && in.requires_grad()) {
    auto parents = make_parents(in);
    auto backward_fn = [out = result](auto& parents) {
      NoGradGuard guard;

      const auto& [input] = parents;
      const Tensor grad = out.grad();

      if (input.requires_grad()) {
        auto input_grad = functional::detail::sum_to_shape(grad, input.shape());
        input.add_grad(input_grad);
      }
    };

    result.set_requires_grad(true);
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }

  return result;
}
};  // namespace tensors
