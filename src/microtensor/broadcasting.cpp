#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"

namespace tensors {

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
