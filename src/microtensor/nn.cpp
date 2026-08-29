#include "microtensor/nn.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "microtensor/tensor_iterator.hpp"

namespace tensors::nn {

namespace {

Tensor causal_mask(size_t sequence_length) {
  Tensor mask = Tensor::zeros({sequence_length, sequence_length});
  for (size_t row = 0; row < sequence_length; ++row) {
    for (size_t column = row + 1; column < sequence_length; ++column) {
      mask.data()[row * sequence_length + column] = 1.0f;
    }
  }
  return mask;
}

}  // namespace

Dropout::Dropout(float probability) : probability_(probability) {
  if (probability < 0.0f || probability > 1.0f) {
    throw std::invalid_argument("Dropout probability must be in [0, 1]");
  }
}

Tensor Dropout::forward(const Tensor& x) {
  if (!is_training() || probability_ == 0.0f) {
    return x.clone();
  }
  if (probability_ == 1.0f) {
    return Tensor::zeros(x.shape());
  }

  Tensor result(x.shape());
  Tensor mask(x.shape());
  const float scale = 1.0f / (1.0f - probability_);
  static thread_local std::mt19937 generator(std::random_device{}());
  std::bernoulli_distribution keep(1.0 - probability_);
  TensorIterator<float, const float> iter(result, x);
  TensorIterator<float, const float, const float> mask_iter(mask, result, x);
  iter.for_each([&](float& out, const float& value) {
    out = keep(generator) ? value * scale : 0.0f;
  });
  mask_iter.for_each(
      [scale](float& mask_value, const float& output, const float& input) {
        mask_value = output == input * scale ? 1.0f : 0.0f;
      });

  if (AutogradContext::is_enabled() && x.requires_grad()) {
    result.set_requires_grad(true);
    auto parents = make_parents(x);
    auto backward_fn = [out = result, mask, scale](const auto& parents) {
      NoGradGuard guard;
      const auto& [input] = parents;
      if (!input.requires_grad()) {
        return;
      }
      Tensor grad(input.shape());
      TensorIterator<float, const float, const float> iter(grad, out.grad(),
                                                           mask);
      iter.for_each(
          [scale](float& dst, const float& upstream, const float& mask_value) {
            dst = mask_value == 0.0f ? 0.0f : upstream * scale;
          });
      input.add_grad(grad);
    };
    result.set_grad_fn(
        make_grad_node(std::move(parents), std::move(backward_fn)));
  }
  return result;
}

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim)
    : weight_(Tensor::zeros({num_embeddings, embedding_dim})) {
  weight_.set_requires_grad(true);
  register_parameters(namedparam("weight", &weight_));
}

Tensor Embedding::forward(const Tensor& x) {
  if (x.empty()) {
    std::vector<size_t> output_shape = x.shape();
    output_shape.push_back(weight_.shape().back());
    return Tensor::zeros(output_shape);
  }
  Tensor flat = x.view({x.numel()});
  Tensor selected = functional::index_select(weight_, 0, flat);
  std::vector<size_t> output_shape = x.shape();
  output_shape.push_back(weight_.shape().back());
  return selected.view(output_shape);
}

LayerNorm::LayerNorm(std::vector<size_t> normalized_shape, float eps)
    : normalized_shape_(std::move(normalized_shape)),
      eps_(eps),
      weight_(Tensor::ones(normalized_shape_)),
      bias_(Tensor::zeros(normalized_shape_)) {
  if (normalized_shape_.empty() || eps_ < 0.0f) {
    throw std::invalid_argument("LayerNorm: invalid shape or epsilon");
  }
  weight_.set_requires_grad(true);
  bias_.set_requires_grad(true);
  register_parameters(namedparam("weight", &weight_),
                      namedparam("bias", &bias_));
}

Tensor LayerNorm::forward(const Tensor& x) {
  return functional::layer_norm(x, normalized_shape_, weight_, bias_, eps_);
}

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads,
                                       float dropout)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : embed_dim / num_heads),
      query_projection_(embed_dim, embed_dim),
      key_projection_(embed_dim, embed_dim),
      value_projection_(embed_dim, embed_dim),
      output_projection_(embed_dim, embed_dim),
      dropout_(dropout) {
  if (embed_dim == 0 || num_heads == 0 || embed_dim % num_heads != 0) {
    throw std::invalid_argument(
        "MultiHeadAttention: embed_dim must be divisible by num_heads");
  }

  for (Linear* projection : {&query_projection_, &key_projection_,
                             &value_projection_, &output_projection_}) {
    projection->weight().set_requires_grad(true);
    projection->bias().set_requires_grad(true);
  }

  register_children(namedchild("query_projection", &query_projection_),
                    namedchild("key_projection", &key_projection_),
                    namedchild("value_projection", &value_projection_),
                    namedchild("output_projection", &output_projection_),
                    namedchild("dropout", &dropout_));
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
  if (x.ndim() != 2 && x.ndim() != 3) {
    throw std::invalid_argument(
        "MultiHeadAttention::forward(): expected [S,E] or [B,S,E]");
  }
  const size_t sequence_length = x.shape()[x.ndim() - 2];
  return forward_impl(x, x, causal_mask(sequence_length));
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& context,
                                   const Tensor& mask) {
  return forward_impl(query, context, mask);
}

Tensor MultiHeadAttention::forward_impl(const Tensor& query,
                                        const Tensor& context,
                                        const Tensor& mask) {
  if ((query.ndim() != 2 && query.ndim() != 3) ||
      (context.ndim() != 2 && context.ndim() != 3)) {
    throw std::invalid_argument(
        "MultiHeadAttention::forward(): expected [S,E] or [B,S,E]");
  }
  if (query.shape().back() != embed_dim_ ||
      context.shape().back() != embed_dim_) {
    throw std::invalid_argument(
        "MultiHeadAttention::forward(): embedding dimension mismatch");
  }
  if (query.ndim() != context.ndim() ||
      (query.ndim() == 3 && query.shape()[0] != context.shape()[0])) {
    throw std::invalid_argument(
        "MultiHeadAttention::forward(): query and context batch mismatch");
  }

  const bool was_unbatched = query.ndim() == 2;
  Tensor query_batched =
      was_unbatched ? query.view({1, query.shape()[0], embed_dim_}) : query;
  Tensor context_batched =
      context.ndim() == 2 ? context.view({1, context.shape()[0], embed_dim_})
                          : context;

  const size_t batch = query_batched.shape()[0];
  const size_t query_length = query_batched.shape()[1];
  const size_t context_length = context_batched.shape()[1];

  Tensor score_mask = mask;
  if (mask.ndim() == 2 && mask.shape()[0] == query_length &&
      mask.shape()[1] == context_length) {
    score_mask = mask.view({1, 1, query_length, context_length});
  } else if (mask.ndim() == 3 && mask.shape()[0] == batch &&
             mask.shape()[1] == query_length &&
             mask.shape()[2] == context_length) {
    score_mask = mask.view({batch, 1, query_length, context_length});
  } else if (mask.ndim() != 4) {
    throw std::invalid_argument(
        "MultiHeadAttention::forward(): invalid attention mask shape");
  }

  Tensor q = query_projection_.forward(query_batched)
                 .view({batch, query_length, num_heads_, head_dim_})
                 .permute({0, 2, 1, 3});
  Tensor k = key_projection_.forward(context_batched)
                 .view({batch, context_length, num_heads_, head_dim_})
                 .permute({0, 2, 1, 3});
  Tensor v = value_projection_.forward(context_batched)
                 .view({batch, context_length, num_heads_, head_dim_})
                 .permute({0, 2, 1, 3});

  Tensor scores =
      functional::mul(functional::matmul(q, k.transpose(-1, -2)),
                      1.0f / std::sqrt(static_cast<float>(head_dim_)));
  scores = functional::masked_fill(scores, score_mask,
                                   -std::numeric_limits<float>::infinity());
  Tensor probabilities = dropout_.forward(functional::softmax(scores, -1));
  Tensor attended = functional::matmul(probabilities, v)
                        .permute({0, 2, 1, 3})
                        .contiguous()
                        .view({batch, query_length, embed_dim_});
  Tensor output = output_projection_.forward(attended);
  return was_unbatched ? output.view({query_length, embed_dim_}) : output;
}

}  // namespace tensors::nn
