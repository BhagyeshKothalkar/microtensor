#include <iostream>
#include <memory>
#include <vector>

#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optimizer.hpp"

using namespace tensors;
using namespace tensors::nn;

namespace {

class GPT2Block : public nn::Module {
 private:
  nn::LayerNorm layer_norm_1_;
  nn::MultiHeadAttention attention_;
  nn::LayerNorm layer_norm_2_;
  nn::Linear feed_forward_1_;
  nn::Linear feed_forward_2_;

 public:
  GPT2Block(size_t embedding_dim, size_t num_heads, size_t hidden_dim)
      : layer_norm_1_(std::vector<size_t>{embedding_dim}),
        attention_(embedding_dim, num_heads),
        layer_norm_2_(std::vector<size_t>{embedding_dim}),
        feed_forward_1_(embedding_dim, hidden_dim),
        feed_forward_2_(hidden_dim, embedding_dim) {
    for (Tensor* parameter : layer_norm_1_.parameters_recursive()) {
      parameter->set_requires_grad(true);
    }
    for (Tensor* parameter : layer_norm_2_.parameters_recursive()) {
      parameter->set_requires_grad(true);
    }
    feed_forward_1_.weight().set_requires_grad(true);
    feed_forward_1_.bias().set_requires_grad(true);
    feed_forward_2_.weight().set_requires_grad(true);
    feed_forward_2_.bias().set_requires_grad(true);

    register_children(namedchild("attention", &attention_),
                      namedchild("layer_norm_1", &layer_norm_1_),
                      namedchild("layer_norm_2", &layer_norm_2_),
                      namedchild("feed_forward_1", &feed_forward_1_),
                      namedchild("feed_forward_2", &feed_forward_2_));
  }

  Tensor forward(const Tensor& x) override {
    Tensor attended = attention_.forward(layer_norm_1_.forward(x));
    Tensor residual = functional::add(x, attended);
    Tensor hidden = functional::gelu(
        feed_forward_1_.forward(layer_norm_2_.forward(residual)));
    return functional::add(residual, feed_forward_2_.forward(hidden));
  }
};

class GPT2 : public nn::Module {
 private:
  size_t context_length_;
  nn::Embedding token_embedding_;
  nn::Embedding position_embedding_;
  std::vector<std::unique_ptr<GPT2Block>> blocks_;
  nn::LayerNorm final_layer_norm_;
  nn::Linear vocabulary_projection_;

 public:
  GPT2(size_t vocabulary_size, size_t context_length, size_t embedding_dim,
       size_t num_heads, size_t num_layers)
      : context_length_(context_length),
        token_embedding_(vocabulary_size, embedding_dim),
        position_embedding_(context_length, embedding_dim),
        final_layer_norm_(std::vector<size_t>{embedding_dim}),
        vocabulary_projection_(embedding_dim, vocabulary_size) {
    for (size_t layer = 0; layer < num_layers; ++layer) {
      auto block = std::make_unique<GPT2Block>(embedding_dim, num_heads,
                                               embedding_dim * 4);
      register_children(namedchild("block", block.get()));
      blocks_.push_back(std::move(block));
    }

    register_children(
        namedchild("token_embedding", &token_embedding_),
        namedchild("position_embedding", &position_embedding_),
        namedchild("final_layer_norm", &final_layer_norm_),
        namedchild("vocabulary_projection", &vocabulary_projection_));

    vocabulary_projection_.weight().set_requires_grad(true);
    vocabulary_projection_.bias().set_requires_grad(true);
    for (Tensor* parameter : final_layer_norm_.parameters_recursive()) {
      parameter->set_requires_grad(true);
    }
  }

  Tensor forward(const Tensor& tokens) override {
    if (tokens.ndim() != 1 && tokens.ndim() != 2) {
      throw std::invalid_argument("GPT2 expects [S] or [B,S] token IDs");
    }
    const size_t sequence_length = tokens.shape().back();
    if (sequence_length > context_length_) {
      throw std::invalid_argument("GPT2 sequence exceeds context length");
    }

    Tensor positions({sequence_length});
    for (size_t i = 0; i < sequence_length; ++i) {
      positions.data()[i] = static_cast<float>(i);
    }

    Tensor hidden = functional::add(token_embedding_.forward(tokens),
                                    position_embedding_.forward(positions));
    for (const auto& block : blocks_) {
      hidden = block->forward(hidden);
    }
    return vocabulary_projection_.forward(final_layer_norm_.forward(hidden));
  }
};

}  // namespace

int main() {
  constexpr size_t vocabulary_size = 8;
  constexpr size_t context_length = 4;
  constexpr size_t batch_size = 2;

  GPT2 model(vocabulary_size, context_length, 8, 2, 2);
  optim::Adam optimizer(model.parameters_recursive(), 1e-2f);

  Tensor input({batch_size, context_length}, {0, 1, 2, 3, 1, 2, 3, 4});
  Tensor targets({batch_size, context_length}, {1, 2, 3, 4, 2, 3, 4, 5});

  for (int step = 0; step < 20; ++step) {
    Tensor logits = model.forward(input);
    Tensor loss = functional::cross_entropy(logits, targets);

    std::cout << "step " << step << " loss=" << loss.data()[0] << '\n';
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
  }
}
