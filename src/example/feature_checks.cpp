#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"

using namespace tensors;

namespace {

int failures = 0;

std::string shape_to_string(const std::vector<size_t>& shape) {
  std::string out = "[";

  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }

  out += "]";
  return out;
}

void fail(std::string_view name, std::string_view message) {
  std::cerr << "[FAIL] " << name << ": " << message << '\n';
  ++failures;
}

void expect_close(float got, float expected, std::string_view name,
                  float tolerance = 1e-4f) {
  if (std::fabs(got - expected) > tolerance) {
    fail(name, std::string("got=") + std::to_string(got) +
                   " expected=" + std::to_string(expected));
  } else {
    std::cout << "[PASS] " << name << '\n';
  }
}

void expect_shape(const Tensor& tensor, const std::vector<size_t>& expected,
                  std::string_view name) {
  if (tensor.shape() != expected) {
    fail(name, "shape mismatch: got " + shape_to_string(tensor.shape()) +
                   ", expected " + shape_to_string(expected));
    return;
  }

  std::cout << "[PASS] " << name << '\n';
}

float scalar_value(const Tensor& tensor) {
  if (tensor.numel() != 1) {
    throw std::runtime_error("expected scalar tensor");
  }

  return tensor.data()[0];
}

template <typename LossFn>
Tensor finite_difference_gradient(const Tensor& input, LossFn&& loss_fn,
                                  float epsilon = 1e-3f) {
  Tensor gradient(input.shape());

  for (size_t i = 0; i < input.numel(); ++i) {
    Tensor plus = input.clone();
    Tensor minus = input.clone();

    plus.data()[i] += epsilon;
    minus.data()[i] -= epsilon;

    const float loss_plus = scalar_value(loss_fn(plus));
    const float loss_minus = scalar_value(loss_fn(minus));

    gradient.data()[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
  }

  return gradient;
}

template <typename LossFn>
void expect_scalar_loss_gradient_close(std::string_view name,
                                       const Tensor& input, LossFn&& loss_fn,
                                       float epsilon = 1e-3f,
                                       float tolerance = 1e-2f) {
  Tensor autograd_input = input.clone();
  autograd_input.set_requires_grad(true);

  Tensor loss = loss_fn(autograd_input);

  if (loss.ndim() != 0) {
    fail(name, "loss must be scalar: got " + shape_to_string(loss.shape()) +
                   ", expected []");
    return;
  }

  loss.backward();

  const Tensor numerical = finite_difference_gradient(input, loss_fn, epsilon);
  const Tensor& analytical = autograd_input.grad();

  if (analytical.shape() != numerical.shape()) {
    fail(name, "gradient shape mismatch");
    return;
  }

  for (size_t i = 0; i < analytical.numel(); ++i) {
    expect_close(analytical.data()[i], numerical.data()[i],
                 std::string(name) + " grad[" + std::to_string(i) + "]",
                 tolerance);
  }
}

void check_max_reduction() {
  Tensor values({2, 3}, {1, 8, 3, 4, 5, 9});
  Tensor result = functional::max(values, {1});

  expect_shape(result, {2}, "max reduces selected dimension");
  expect_close(result[0], 8.0f, "max row 0");
  expect_close(result[1], 9.0f, "max row 1");
}

void check_softmax_and_logsoftmax() {
  Tensor logits({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor probabilities = functional::softmax(logits, -1);
  Tensor log_probabilities = functional::logsoftmax(logits, -1);

  expect_shape(probabilities, {2, 3}, "softmax preserves shape");
  expect_close(probabilities.data()[0] + probabilities.data()[1] +
                   probabilities.data()[2],
               1.0f, "softmax row 0 sums to one");
  expect_close(probabilities.data()[3] + probabilities.data()[4] +
                   probabilities.data()[5],
               1.0f, "softmax row 1 sums to one");
  for (size_t i = 0; i < logits.numel(); ++i) {
    expect_close(std::exp(log_probabilities.data()[i]), probabilities.data()[i],
                 "logsoftmax exponent matches softmax");
  }
}

void check_max_gradient() {
  Tensor input({2, 3}, {1, 8, 3, 4, 5, 9});
  expect_scalar_loss_gradient_close(
      "max reduction gradient", input, [](const Tensor& x) {
        return functional::sum(functional::max(x, {1}), {0});
      });
}

void check_softmax_gradients() {
  Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor weights({2, 3}, {1, -2, 3, -1, 2, -3});

  expect_scalar_loss_gradient_close(
      "softmax gradient", input, [&weights](const Tensor& x) {
        return functional::sum(
            functional::mul(functional::softmax(x, -1), weights), {0, 1});
      });
  expect_scalar_loss_gradient_close(
      "logsoftmax gradient", input, [&weights](const Tensor& x) {
        return functional::sum(
            functional::mul(functional::logsoftmax(x, -1), weights), {0, 1});
      });
}

void check_tensor_data_movement() {
  Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor mask({2, 3}, {0, 1, 0, 1, 0, 1});
  Tensor filled = functional::masked_fill(input, mask, -1.0f);
  expect_close(filled.data()[0], 1.0f, "masked_fill keeps false entries");
  expect_close(filled.data()[1], -1.0f, "masked_fill replaces true entries");

  Tensor indices({2}, {2, 0});
  Tensor selected = functional::index_select(input, 1, indices);
  expect_shape(selected, {2, 2}, "index_select replaces selected dimension");
  expect_close(selected.data()[0], 3.0f, "index_select first value");
  expect_close(selected.data()[1], 1.0f, "index_select second value");

  Tensor concatenated = functional::cat({input, input}, 0);
  expect_shape(concatenated, {4, 3}, "cat extends selected dimension");
  expect_close(concatenated.data()[3], 4.0f, "cat copies second tensor");

  expect_scalar_loss_gradient_close(
      "masked_fill gradient", input, [&mask](const Tensor& x) {
        return functional::sum(functional::masked_fill(x, mask, -1.0f), {0, 1});
      });
  expect_scalar_loss_gradient_close(
      "index_select gradient", input, [&indices](const Tensor& x) {
        return functional::sum(functional::index_select(x, 1, indices), {0, 1});
      });
  expect_scalar_loss_gradient_close("cat gradient", input, [](const Tensor& x) {
    return functional::sum(functional::cat({x, x}, 0), {0, 1});
  });
}

void check_nn_modules_and_losses() {
  nn::Sequential model;
  nn::Dropout& child_dropout = model.emplace<nn::Dropout>(0.5f);
  model.eval();
  if (model.is_training() || child_dropout.is_training()) {
    fail("module eval propagation", "eval state did not reach child module");
  }
  model.train();
  if (!model.is_training() || !child_dropout.is_training()) {
    fail("module train propagation", "train state did not reach child module");
  }

  nn::Dropout dropout(0.5f);
  Tensor input = Tensor::ones({128});
  dropout.eval();
  Tensor evaluation = dropout.forward(input);
  expect_close(evaluation.data()[0], 1.0f, "dropout eval is identity");
  dropout.train();
  Tensor training = dropout.forward(input);
  bool saw_zero = false;
  for (size_t i = 0; i < training.numel(); ++i) {
    saw_zero = saw_zero || training.data()[i] == 0.0f;
  }
  if (!saw_zero) {
    fail("dropout training mask", "training dropout produced no masked values");
  }

  nn::Embedding embedding(3, 2);
  Tensor tokens({2}, {2, 0});
  Tensor embeddings = embedding.forward(tokens);
  expect_shape(embeddings, {2, 2}, "embedding appends embedding dimension");
  expect_close(embeddings.data()[0], embedding.weight().data()[4],
               "embedding selects token row");
  Tensor embedding_loss = functional::sum(embeddings, {0, 1});
  embedding_loss.backward();
  expect_close(embedding.weight().grad().data()[4], 1.0f,
               "embedding accumulates selected row gradient");

  nn::LayerNorm norm({3});
  Tensor norm_input({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor normalized = norm.forward(norm_input);
  expect_shape(normalized, {2, 3}, "layer norm preserves shape");
  expect_close(functional::mean(normalized, {1}).data()[0], 0.0f,
               "layer norm row mean");

  Tensor logits({2, 3}, {1, 2, 3, 3, 2, 1});
  Tensor targets({2}, {2, 0});
  Tensor loss = functional::cross_entropy(logits, targets);
  expect_shape(loss, {}, "cross entropy is scalar");
  expect_scalar_loss_gradient_close(
      "cross entropy gradient", logits, [&targets](const Tensor& x) {
        return functional::cross_entropy(x, targets);
      });
}

void check_multihead_attention() {
  nn::MultiHeadAttention attention(4, 2);
  for (Tensor* parameter : attention.parameters_recursive()) {
    std::fill_n(parameter->data(), parameter->numel(), 0.0f);
    if (parameter->ndim() == 2) {
      const size_t diagonal =
          std::min(parameter->shape()[0], parameter->shape()[1]);
      for (size_t i = 0; i < diagonal; ++i) {
        parameter->data()[i * parameter->shape()[1] + i] = 1.0f;
      }
    }
  }
  Tensor input({1, 3, 4}, {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6});
  Tensor causal_output = attention.forward(input);
  expect_shape(causal_output, {1, 3, 4},
               "causal attention preserves batched shape");
  Tensor changed_future = input.clone();
  std::fill_n(changed_future.data() + 4, 8, 100.0f);
  Tensor changed_output = attention.forward(changed_future);
  for (size_t i = 0; i < 4; ++i) {
    expect_close(causal_output.data()[i], changed_output.data()[i],
                 "causal attention blocks future token");
  }

  Tensor query({1, 2, 4}, {1, 2, 3, 4, 4, 3, 2, 1});
  Tensor context({1, 3, 4}, {1, 0, 2, 0, 0, 1, 0, 2, 2, 1, 0, 1});
  Tensor mask({2, 3}, {0, 0, 1, 0, 0, 0});
  Tensor cross_output = attention.forward(query, context, mask);
  expect_shape(cross_output, {1, 2, 4},
               "cross attention preserves query shape");

  expect_scalar_loss_gradient_close(
      "multihead attention gradient", input, [&attention](const Tensor& x) {
        return functional::sum(attention.forward(x), {0, 1, 2});
      });
}

void check_scalar_loss_gradient_helper() {
  Tensor input({2, 3}, {1, 2, 3, 4, 5, 6});

  expect_scalar_loss_gradient_close(
      "sum of squares gradient", input, [](const Tensor& x) {
        Tensor squared = functional::mul(x, x);
        auto ret = functional::sum(squared, {0, 1}, false);
        return ret;
      });
}

}  // namespace

int main() {
  check_max_reduction();
  check_softmax_and_logsoftmax();
  check_max_gradient();
  check_softmax_gradients();
  check_tensor_data_movement();
  check_nn_modules_and_losses();
  check_multihead_attention();
  check_scalar_loss_gradient_helper();

  if (failures != 0) {
    std::cerr << failures << " checks failed\n";
    return 1;
  }

  std::cout << "all feature checks passed\n";
  return 0;
}
