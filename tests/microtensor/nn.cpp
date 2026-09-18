#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "microtensor/tensor.hpp"
#include "test_tensor.h"

using namespace tensors;
using namespace tensors::nn;

namespace {
class AddOneModule : public Module {
 public:
  Tensor forward(const Tensor& x) override {
    Tensor result = x.clone();
    for (size_t i = 0; i < result.numel(); ++i) result.data()[i] += 1.0f;
    return result;
  }
};

void expect_values(const Tensor& actual, const std::vector<float>& expected,
                   float tolerance = 1e-5f) {
  ASSERT_EQ(actual.numel(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(actual.data()[i], expected[i], tolerance) << "index " << i;
}

void initialize_attention(MultiHeadAttention& attention) {
  for (Tensor* parameter : attention.parameters_recursive()) {
    ASSERT_NE(parameter, nullptr);
    std::fill_n(parameter->data(), parameter->numel(), 0.0f);
    if (parameter->ndim() == 2) {
      const size_t diagonal =
          std::min(parameter->shape()[0], parameter->shape()[1]);
      for (size_t i = 0; i < diagonal; ++i)
        parameter->data()[i * parameter->shape()[1] + i] = 1.0f;
    }
  }
}
}  // namespace

TEST_F(TensorTests, ModuleRegistrationAndModePropagation) {
  Sequential model;
  Dropout& dropout = model.emplace<Dropout>(0.5f);
  model.emplace<AddOneModule>();
  ASSERT_EQ(model.children().size(), 2u);
  EXPECT_EQ(model.children()[0].first, "0");
  EXPECT_EQ(model.children()[1].first, "1");
  model.eval();
  EXPECT_FALSE(model.is_training());
  EXPECT_FALSE(dropout.is_training());
  model.train();
  EXPECT_TRUE(model.is_training());
  EXPECT_TRUE(dropout.is_training());
}

TEST_F(TensorTests, SequentialAndLinearHaveDeterministicForwardValues) {
  auto linear = std::make_unique<Linear>(2, 2);
  linear->weight().set_requires_grad(true);
  linear->bias().set_requires_grad(true);
  linear->weight() = Tensor({2, 2}, {1, 2, 3, 4});
  linear->bias() = Tensor({2}, {0.5f, -1.0f});
  Sequential model(std::move(linear));
  Tensor output = model.forward(Tensor({1, 2}, {2, 3}));
  EXPECT_EQ(output.shape(), (std::vector<size_t>{1, 2}));
  expect_values(output, {11.5f, 15.0f});
  ASSERT_EQ(model.named_parameters_recursive().size(), 2u);
  EXPECT_EQ(model.named_parameters_recursive()[0].first, "0.weight");
  EXPECT_EQ(model.named_parameters_recursive()[1].first, "0.bias");
}

TEST_F(TensorTests, DropoutEvalIsIdentityAndTrainingMasksWithoutChangingShape) {
  Dropout dropout(0.5f);
  Tensor input = Tensor::ones({256});
  dropout.eval();
  Tensor evaluation = dropout.forward(input);
  EXPECT_EQ(evaluation.shape(), input.shape());
  expect_values(evaluation, std::vector<float>(256, 1.0f));
  dropout.train();
  Tensor training = dropout.forward(input);
  EXPECT_EQ(training.shape(), input.shape());
  bool saw_zero = false;
  bool saw_kept = false;
  for (size_t i = 0; i < training.numel(); ++i) {
    saw_zero = saw_zero || training.data()[i] == 0.0f;
    saw_kept = saw_kept || training.data()[i] == 2.0f;
    EXPECT_TRUE(training.data()[i] == 0.0f || training.data()[i] == 2.0f);
  }
  EXPECT_TRUE(saw_zero);
  EXPECT_TRUE(saw_kept);
}

TEST_F(TensorTests, EmbeddingSelectsRowsAndAccumulatesSelectedGradients) {
  Embedding embedding(3, 2);
  embedding.weight() = Tensor({3, 2}, {1, 2, 3, 4, 5, 6});
  embedding.weight().set_requires_grad(true);
  Tensor output = embedding.forward(Tensor({3}, {2, 0, 2}));
  EXPECT_EQ(output.shape(), (std::vector<size_t>{3, 2}));
  expect_values(output, {5, 6, 1, 2, 5, 6});
  Tensor loss = functional::sum(output, {0, 1});
  ASSERT_EQ(loss.shape(), (std::vector<size_t>{}));
  loss.backward();
  expect_values(embedding.weight().grad(), {1, 1, 0, 0, 2, 2});
}

TEST_F(TensorTests, LayerNormPreservesShapeAndNormalizesRows) {
  LayerNorm norm({3});
  Tensor output = norm.forward(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(output.shape(), (std::vector<size_t>{2, 3}));
  for (size_t row = 0; row < 2; ++row) {
    float mean = 0.0f;
    for (size_t column = 0; column < 3; ++column)
      mean += output.data()[row * 3 + column];
    EXPECT_NEAR(mean / 3.0f, 0.0f, 1e-5f);
  }
  EXPECT_NEAR(output.data()[0], -1.22474f, 1e-4f);
  EXPECT_NEAR(output.data()[2], 1.22474f, 1e-4f);
}

TEST_F(TensorTests, AttentionCausalForwardDoesNotReadFutureTokens) {
  MultiHeadAttention attention(4, 2);
  initialize_attention(attention);
  Tensor input({1, 3, 4}, {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6});
  Tensor original = attention.forward(input);
  EXPECT_EQ(original.shape(), (std::vector<size_t>{1, 3, 4}));
  Tensor changed_future = input.clone();
  std::fill_n(changed_future.data() + 4, 8, 100.0f);
  Tensor changed = attention.forward(changed_future);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(original.data()[i], changed.data()[i], 1e-5f);
}

TEST_F(TensorTests, AttentionCrossForwardPreservesQueryShapeAndMask) {
  MultiHeadAttention attention(4, 2);
  initialize_attention(attention);
  Tensor query({1, 2, 4}, {1, 2, 3, 4, 4, 3, 2, 1});
  Tensor context({1, 3, 4}, {1, 0, 2, 0, 0, 1, 0, 2, 2, 1, 0, 1});
  Tensor mask({2, 3}, {0, 0, 1, 0, 0, 0});
  Tensor output = attention.forward(query, context, mask);
  EXPECT_EQ(output.shape(), (std::vector<size_t>{1, 2, 4}));
  for (size_t i = 0; i < output.numel(); ++i)
    EXPECT_TRUE(std::isfinite(output.data()[i]));
}

TEST_F(TensorTests, AdamUpdatesModelAndZeroGradClearsGradients) {
  auto first = std::make_unique<Linear>(2, 3);
  auto second = std::make_unique<Linear>(3, 1);
  first->weight() = Tensor({2, 3}, {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f});
  first->bias() = Tensor({3}, {0, 0, 0});
  second->weight() = Tensor({3, 1}, {0.2f, -0.3f, 0.4f});
  second->bias() = Tensor({1}, {0});
  first->weight().set_requires_grad(true);
  first->bias().set_requires_grad(true);
  second->weight().set_requires_grad(true);
  second->bias().set_requires_grad(true);
  Sequential model(std::move(first), std::move(second));
  optim::Adam optimizer(model.parameters_recursive(), 0.01f);
  Tensor before = model.parameters_recursive()[0]->clone();
  Tensor input({2}, {1.0f, -2.0f});
  Tensor target({1}, {0.75f});
  for (int step = 0; step < 3; ++step) {
    Tensor prediction = model.forward(input);
    Tensor diff = functional::sub(prediction, target);
    Tensor loss = functional::mean(functional::mul(diff, diff));
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
  }
  EXPECT_NE(model.parameters_recursive()[0]->data()[0], before.data()[0]);
  for (Tensor* parameter : model.parameters_recursive())
    if (parameter->has_grad())
      expect_values(parameter->grad(), std::vector<float>(parameter->numel(), 0.0f));
}
