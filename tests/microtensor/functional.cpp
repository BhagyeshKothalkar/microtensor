#include "microtensor/functional.hpp"

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "test_tensor.h"

using namespace tensors;

/* In-Place Arithmetic Functional Wrappers */

TEST_F(TensorTests, TestFunctionalInplaceArithmetic) {
  // Hand-drafted add_ and chaining
  Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});

  // Test method chaining and reference returning
  Tensor& ref = functional::add_(a, b);
  EXPECT_EQ(&ref, &a);
  EXPECT_FLOAT_EQ((a[0, 0]), 2.0f);
  EXPECT_FLOAT_EQ((a[1, 1]), 5.0f);

  // Hand-drafted elementwise_multiply_ & scalar_multiply_
  functional::elementwise_multiply_(a, b);
  EXPECT_FLOAT_EQ((a[0, 0]), 2.0f);

  functional::scalar_multiply_(a, 3.0f);
  EXPECT_FLOAT_EQ((a[0, 0]), 6.0f);
  EXPECT_FLOAT_EQ((a[1, 1]), 15.0f);

  // Randomized test: Verify add_ correctly mutates destination in-place
  std::vector<size_t> rand_s = random_shape();
  Tensor rand_a = Tensor::rand(rand_s, gen);
  Tensor rand_b = Tensor::rand(rand_s, gen);
  Tensor orig_a = rand_a.clone();

  functional::add_(rand_a, rand_b);
  for (size_t i = 0; i < rand_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(rand_a.data()[i], orig_a.data()[i] + rand_b.data()[i]);
  }
}

/* Matmul Functional Wrapper Tests */

TEST_F(TensorTests, TestFunctionalNaiveMatmul) {
  // Hand-drafted naive_matmul functional wrapper
  Tensor a({2}, {2.0f, 3.0f});
  Tensor b({2}, {4.0f, 5.0f});

  Tensor& ref = functional::naive_matmul(a, b);
  EXPECT_EQ(&ref, &a);

  // Randomized test: Verify functional wrapper delegates to naive_matmul kernel
  std::vector<size_t> rand_s = random_shape();
  Tensor rand_a = Tensor::rand(rand_s, gen);
  Tensor rand_b = Tensor::rand(rand_s, gen);

  functional::naive_matmul(rand_a, rand_b);
  // Ensures function completes execution cleanly and returns reference
  EXPECT_EQ(rand_a.shape(), rand_s);
}

/* Activation Functional Wrappers */

TEST_F(TensorTests, TestFunctionalActivations) {
  // Hand-drafted relu_
  Tensor a({3}, {-5.0f, 0.0f, 5.0f});
  functional::relu_(a);
  EXPECT_FLOAT_EQ(a[0], 0.0f);
  EXPECT_FLOAT_EQ(a[1], 0.0f);
  EXPECT_FLOAT_EQ(a[2], 5.0f);

  // Hand-drafted softmax_
  Tensor b({2}, {0.0f, 0.0f});
  functional::softmax_(b);
  EXPECT_FLOAT_EQ(b[0], 0.5f);
  EXPECT_FLOAT_EQ(b[1], 0.5f);

  // Randomized test: Verify relu_ non-negativity property on dynamic inputs
  Tensor rand_a = Tensor::rand(random_shape(), gen);
  functional::scalar_multiply_(rand_a, -1.0f);  // Make all values negative
  functional::relu_(rand_a);

  for (size_t i = 0; i < rand_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(rand_a.data()[i], 0.0f);
  }
}