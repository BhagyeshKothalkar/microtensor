#include "microtensor/cpu_kernels.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test_tensor.h"

using namespace tensors;

/* Addition Kernel Tests */

TEST_F(TensorTests, TestCpuKernelsAdd) {
  // Hand-drafted 2D elementwise addition
  Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
  cpu_kernels::add(a, b);

  EXPECT_FLOAT_EQ((a[0, 0]), 11.0f);
  EXPECT_FLOAT_EQ((a[0, 1]), 22.0f);
  EXPECT_FLOAT_EQ((a[1, 0]), 33.0f);
  EXPECT_FLOAT_EQ((a[1, 1]), 44.0f);

  // Edge case: Empty tensor addition
  Tensor empty_a;
  Tensor empty_b;
  cpu_kernels::add(empty_a, empty_b);
  EXPECT_TRUE(empty_a.empty());

  // Randomized test: Verify dynamic tensor addition
  std::vector<size_t> rand_shape = random_shape();
  Tensor rand_a = Tensor::rand(rand_shape, gen);
  Tensor rand_b = Tensor::rand(rand_shape, gen);
  Tensor expected = rand_a.clone();

  cpu_kernels::add(rand_a, rand_b);
  for (size_t i = 0; i < rand_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(rand_a.data()[i], expected.data()[i] + rand_b.data()[i]);
  }
}

/* Elementwise & Scalar Multiplication Kernel Tests */

TEST_F(TensorTests, TestCpuKernelsMultiply) {
  // Hand-drafted elementwise multiply
  Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b({2, 3}, {2.0f, 0.5f, 3.0f, 0.25f, 2.0f, 0.0f});
  cpu_kernels::elementwise_multiply(a, b);

  EXPECT_FLOAT_EQ((a[0, 0]), 2.0f);
  EXPECT_FLOAT_EQ((a[0, 1]), 1.0f);
  EXPECT_FLOAT_EQ((a[0, 2]), 9.0f);
  EXPECT_FLOAT_EQ((a[1, 0]), 1.0f);
  EXPECT_FLOAT_EQ((a[1, 1]), 10.0f);
  EXPECT_FLOAT_EQ((a[1, 2]), 0.0f);

  // Hand-drafted scalar multiply
  Tensor c({2}, {3.0f, -4.0f});
  cpu_kernels::scalar_multiply(c, 2.5f);
  EXPECT_FLOAT_EQ(c[0], 7.5f);
  EXPECT_FLOAT_EQ(c[1], -10.0f);

  // Randomized test: Verify scalar multiplication on dynamic shapes
  Tensor rand_a = Tensor::rand(random_shape(), gen);
  Tensor orig = rand_a.clone();
  std::uniform_real_distribution<float> scalar_dist(-10.0f, 10.0f);
  float s = scalar_dist(gen);

  cpu_kernels::scalar_multiply(rand_a, s);
  for (size_t i = 0; i < rand_a.numel(); ++i) {
    EXPECT_FLOAT_EQ(rand_a.data()[i], orig.data()[i] * s);
  }
}

/* Naive Matmul Kernel Tests */

TEST_F(TensorTests, TestCpuKernelsNaiveMatmul) {
  // Hand-drafted elementwise lock-step accumulation view simulation
  Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor res({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

  cpu_kernels::naive_matmul(a, b, res);
  EXPECT_FLOAT_EQ((res[0, 0]), 5.0f);
  EXPECT_FLOAT_EQ((res[0, 1]), 12.0f);
  EXPECT_FLOAT_EQ((res[1, 0]), 21.0f);
  EXPECT_FLOAT_EQ((res[1, 1]), 32.0f);

  // Randomized test: Ensure accumulator lock-step behavior stays consistent
  std::vector<size_t> shape = random_shape();
  Tensor r_a = Tensor::rand(shape, gen);
  Tensor r_b = Tensor::rand(shape, gen);
  Tensor r_res = Tensor::zeros(shape);

  cpu_kernels::naive_matmul(r_a, r_b, r_res);
  for (size_t i = 0; i < r_res.numel(); ++i) {
    EXPECT_FLOAT_EQ(r_res.data()[i], r_a.data()[i] * r_b.data()[i]);
  }
}

/* Activations: ReLU & Softmax Kernel Tests */

TEST_F(TensorTests, TestCpuKernelsActivations) {
  // Hand-drafted ReLU
  Tensor x_relu({4}, {-2.0f, 0.0f, 3.5f, -0.01f});
  cpu_kernels::relu(x_relu);
  EXPECT_FLOAT_EQ(x_relu[0], 0.0f);
  EXPECT_FLOAT_EQ(x_relu[1], 0.0f);
  EXPECT_FLOAT_EQ(x_relu[2], 3.5f);
  EXPECT_FLOAT_EQ(x_relu[3], 0.0f);

  // Hand-drafted Softmax check (Sum of output must equal 1.0)
  Tensor x_soft({3}, {1.0f, 2.0f, 3.0f});
  cpu_kernels::softmax(x_soft);

  float expected_sum = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
  EXPECT_FLOAT_EQ(x_soft[0], std::exp(1.0f) / expected_sum);
  EXPECT_FLOAT_EQ(x_soft[1], std::exp(2.0f) / expected_sum);
  EXPECT_FLOAT_EQ(x_soft[2], std::exp(3.0f) / expected_sum);

  // Randomized test: Verify softmax output sums to 1.0 for dynamic 1D tensor
  std::uniform_int_distribution<size_t> len_dist(2, 20);
  size_t len = len_dist(gen);
  Tensor r_soft = Tensor::rand({len}, gen);

  cpu_kernels::softmax(r_soft);
  float sum = 0.0f;
  for (size_t i = 0; i < r_soft.numel(); ++i) {
    EXPECT_GE(r_soft.data()[i], 0.0f);
    EXPECT_LE(r_soft.data()[i], 1.0f);
    sum += r_soft.data()[i];
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}