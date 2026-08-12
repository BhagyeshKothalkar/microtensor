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
  // =========================================================================
  // 1. HANDWRITTEN EDGE CASES
  // =========================================================================

  // --- Case A: Standard 2D Matrix Multiplication (2x3 * 3x2 -> 2x2) ---
  {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // B = [[7,  8],
    //      [9,  10],
    //      [11, 12]]
    Tensor b({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    // C = A * B = [[58,  64],
    //              [139, 154]]
    Tensor res = functional::naive_matmul(a, b);

    EXPECT_EQ(res.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_EQ(res.numel(), 4);

    if constexpr (requires { res.data(); }) {
      EXPECT_FLOAT_EQ(res.data()[0], 58.0f);
      EXPECT_FLOAT_EQ(res.data()[1], 64.0f);
      EXPECT_FLOAT_EQ(res.data()[2], 139.0f);
      EXPECT_FLOAT_EQ(res.data()[3], 154.0f);
    }
  }

  // --- Case B: Identity Matrix Multiplication (A * I = A) ---
  {
    Tensor a({2, 2}, {3.0f, -1.0f, 2.0f, 4.0f});
    Tensor eye({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    Tensor res = functional::naive_matmul(a, eye);

    EXPECT_EQ(res.shape(), a.shape());
    if constexpr (requires { res.data(); }) {
      for (size_t i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(res.data()[i], a.data()[i]);
      }
    }
  }

  // --- Case C: Single Element / 1x1 Matrix Multiplication ---
  {
    Tensor a({1, 1}, {3.5f});
    Tensor b({1, 1}, {2.0f});

    Tensor res = functional::naive_matmul(a, b);

    EXPECT_EQ(res.shape(), (std::vector<size_t>{1, 1}));
    if constexpr (requires { res.data(); }) {
      EXPECT_FLOAT_EQ(res.data()[0], 7.0f);
    }
  }

  // --- Case D: Inner Dimension Mismatch Exception ---
  {
    Tensor a({2, 3});
    Tensor b({2, 2});  // Inner dims: 3 != 2

    EXPECT_THROW(functional::naive_matmul(a, b), std::runtime_error);
  }

  // --- Case E: Non-2D Tensor Rank Exception ---
  {
    Tensor a_1d({3});
    Tensor b_2d({3, 2});
    EXPECT_THROW(functional::naive_matmul(a_1d, b_2d), std::runtime_error);

    Tensor a_3d({2, 3, 4});
    Tensor b_3d({2, 4, 3});
    EXPECT_THROW(functional::naive_matmul(a_3d, b_3d), std::runtime_error);
  }

  // =========================================================================
  // 2. RANDOMIZED TESTS USING FIXTURE (RANDOM 2D SHAPES)
  // =========================================================================

  const size_t num_random_iterations = 20;

  for (size_t iter = 0; iter < num_random_iterations; ++iter) {
    // Generate valid 2D matrix dimensions: A(M, K) x B(K, N) -> C(M, N)
    std::uniform_int_distribution<size_t> dim_distrib(1, 8);
    size_t M = dim_distrib(gen);
    size_t K = dim_distrib(gen);
    size_t N = dim_distrib(gen);

    Tensor a = Tensor::rand({M, K}, gen);
    Tensor b = Tensor::rand({K, N}, gen);

    Tensor c = functional::naive_matmul(a, b);

    // Verify output properties
    EXPECT_EQ(c.shape(), (std::vector<size_t>{M, N}));
    EXPECT_EQ(c.numel(), M * N);

    // Verify element-wise mathematical correctness against CPU reference loop
    if constexpr (requires {
                    a.data();
                    b.data();
                    c.data();
                  }) {
      for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
          float expected_sum = 0.0f;
          for (size_t k = 0; k < K; ++k) {
            expected_sum += a.data()[m * K + k] * b.data()[k * N + n];
          }
          size_t c_idx = m * N + n;
          EXPECT_NEAR(c.data()[c_idx], expected_sum, 1e-4f);
        }
      }
    }
  }
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