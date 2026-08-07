#include "microtensor/autograd.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"

using namespace tensors;

TEST(AutogradTests, TestBasicAddSubMulDiv) {
  Tensor x({1}, {3.0f});
  Tensor y({1}, {4.0f});

  x.set_requires_grad(true);
  y.set_requires_grad(true);

  // z = x * y + x - y
  Tensor z = (x * y) + x - y;
  EXPECT_TRUE(z.requires_grad());
  EXPECT_NE(z.grad_fn(), nullptr);

  z.backward();

  // dz/dx = y + 1 = 4 + 1 = 5
  // dz/dy = x - 1 = 3 - 1 = 2
  EXPECT_FLOAT_EQ(x.grad()[0], 5.0f);
  EXPECT_FLOAT_EQ(y.grad()[0], 2.0f);
}

TEST(AutogradTests, TestBranchConvergenceAccumulation) {
  // f(x) = x^2 + x^3 = x * x + x * x * x
  Tensor x({1}, {2.0f});
  x.set_requires_grad(true);

  Tensor x2 = x * x;
  Tensor x3 = x2 * x;
  Tensor f = x2 + x3;

  f.backward();

  // f'(x) = 2x + 3x^2 evaluated at x=2 -> 2(2) + 3(4) = 4 + 12 = 16
  EXPECT_FLOAT_EQ(x.grad()[0], 16.0f);
}

TEST(AutogradTests, TestNoGradGuard) {
  Tensor x({1}, {3.0f});
  Tensor y({1}, {4.0f});
  x.set_requires_grad(true);
  y.set_requires_grad(true);

  Tensor z_grad;
  Tensor z_nograd;

  {
    NoGradGuard guard;
    z_nograd = x * y;
    EXPECT_FALSE(z_nograd.requires_grad());
    EXPECT_EQ(z_nograd.grad_fn(), nullptr);
  }

  // Back outside guard, autograd is re-enabled
  z_grad = x * y;
  EXPECT_TRUE(z_grad.requires_grad());
  EXPECT_NE(z_grad.grad_fn(), nullptr);
}

TEST(AutogradTests, TestNonDifferentiableInputsNoNodeCreated) {
  Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

  // Neither a nor b require grad
  Tensor c = a + b;
  EXPECT_FALSE(c.requires_grad());
  EXPECT_EQ(c.grad_fn(), nullptr);
}

TEST(AutogradTests, TestSinCosRelu) {
  Tensor x({3}, {0.0f, 0.5f * 3.14159265f, -1.0f});
  x.set_requires_grad(true);

  // y = sin(x) + cos(x) + relu(x)
  Tensor y = functional::sum(functional::sin(x) + functional::cos(x) +
                             functional::relu(x));
  y.backward();

  // dy/dx_i = cos(x_i) - sin(x_i) + (x_i > 0 ? 1 : 0)
  EXPECT_NEAR(x.grad()[0], 1.0f, 1e-4f);
  EXPECT_NEAR(x.grad()[1], 0.0f, 1e-4f);
  EXPECT_NEAR(x.grad()[2], std::cos(-1.0f) - std::sin(-1.0f), 1e-4f);
}

TEST(AutogradTests, TestBroadcastingUnbroadcastingGradients) {
  // A is (3, 1), B is (1, 4)
  Tensor A({3, 1}, {1.0f, 2.0f, 3.0f});
  Tensor B({1, 4}, {10.0f, 20.0f, 30.0f, 40.0f});
  A.set_requires_grad(true);
  B.set_requires_grad(true);

  // C = A + B (shape 3, 4)
  Tensor C = A + B;
  Tensor L = functional::sum(C);

  L.backward();

  // dL/dC is all ones (3, 4)
  // dL/dA sums over axis 1 (4 columns) -> shape (3, 1), each element = 4.0
  // dL/dB sums over axis 0 (3 rows) -> shape (1, 4), each element = 3.0
  EXPECT_EQ(A.grad().shape(), (std::vector<size_t>{3, 1}));
  EXPECT_EQ(B.grad().shape(), (std::vector<size_t>{1, 4}));

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ((A.grad()[i, 0]), 4.0f);
  }
  for (size_t j = 0; j < 4; ++j) {
    EXPECT_FLOAT_EQ((B.grad()[0, j]), 3.0f);
  }
}

TEST(AutogradTests, TestMatmulBackprop) {
  // A is (2, 3), B is (3, 2)
  Tensor A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor B({3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  A.set_requires_grad(true);
  B.set_requires_grad(true);

  Tensor C = functional::matmul(A, B);  // (2, 2)
  Tensor L = functional::sum(C);

  L.backward();

  // grad_C is ones (2, 2)
  // grad_A = grad_C * B^T
  // B^T is (2, 3): {{7, 9, 11}, {8, 10, 12}}
  // grad_A = {{1, 1}, {1, 1}} * {{7, 9, 11}, {8, 10, 12}} = {{15, 19, 23}, {15,
  // 19, 23}}
  EXPECT_EQ(A.grad().shape(), (std::vector<size_t>{2, 3}));
  EXPECT_FLOAT_EQ((A.grad()[0, 0]), 15.0f);
  EXPECT_FLOAT_EQ((A.grad()[0, 1]), 19.0f);
  EXPECT_FLOAT_EQ((A.grad()[0, 2]), 23.0f);
  EXPECT_FLOAT_EQ((A.grad()[1, 0]), 15.0f);
  EXPECT_FLOAT_EQ((A.grad()[1, 1]), 19.0f);
  EXPECT_FLOAT_EQ((A.grad()[1, 2]), 23.0f);

  // grad_B = A^T * grad_C
  // A^T is (3, 2): {{1, 4}, {2, 5}, {3, 6}}
  // grad_B = {{1, 4}, {2, 5}, {3, 6}} * {{1, 1}, {1, 1}} = {{5, 5}, {7, 7}, {9,
  // 9}}
  EXPECT_EQ(B.grad().shape(), (std::vector<size_t>{3, 2}));
  EXPECT_FLOAT_EQ((B.grad()[0, 0]), 5.0f);
  EXPECT_FLOAT_EQ((B.grad()[0, 1]), 5.0f);
  EXPECT_FLOAT_EQ((B.grad()[1, 0]), 7.0f);
  EXPECT_FLOAT_EQ((B.grad()[1, 1]), 7.0f);
  EXPECT_FLOAT_EQ((B.grad()[2, 0]), 9.0f);
  EXPECT_FLOAT_EQ((B.grad()[2, 1]), 9.0f);
}

TEST(AutogradTests, TestReshapeTransposeBroadcastingViewOps) {
  Tensor x({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  x.set_requires_grad(true);

  Tensor x_reshaped = functional::reshape(x, {2, 3});
  Tensor x_transposed =
      functional::transpose(x_reshaped, 0, 1);  // shape (3, 2)
  Tensor loss = functional::sum(x_transposed);

  loss.backward();

  EXPECT_EQ(x.grad().shape(), (std::vector<size_t>{6}));
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(x.grad()[i], 1.0f);
  }
}

TEST(AutogradTests, TestFiniteDifferenceNumericalCheck) {
  // Check autograd derivative vs central finite difference for f(x) = sin(x) *
  // x + (x * x)
  auto f_eval = [](float val) -> float {
    return std::sin(val) * val + (val * val);
  };

  float x_val = 1.5f;
  float eps = 1e-3f;
  float numerical_grad =
      (f_eval(x_val + eps) - f_eval(x_val - eps)) / (2.0f * eps);

  Tensor x({1}, {x_val});
  x.set_requires_grad(true);

  Tensor y = functional::sin(x) * x + (x * x);
  y.backward();

  float analytical_grad = x.grad()[0];
  EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3f);
}
