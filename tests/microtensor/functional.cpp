#include "microtensor/functional.hpp"

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "opinfo_test.h"
#include "test_tensor.h"

using namespace tensors;
using test_support::FunctionalCase;

namespace {

void expect_functional_case(const FunctionalCase& test_case) {
  SCOPED_TRACE(test_case.name);
  Tensor actual = test_case.run();
  test_support::expect_tensor_shape(actual, test_case.expected_shape);
  test_support::expect_tensor_values(actual, test_case.expected_values,
                                     test_case.tolerance);
}

}  // namespace

TEST(FunctionalOpInfo, Reductions) {
  const std::vector<FunctionalCase> cases = {
      {"sum_last_dimension",
       [] { return functional::sum(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}), {-1}); },
       {2}, {6, 15}},
      {"sum_negative_dimension_keepdims",
       [] {
         return functional::sum(Tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
                                {-2}, true);
       },
       {2, 1, 2}, {4, 6, 12, 14}},
      {"sum_all_to_scalar",
       [] { return functional::sum(Tensor({2, 2}, {1, 2, 3, 4}), {0, 1}); },
       {}, {10}},
      {"max_last_dimension",
       [] { return functional::max(Tensor({2, 3}, {1, 7, 3, 9, 5, 6}), {-1}); },
       {2}, {7, 9}},
      {"mean_last_dimension",
       [] { return functional::mean(Tensor({2, 2}, {1, 2, 5, 8}), {-1}); },
       {2}, {1.5f, 6.5f}},
  };
  for (const auto& test_case : cases) expect_functional_case(test_case);
}

TEST(FunctionalOpInfo, ActivationsAndNormalization) {
  const std::vector<FunctionalCase> cases = {
      {"softmax_negative_last_dimension",
       [] { return functional::softmax(Tensor({2, 3}, {0, 1, 2, 2, 1, 0}), -1); },
       {2, 3}, {0.09003057f, 0.24472847f, 0.66524096f,
                0.66524096f, 0.24472847f, 0.09003057f},
       1e-5f},
      {"logsoftmax_negative_last_dimension",
       [] {
         return functional::logsoftmax(Tensor({2, 3}, {0, 1, 2, 2, 1, 0}), -1);
       },
       {2, 3}, {-2.4076059f, -1.4076059f, -0.40760595f,
                -0.40760595f, -1.4076059f, -2.4076059f},
       1e-5f},
      {"rmsnorm_last_dimension",
       [] {
         return functional::rmsnorm(Tensor({2, 2}, {3, 4, 0, 5}), {-1}, 0.0f);
       },
       {2, 2}, {0.84852815f, 1.1313709f, 0.0f, 1.4142135f}, 1e-5f},
  };
  for (const auto& test_case : cases) expect_functional_case(test_case);
}

TEST(FunctionalOpInfo, TensorSelectionAndComposition) {
  const Tensor mask({2, 3}, {0, 1, 0, 1, 0, 1});
  const Tensor indices({2}, {2, 0});
  const std::vector<FunctionalCase> cases = {
      {"masked_fill",
       [mask] {
         return functional::masked_fill(Tensor({2, 3}, {1, 2, 3, 4, 5, 6}),
                                        mask, -10.0f);
       },
       {2, 3}, {1, -10, 3, -10, 5, -10}},
      {"index_select_columns",
       [indices] {
         return functional::index_select(
             Tensor({2, 3}, {1, 2, 3, 4, 5, 6}), 1, indices);
       },
       {2, 2}, {3, 1, 6, 4}},
      {"cat_rows",
       [] {
         return functional::cat({Tensor({1, 2}, {1, 2}),
                                  Tensor({2, 2}, {3, 4, 5, 6})}, 0);
       },
       {3, 2}, {1, 2, 3, 4, 5, 6}},
      {"cat_columns",
       [] {
         return functional::cat({Tensor({2, 1}, {1, 2}),
                                  Tensor({2, 2}, {3, 4, 5, 6})}, -1);
       },
       {2, 3}, {1, 3, 4, 2, 5, 6}},
  };
  for (const auto& test_case : cases) expect_functional_case(test_case);
}

TEST(FunctionalOpInfo, MatmulAndCrossEntropy) {
  const std::vector<FunctionalCase> cases = {
      {"batched_matmul",
       [] {
         return functional::matmul(
             Tensor({2, 2, 2}, {1, 2, 3, 4, 2, 0, 1, 2}),
             Tensor({2, 2, 3}, {1, 0, 0, 0, 1, 0,
                                2, 1, 0, 0, 1, 1}));
       },
       {2, 2, 3}, {1, 2, 0, 3, 4, 0, 4, 2, 0, 2, 3, 2}, 1e-5f},
      {"cross_entropy_scalar",
       [] {
         return functional::cross_entropy(
             Tensor({2, 3}, {1, 2, 3, 2, 1, 0}), Tensor({2}, {2, 0}));
       },
       {}, {0.40760595f}, 1e-5f},
  };
  for (const auto& test_case : cases) expect_functional_case(test_case);
}

TEST_F(TensorTests, TestFunctionalInplaceArithmetic) {
  Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor b({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  a = functional::add(a, b);
  test_support::expect_tensor_values(a, {2, 3, 4, 5});
  a = functional::mul(a, b);
  a = functional::mul(a, 3.0f);
  test_support::expect_tensor_values(a, {6, 9, 12, 15});
}

TEST(FunctionalContracts, InvalidReductionAndMatmulInputs) {
  EXPECT_THROW(functional::sum(Tensor({2, 2}, {1, 2, 3, 4}), {2}),
               std::out_of_range);
  EXPECT_THROW(functional::matmul(Tensor({2, 3}), Tensor({2, 2})),
               std::invalid_argument);
}

TEST(FunctionalActivations, ReluAndInplaceSoftmax) {
  Tensor relu_input({3}, {-5.0f, 0.0f, 5.0f});
  relu_input = functional::relu(relu_input);
  test_support::expect_tensor_values(relu_input, {0, 0, 5});
  Tensor softmax_input({2}, {0.0f, 0.0f});
  softmax_input = functional::softmax(softmax_input, -1);
  test_support::expect_tensor_values(softmax_input, {0.5f, 0.5f});
}
