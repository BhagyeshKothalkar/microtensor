#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "microtensor/tensor.hpp"

namespace test_support {

using tensors::Tensor;

inline void expect_tensor_shape(const Tensor& actual,
                                const std::vector<size_t>& expected_shape) {
  EXPECT_EQ(actual.shape(), expected_shape);
}

inline void expect_tensor_values(const Tensor& actual,
                                 const std::vector<float>& expected_values,
                                 float tolerance = 1e-5f) {
  ASSERT_EQ(actual.numel(), expected_values.size());
  ASSERT_NE(actual.data(), nullptr);
  for (size_t i = 0; i < expected_values.size(); ++i) {
    EXPECT_NEAR(actual.data()[i], expected_values[i], tolerance)
        << "flat index " << i;
  }
}

inline void expect_tensor_close(const Tensor& actual, const Tensor& expected,
                                float tolerance = 1e-5f) {
  expect_tensor_shape(actual, expected.shape());
  std::vector<float> values(expected.data(), expected.data() + expected.numel());
  expect_tensor_values(actual, values, tolerance);
}

inline float scalar_value(const Tensor& tensor) {
  EXPECT_TRUE(tensor.shape().empty());
  EXPECT_EQ(tensor.numel(), 1u);
  return tensor.data()[0];
}

inline Tensor finite_difference_gradient(
    const Tensor& input,
    const std::function<Tensor(const Tensor&)>& loss,
    float epsilon = 1e-3f) {
  Tensor gradient(input.shape());
  for (size_t i = 0; i < input.numel(); ++i) {
    Tensor plus = input.clone();
    Tensor minus = input.clone();
    plus.data()[i] += epsilon;
    minus.data()[i] -= epsilon;
    gradient.data()[i] = (scalar_value(loss(plus)) - scalar_value(loss(minus))) /
                         (2.0f * epsilon);
  }
  return gradient;
}

struct FunctionalCase {
  std::string name;
  std::function<Tensor()> run;
  std::vector<size_t> expected_shape;
  std::vector<float> expected_values;
  float tolerance = 1e-5f;
};

}  // namespace test_support
