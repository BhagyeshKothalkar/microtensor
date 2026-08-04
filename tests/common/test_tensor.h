#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "microtensor/tensor.hpp"

class TensorTests : public ::testing::Test {
 protected:
  void SetUp() override { gen.seed(rd()); }
  void TearDown() override {}

  std::random_device rd;
  std::mt19937 gen;

  std::uniform_int_distribution<size_t> shape_distrib{1, 4};
  std::uniform_int_distribution<size_t> ndims_distrib{1, 4};

  std::vector<size_t> random_shape(size_t ndims) {
    std::vector<size_t> ret(ndims);
    for (auto& dim : ret) {
      dim = shape_distrib(gen);
    }
    return ret;
  }

  std::vector<size_t> random_shape() {
    return random_shape(ndims_distrib(gen));
  }

  tensors::Tensor random_tensor() {
    return tensors::Tensor::rand(random_shape(), gen);
  }
};