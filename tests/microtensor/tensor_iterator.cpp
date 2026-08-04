#include "microtensor/tensor_iterator.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test_tensor.h"

using namespace tensors;

/* Iterator Basic Traversal & Hand-Drafted Operations */

TEST_F(TensorTests, TestTensorIteratorBasicTraversal) {
  // Hand-drafted 1D elementwise addition: dst = a + b
  Tensor a({3}, {1.0f, 2.0f, 3.0f});
  Tensor b({3}, {10.0f, 20.0f, 30.0f});
  Tensor dst({3}, {0.0f, 0.0f, 0.0f});

  TensorIterator<float, const float, const float> it(dst, a, b);

  size_t count = 0;
  while (it.has_next()) {
    auto [out, x, y] = it.next();
    out = x + y;
    count++;
  }

  EXPECT_EQ(count, 3u);
  EXPECT_FALSE(it.has_next());
  EXPECT_EQ(dst[0], 11.0f);
  EXPECT_EQ(dst[1], 22.0f);
  EXPECT_EQ(dst[2], 33.0f);

  // Randomized test: Dynamic 1D traversal verifying sum correctness across all
  // elements
  std::uniform_int_distribution<size_t> len_dist(5, 50);
  size_t len = len_dist(gen);
  Tensor rand_a = Tensor::rand({len}, gen);
  Tensor rand_b = Tensor::rand({len}, gen);
  Tensor rand_dst({len});

  TensorIterator<float, const float, const float> rand_it(rand_dst, rand_a,
                                                          rand_b);
  size_t rand_count = 0;
  while (rand_it.has_next()) {
    auto [out, x, y] = rand_it.next();
    out = x + y;
    rand_count++;
  }

  EXPECT_EQ(rand_count, len);
  for (size_t i = 0; i < len; ++i) {
    EXPECT_FLOAT_EQ(rand_dst[i], rand_a[i] + rand_b[i]);
  }
}

/* Multi-Dimensional & Strided Traversal */

TEST_F(TensorTests, TestTensorIteratorMultiDimAndStrides) {
  // Hand-drafted 2D matrix elementwise multiplication (2x3)
  Tensor a({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b({2, 3}, {2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f});
  Tensor dst({2, 3});

  TensorIterator<float, const float, const float> it(dst, a, b);

  size_t count = 0;
  while (it.has_next()) {
    auto [out, x, y] = it.next();
    out = x * y;
    count++;
  }

  EXPECT_EQ(count, 6u);
  EXPECT_EQ((dst[0, 0]), 2.0f);
  EXPECT_EQ((dst[0, 2]), 6.0f);
  EXPECT_EQ((dst[1, 0]), 12.0f);
  EXPECT_EQ((dst[1, 2]), 18.0f);

  // Randomized test: Multi-dimensional row-major traversal on random 3D shapes
  std::vector<size_t> rand_shape = random_shape(3);
  size_t total_elements = compute_size(rand_shape);
  Tensor r_a = Tensor::rand(rand_shape, gen);
  Tensor r_dst(rand_shape);

  TensorIterator<float, const float> rand_it(r_dst, r_a);
  size_t visited = 0;
  while (rand_it.has_next()) {
    auto [out, in] = rand_it.next();
    out = in * 2.0f;
    visited++;
  }

  EXPECT_EQ(visited, total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    EXPECT_FLOAT_EQ(r_dst.data()[i], r_a.data()[i] * 2.0f);
  }
}

/* Broadcasting & Variable Terms */

TEST_F(TensorTests, TestTensorIteratorBroadcastingAndTerms) {
  // Hand-drafted 1D broadcast simulation: vector (3 elements) + scalar stride
  // (0 stride)
  Tensor dst({3}, {0.0f, 0.0f, 0.0f});
  Tensor a({3}, {1.0f, 2.0f, 3.0f});
  // Simulate a broadcasted scalar: shape={3}, stride={0}, sharing single float
  // storage
  auto scalar_storage = std::make_shared<float[]>(1);
  scalar_storage[0] = 5.0f;
  Tensor b_scalar({3}, {0}, scalar_storage, 0);

  TensorIterator<float, const float, const float> it(dst, a, b_scalar);

  while (it.has_next()) {
    auto [out, x, y] = it.next();
    out = x + y;
  }

  EXPECT_EQ(dst[0], 6.0f);
  EXPECT_EQ(dst[1], 7.0f);
  EXPECT_EQ(dst[2], 8.0f);

  // Randomized test: Traversing across 4 participating tensors (1 Dest + 3
  // Sources)
  std::vector<size_t> rand_s = random_shape();
  size_t numel = compute_size(rand_s);
  Tensor r_dst(rand_s);
  Tensor r_src1 = Tensor::rand(rand_s, gen);
  Tensor r_src2 = Tensor::rand(rand_s, gen);
  Tensor r_src3 = Tensor::rand(rand_s, gen);

  TensorIterator<float, const float, const float, const float> multi_it(
      r_dst, r_src1, r_src2, r_src3);

  size_t iterated = 0;
  while (multi_it.has_next()) {
    auto [out, s1, s2, s3] = multi_it.next();
    out = s1 + s2 + s3;
    iterated++;
  }

  EXPECT_EQ(iterated, numel);
  for (size_t i = 0; i < numel; ++i) {
    EXPECT_FLOAT_EQ(r_dst.data()[i],
                    r_src1.data()[i] + r_src2.data()[i] + r_src3.data()[i]);
  }
}

/* Edge Cases & Empty Tensors */

TEST_F(TensorTests, TestTensorIteratorEdgeCases) {
  // Empty tensor traversal edge case
  Tensor empty_dst;
  Tensor empty_src;
  TensorIterator<float, const float> empty_it(empty_dst, empty_src);

  EXPECT_FALSE(empty_it.has_next());

  // Single-element scalar tensor edge case
  Tensor single_dst({1}, {0.0f});
  Tensor single_src({1}, {42.0f});
  TensorIterator<float, const float> single_it(single_dst, single_src);

  EXPECT_TRUE(single_it.has_next());
  auto [out, in] = single_it.next();
  out = in;
  EXPECT_FALSE(single_it.has_next());
  EXPECT_EQ(single_dst[0], 42.0f);

  // Randomized test: Verify immediate termination for any zero-size dimension
  // shape
  std::vector<size_t> zero_shape = {2, 0, 3};
  Tensor z_dst(zero_shape);
  Tensor z_src(zero_shape);
  TensorIterator<float, const float> rand_zero_it(z_dst, z_src);
  EXPECT_FALSE(rand_zero_it.has_next());
}