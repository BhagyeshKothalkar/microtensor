#include "microtensor/broadcasting.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "test_tensor.h"

using namespace tensors;

/* --- Broadcast Compatibility Check Tests --- */

TEST_F(TensorTests, TestAreBroadcastable) {
  // Hand-drafted compatible cases
  Tensor t_2_3_4({2, 3, 4});
  Tensor t_3_4({3, 4});
  Tensor t_5_1_7({5, 1, 7});
  Tensor t_1_8_7({1, 8, 7});

  EXPECT_TRUE(are_broadcastable());
  EXPECT_TRUE(are_broadcastable(t_2_3_4, t_3_4));
  EXPECT_TRUE(are_broadcastable(t_5_1_7, t_1_8_7));

  // Hand-drafted incompatible cases
  Tensor t_2_3({2, 3});
  Tensor t_4_3({4, 3});
  EXPECT_FALSE(are_broadcastable(t_2_3, t_4_3));

  // Randomized test: Verify that appending leading 1s retains compatibility
  std::vector<size_t> rand_s = random_shape();
  Tensor base(rand_s);

  std::vector<size_t> expanded_s = {1, 1};
  expanded_s.insert(expanded_s.end(), rand_s.begin(), rand_s.end());
  Tensor expanded(expanded_s);

  EXPECT_TRUE(are_broadcastable(base, expanded));
}

/* --- Broadcast Shape Computation Tests --- */

TEST_F(TensorTests, TestGetBroadcastShape) {
  // Hand-drafted expected shapes
  Tensor a({2, 3, 4});
  Tensor b({3, 4});
  std::vector<size_t> expected_ab = {2, 3, 4};
  EXPECT_EQ(get_broadcast_shape(a, b), expected_ab);

  Tensor c({5, 1, 7});
  Tensor d({1, 8, 7});
  std::vector<size_t> expected_cd = {5, 8, 7};
  EXPECT_EQ(get_broadcast_shape(c, d), expected_cd);

  // Exception handling for non-broadcastable shapes
  Tensor e({2, 3});
  Tensor f({4, 3});
  EXPECT_THROW(get_broadcast_shape(e, f), std::invalid_argument);

  // Edge case: Empty input parameter pack
  EXPECT_TRUE(get_broadcast_shape().empty());

  // Randomized test: Compute broadcast shape between base tensor and random
  // ones-padded tensor
  std::vector<size_t> rand_s = random_shape();
  std::vector<size_t> padded_s = rand_s;
  for (size_t& dim : padded_s) {
    if (gen() % 2 == 0) dim = 1;  // Randomly set dimensions to 1
  }

  Tensor rand_t1(rand_s);
  Tensor rand_t2(padded_s);
  EXPECT_EQ(get_broadcast_shape(rand_t1, rand_t2), rand_s);
}

/* --- Tensor Broadcast Views & Strides --- */

TEST_F(TensorTests, TestBroadcastToShape) {
  // Hand-drafted stride checks: expanding {3, 1} to {3, 4}
  Tensor in({3, 1}, {1.0f, 2.0f, 3.0f});
  std::vector<size_t> target_shape = {3, 4};
  Tensor view = broadcast_to_shape(in, target_shape);

  EXPECT_EQ(view.shape(), target_shape);
  // Strides for {3, 1} are {1, 1}. Broadcasted axis along dimension size 1 gets
  // stride 0.
  std::vector<size_t> expected_strides = {1, 0};
  EXPECT_EQ(view.stride(), expected_strides);
  EXPECT_EQ(view.storage().get(), in.storage().get());

  // Single-element scalar to multidimensional shape edge case
  Tensor scalar({1}, {42.0f});
  std::vector<size_t> multi_dim = {2, 3, 4};
  Tensor scalar_view = broadcast_to_shape(scalar, multi_dim);

  EXPECT_EQ(scalar_view.shape(), multi_dim);
  EXPECT_EQ(scalar_view.stride(), std::vector<size_t>({0, 0, 0}));

  // Randomized test: Verify broadcasted view preserves data across random
  // shapes
  std::vector<size_t> base_shape = {1, random_shape(1)[0]};
  Tensor rand_in = Tensor::rand(base_shape, gen);

  std::vector<size_t> rand_target = {shape_distrib(gen), base_shape[1]};
  Tensor rand_view = broadcast_to_shape(rand_in, rand_target);

  EXPECT_EQ(rand_view.shape(), rand_target);
  EXPECT_EQ(rand_view.stride()[0], 0u);
  EXPECT_EQ(rand_view.data(), rand_in.data());
}

/* --- Multi-Tensor Broadcast Functions --- */

TEST_F(TensorTests, TestBroadcastTensors) {
  // Hand-drafted explicit shape broadcasting
  Tensor a({3, 1}, {1.0f, 2.0f, 3.0f});
  Tensor b({1, 4}, {10.0f, 20.0f, 30.0f, 40.0f});
  std::vector<size_t> target = {3, 4};

  auto [v_a, v_b] = broadcast_tensors(target, a, b);
  EXPECT_EQ(v_a.shape(), target);
  EXPECT_EQ(v_b.shape(), target);
  EXPECT_EQ(v_a.stride(), std::vector<size_t>({1, 0}));
  EXPECT_EQ(v_b.stride(), std::vector<size_t>({0, 1}));

  // Automatic broadcast helper: broadcast_tensors(a, b)
  Tensor c({2, 1, 3});
  Tensor d({1, 4, 3});
  auto [auto_shape, views] = broadcast_tensors(c, d);
  auto [v_c, v_d] = views;

  std::vector<size_t> expected_auto_shape = {2, 4, 3};
  EXPECT_EQ(auto_shape, expected_auto_shape);
  EXPECT_EQ(v_c.shape(), expected_auto_shape);
  EXPECT_EQ(v_d.shape(), expected_auto_shape);

  // Randomized test: Verify multi-tensor auto broadcast across randomized terms
  std::vector<size_t> rand_shape1 = random_shape(2);
  std::vector<size_t> rand_shape2 = {rand_shape1[0], 1};

  Tensor r1(rand_shape1);
  Tensor r2(rand_shape2);

  auto [res_shape, res_views] = broadcast_tensors(r1, r2);
  auto [view1, view2] = res_views;

  EXPECT_EQ(res_shape, rand_shape1);
  EXPECT_EQ(view1.shape(), rand_shape1);
  EXPECT_EQ(view2.shape(), rand_shape1);
  EXPECT_EQ(view2.stride()[1], 0u);
}