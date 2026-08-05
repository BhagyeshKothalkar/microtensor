#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "microtensor/tensor.hpp"
#include "test_tensor.h"

using namespace tensors;

/* Helper Utility Functions */

TEST_F(TensorTests, TestComputeStridesAndSize) {
  // Empty shape edge case
  EXPECT_TRUE(compute_strides({}).empty());
  EXPECT_EQ(compute_size({}), 0u);

  // 1D shape
  EXPECT_EQ(compute_strides({5}), std::vector<size_t>({1}));
  EXPECT_EQ(compute_size({5}), 5u);

  // 3D shape
  std::vector<size_t> shape = {2, 3, 4};
  std::vector<size_t> expected_strides = {12, 4, 1};
  EXPECT_EQ(compute_strides(shape), expected_strides);
  EXPECT_EQ(compute_size(shape), 24u);

  // Dimension containing zero
  std::vector<size_t> zero_shape = {2, 0, 4};
  EXPECT_EQ(compute_size(zero_shape), 0u);

  // Randomized test: Verify row-major strides identity for arbitrary shapes
  std::vector<size_t> rand_shape = random_shape();
  std::vector<size_t> strides = compute_strides(rand_shape);
  size_t expected_size = compute_size(rand_shape);
  if (!rand_shape.empty()) {
    EXPECT_EQ(strides.back(), 1u);
    size_t accumulated = 1;
    for (size_t i = rand_shape.size(); i > 0; --i) {
      EXPECT_EQ(strides[i - 1], accumulated);
      accumulated *= rand_shape[i - 1];
    }
    EXPECT_EQ(accumulated, expected_size);
  }
}

/* Construct and Initialization Tests */
TEST_F(TensorTests, TestTensorCreation) {
  // Default constructor
  Tensor a1;
  EXPECT_TRUE(a1.empty());
  EXPECT_EQ(a1.numel(), 0u);
  EXPECT_EQ(a1.ndim(), 1u);
  EXPECT_EQ(a1.data(), nullptr);

  // Shape constructor
  Tensor a2(random_shape());
  EXPECT_FALSE(a2.empty());
  EXPECT_NE(a2.data(), nullptr);

  // Custom storage view constructor
  std::vector<size_t> s = random_shape();
  size_t s_size = compute_size(s);
  size_t offset = std::uniform_int_distribution<size_t>(0, s_size - 1)(gen);
  auto storage = std::make_shared_for_overwrite<float[]>(s_size);

  Tensor a3(s, compute_strides(s), storage, offset);
  EXPECT_EQ(a3.data(), a3.storage().get() + offset);
  EXPECT_EQ(a3.offset(), offset);

  // Initializer list constructor (Hand-drafted)
  Tensor a4({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  EXPECT_EQ(a4.numel(), 6u);
  EXPECT_EQ((a4[0, 0]), 1.0f);
  EXPECT_EQ((a4[0, 2]), 3.0f);
  EXPECT_EQ((a4[1, 0]), 4.0f);
  EXPECT_EQ((a4[1, 2]), 6.0f);

  // Randomized test: Verify storage sharing invariants across arbitrary tensor
  // shapes
  std::vector<size_t> rand_s = random_shape();
  size_t rand_size = compute_size(rand_s);
  size_t rand_offset =
      std::uniform_int_distribution<size_t>(0, rand_size - 1)(gen);
  auto rand_storage = std::make_shared_for_overwrite<float[]>(rand_size);
  Tensor rand_view(rand_s, compute_strides(rand_s), rand_storage, rand_offset);
  EXPECT_EQ(rand_view.numel(), rand_size);
  EXPECT_EQ(rand_view.ndim(), rand_s.size());
  EXPECT_EQ(rand_view.data(), rand_storage.get() + rand_offset);
}

/* Factory Methods */

TEST_F(TensorTests, TestFactoryZerosOnesFull) {
  std::vector<size_t> shape = {2, 3};
  size_t total = compute_size(shape);

  // zeros
  Tensor z = Tensor::zeros(shape);
  EXPECT_EQ(z.numel(), total);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((z[i, j]), 0.0f);
    }
  }

  // ones
  Tensor o = Tensor::ones(shape);
  EXPECT_EQ(o.numel(), total);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((o[i, j]), 1.0f);
    }
  }

  // full
  Tensor f = Tensor::full(shape, 3.14f);
  EXPECT_EQ(f.numel(), total);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ((f[i, j]), 3.14f);
    }
  }

  // Randomized test: Verify full() factory on randomized shapes and scalar
  // values
  std::vector<size_t> rand_shape_full = random_shape();
  std::uniform_real_distribution<float> val_dist(-100.0f, 100.0f);
  float fill_val = val_dist(gen);
  Tensor rand_f = Tensor::full(rand_shape_full, fill_val);
  EXPECT_EQ(rand_f.numel(), compute_size(rand_shape_full));
  for (size_t i = 0; i < rand_f.numel(); ++i) {
    EXPECT_EQ(rand_f.data()[i], fill_val);
  }
}

TEST_F(TensorTests, TestLinspace) {
  // Hand-drafted 1D linspace
  Tensor l = Tensor::linspace(0.0f, 1.0f, 5);
  EXPECT_EQ(l.ndim(), 1u);
  EXPECT_EQ(l.numel(), 5u);
  EXPECT_NEAR(l[0], 0.0f, 1e-4);
  EXPECT_NEAR(l[1], 0.25f, 1e-4);
  EXPECT_NEAR(l[2], 0.50f, 1e-4);
  EXPECT_NEAR(l[3], 0.75f, 1e-4);
  EXPECT_NEAR(l[4], 1.0f, 1e-4);

  // Single element linspace edge case
  Tensor l_single = Tensor::linspace(5.0f, 10.0f, 1);
  EXPECT_EQ(l_single.numel(), 1u);

  // Randomized test: Verify endpoint accuracy and interval step uniformity
  std::uniform_real_distribution<float> range_dist(-50.0f, 50.0f);
  std::uniform_int_distribution<size_t> count_dist(2, 50);
  float start = range_dist(gen);
  float end = range_dist(gen);
  size_t num = count_dist(gen);
  Tensor rand_l = Tensor::linspace(start, end, num);
  EXPECT_NEAR(rand_l[0], start, 1e-4);
  EXPECT_NEAR(rand_l[num - 1], end, 1e-4);
  float step = (end - start) / static_cast<float>(num - 1);
  for (size_t i = 0; i < num; ++i) {
    EXPECT_NEAR(rand_l[i], start + static_cast<float>(i) * step, 1e-4f);
  }
}

TEST_F(TensorTests, TestRand) {
  auto shape = random_shape();
  Tensor r = Tensor::rand(shape, gen);
  EXPECT_EQ(r.shape(), shape);

  // Verify all elements are within [0.0, 1.0)
  for (size_t i = 0; i < r.numel(); ++i) {
    EXPECT_GE(r.data()[i], 0.0f);
    EXPECT_LT(r.data()[i], 1.0f);
  }

  // Randomized test: Verify statistical range consistency on dynamic shapes
  std::vector<size_t> dynamic_shape = random_shape();
  Tensor rand_dynamic = Tensor::rand(dynamic_shape, gen);
  EXPECT_EQ(rand_dynamic.shape(), dynamic_shape);
  for (size_t i = 0; i < rand_dynamic.numel(); ++i) {
    EXPECT_GE(rand_dynamic.data()[i], 0.0f);
    EXPECT_LT(rand_dynamic.data()[i], 1.0f);
  }
}

/* Element Access and Indexing */

TEST_F(TensorTests, TestIndexingAndMutations) {
  // 1D Indexing
  Tensor t1({4}, {10.0f, 20.0f, 30.0f, 40.0f});
  EXPECT_EQ(t1[0], 10.0f);
  EXPECT_EQ(t1[3], 40.0f);
  t1[2] = 99.0f;
  EXPECT_EQ(t1[2], 99.0f);

  // 3D Indexing (Hand-drafted calculation)
  Tensor t3({2, 2, 2}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  EXPECT_EQ((t3[0, 0, 0]), 0.0f);
  EXPECT_EQ((t3[0, 1, 1]), 3.0f);
  EXPECT_EQ((t3[1, 0, 0]), 4.0f);
  EXPECT_EQ((t3[1, 1, 1]), 7.0f);

  // Const reference access check
  const Tensor& t3_const = t3;
  EXPECT_EQ((t3_const[1, 1, 0]), 6.0f);

  // Randomized test: Flat index mapping vs variadic subscript operator on a 2D
  // tensor
  std::uniform_int_distribution<size_t> dim_dist(2, 6);
  size_t rows = dim_dist(gen);
  size_t cols = dim_dist(gen);
  Tensor rand_2d({rows, cols});
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      float val = static_cast<float>(r * cols + c);
      rand_2d[r, c] = val;
      EXPECT_EQ(rand_2d.data()[r * cols + c], val);
    }
  }
}

/* Cloning and Storage Sharing */

TEST_F(TensorTests, TestClone) {
  Tensor original = Tensor::rand({2, 3}, gen);
  Tensor cloned = original.clone();

  // Distinct storage memory allocation
  EXPECT_NE(original.storage().get(), cloned.storage().get());
  EXPECT_EQ(original.shape(), cloned.shape());
  EXPECT_EQ(original.stride(), cloned.stride());

  // Deep copy mutation independence
  float original_val = original[0, 0];
  cloned[0, 0] = original_val + 10.0f;
  EXPECT_EQ((original[0, 0]), original_val);
  EXPECT_EQ((cloned[0, 0]), original_val + 10.0f);

  // Randomized test: Verify clone independence on dynamic shapes and values
  Tensor rand_orig = Tensor::rand(random_shape(), gen);
  Tensor rand_cloned = rand_orig.clone();
  EXPECT_NE(rand_orig.storage().get(), rand_cloned.storage().get());
  for (size_t i = 0; i < rand_orig.numel(); ++i) {
    EXPECT_EQ(rand_orig.data()[i], rand_cloned.data()[i]);
  }
  if (!rand_orig.empty()) {
    rand_cloned.data()[0] += 5.0f;
    EXPECT_NE(rand_orig.data()[0], rand_cloned.data()[0]);
  }
}

TEST_F(TensorTests, TestSharedStorageView) {
  Tensor original({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  // Create a view sharing original storage with an offset of 2
  Tensor view({2}, {1}, original.storage(), 2);

  EXPECT_EQ(view.data(), original.data() + 2);
  EXPECT_EQ(view[0], 3.0f);
  EXPECT_EQ(view[1], 4.0f);

  // Modifying view modifies underlying shared storage
  view[0] = 300.0f;
  EXPECT_EQ((original[1, 0]), 300.0f);

  // Randomized test: Verify view mutation propagation on random offset
  // positions
  std::vector<size_t> rand_s = random_shape();
  Tensor rand_base = Tensor::rand(rand_s, gen);
  size_t rand_size = rand_base.numel();
  if (rand_size > 1) {
    std::uniform_int_distribution<size_t> off_dist(0, rand_size - 1);
    size_t rand_off = off_dist(gen);
    size_t view_len = rand_size - rand_off;
    Tensor rand_view({view_len}, {1}, rand_base.storage(), rand_off);

    float new_val = 123.45f;
    rand_view[0] = new_val;
    EXPECT_EQ(rand_base.data()[rand_off], new_val);
  }
}

/* Metadata and Properties */

TEST_F(TensorTests, TestContiguityAndMetadata) {
  // Standard contiguous row-major tensor
  Tensor contiguous_tensor({2, 3, 4});
  EXPECT_TRUE(contiguous_tensor.is_contiguous());
  EXPECT_FALSE(contiguous_tensor.empty());
  EXPECT_EQ(contiguous_tensor.ndim(), 3u);
  EXPECT_EQ(contiguous_tensor.numel(), 24u);

  // Non-contiguous view (transposed strides simulation)
  std::vector<size_t> shape = {2, 3};
  std::vector<size_t> non_contiguous_strides = {1, 2};  // Column-major strides
  Tensor non_contiguous(shape, non_contiguous_strides,
                        contiguous_tensor.storage(), 0);
  EXPECT_FALSE(non_contiguous.is_contiguous());

  // Empty tensor contiguity edge case
  Tensor empty_tensor;
  EXPECT_TRUE(empty_tensor.is_contiguous());
  EXPECT_TRUE(empty_tensor.empty());

  // Randomized test: Verify properties match expected dimensions for random
  // shapes
  std::vector<size_t> rand_s = random_shape();
  Tensor rand_t(rand_s);
  EXPECT_TRUE(rand_t.is_contiguous());
  EXPECT_EQ(rand_t.ndim(), rand_s.size());
  EXPECT_EQ(rand_t.numel(), compute_size(rand_s));
  EXPECT_EQ(rand_t.shape(), rand_s);
  EXPECT_EQ(rand_t.stride(), compute_strides(rand_s));
}

TEST_F(TensorTests, ViewFunctionality) {
  // =========================================================================
  // 1. HANDWRITTEN EDGE CASES
  // =========================================================================

  // --- Case A: Identity View (Same Shape) ---
  {
    std::vector<size_t> shape = {2, 3, 4};
    tensors::Tensor t = tensors::Tensor::rand(shape, gen);
    tensors::Tensor v = t.view(shape);

    EXPECT_EQ(v.shape(), shape);
    EXPECT_EQ(v.stride(), t.stride());
    EXPECT_EQ(v.numel(), t.numel());
    EXPECT_EQ(v.offset(), t.offset());
  }

  // --- Case B: Flattening Dimensions (1D View) ---
  {
    std::vector<size_t> orig_shape = {2, 3, 4};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);
    std::vector<size_t> new_shape = {24};
    tensors::Tensor v = t.view(new_shape);

    EXPECT_EQ(v.shape(), new_shape);
    EXPECT_EQ(v.stride(), (std::vector<size_t>{1}));
    EXPECT_EQ(v.numel(), t.numel());
    EXPECT_EQ(v.offset(), t.offset());
  }

  // --- Case C: Unflattening Dimensions (Expanding Rank) ---
  {
    std::vector<size_t> orig_shape = {24};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);
    std::vector<size_t> new_shape = {2, 3, 4};
    tensors::Tensor v = t.view(new_shape);

    EXPECT_EQ(v.shape(), new_shape);
    EXPECT_EQ(v.stride(), (std::vector<size_t>{12, 4, 1}));
    EXPECT_EQ(v.numel(), t.numel());
  }

  // --- Case D: 1-Sized / Scalar Dimensions Insertion ---
  {
    std::vector<size_t> orig_shape = {2, 3};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);
    std::vector<size_t> new_shape = {1, 2, 1, 3, 1};
    tensors::Tensor v = t.view(new_shape);

    EXPECT_EQ(v.shape(), new_shape);
    EXPECT_EQ(v.stride(), (std::vector<size_t>{6, 3, 3, 1, 1}));
    EXPECT_EQ(v.numel(), t.numel());
  }

  // --- Case E: Transposed Non-Contiguous Tensor (Incompatible Flat View) ---
  {
    std::vector<size_t> orig_shape = {4, 4};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);

    if constexpr (requires { t.transpose(0, 1); }) {
      tensors::Tensor tr = t.transpose(0, 1);
      // Flattening across non-contiguous boundaries must fail
      EXPECT_THROW(tr.view({16}), std::runtime_error);
    }
  }

  // --- Case F: Non-Contiguous Dimension Merging Failure ---
  {
    std::vector<size_t> orig_shape = {2, 3, 4};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);

    if constexpr (requires { t.transpose(0, 1); }) {
      tensors::Tensor tr = t.transpose(0, 1); // shape becomes [3, 2, 4]
      // Attempting to merge non-contiguous dimensions {3, 2} into {6} fails
      EXPECT_THROW(tr.view({6, 4}), std::runtime_error);
    }
  }

  // --- Case G: Singleton Round-Trip ---
  {
    std::vector<size_t> orig_shape = {2, 3};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);
    tensors::Tensor v = t.view({1, 2, 1, 3, 1});
    tensors::Tensor r = v.view({2, 3});

    EXPECT_EQ(r.shape(), t.shape());
    EXPECT_EQ(r.stride(), t.stride());
  }

  // --- Case H: Non-Contiguous Reshape Without Crossing Boundary (Allowed in PyTorch) ---
  {
    std::vector<size_t> orig_shape = {2, 3, 4};
    tensors::Tensor t = tensors::Tensor::rand(orig_shape, gen);

    if constexpr (requires { t.transpose(0, 1); }) {
      tensors::Tensor tr = t.transpose(0, 1); // shape: [3, 2, 4]
      // Splitting dim 2 (size 4) into {2, 2} does NOT cross non-contiguous boundary
      tensors::Tensor v = tr.view({3, 2, 2, 2});
      EXPECT_EQ(v.shape(), (std::vector<size_t>{3, 2, 2, 2}));
    }
  }

  // =========================================================================
  // 2. RANDOMIZED TESTS USING FIXTURE (WITH DATA INTEGRITY CHECKS)
  // =========================================================================

  const size_t num_random_iterations = 50;

  for (size_t i = 0; i < num_random_iterations; ++i) {
    tensors::Tensor t = random_tensor();
    size_t total_elements = t.numel();

    // -----------------------------------------------------------------------
    // Random Case 1: Flattening to 1D and re-shaping to original
    // -----------------------------------------------------------------------
    std::vector<size_t> flat_shape = {total_elements};
    tensors::Tensor flat_view = t.view(flat_shape);

    EXPECT_EQ(flat_view.numel(), t.numel());
    EXPECT_EQ(flat_view.shape(), flat_shape);
    EXPECT_EQ(flat_view.offset(), t.offset());

    // Verify Data Mapping via Iterators / Indexing
    if constexpr (requires { t.data(); }) {
      for (size_t idx = 0; idx < total_elements; ++idx) {
        EXPECT_EQ(flat_view.data()[idx], t.data()[idx]);
      }
    }

    // Round-trip back to original shape
    tensors::Tensor restored_view = flat_view.view(t.shape());
    EXPECT_EQ(restored_view.shape(), t.shape());
    EXPECT_EQ(restored_view.stride(), t.stride());

    // -----------------------------------------------------------------------
    // Random Case 2: Factorizing total elements into valid random dimensions
    // -----------------------------------------------------------------------
    std::vector<size_t> random_new_shape;
    size_t rem = total_elements;

    std::uniform_int_distribution<size_t> factor_distrib(1, 4);
    while (rem > 1) {
      size_t factor = factor_distrib(gen);
      while (rem % factor != 0 && factor > 1) {
        --factor;
      }
      random_new_shape.push_back(factor);
      rem /= factor;
    }

    if (random_new_shape.empty()) {
      random_new_shape.push_back(1);
    }

    tensors::Tensor factorized_view = t.view(random_new_shape);
    EXPECT_EQ(factorized_view.numel(), t.numel());
    EXPECT_EQ(factorized_view.shape(), random_new_shape);
    EXPECT_EQ(factorized_view.offset(), t.offset());

    // Verify Data Mapping
    if constexpr (requires { t.data(); }) {
      for (size_t idx = 0; idx < total_elements; ++idx) {
        EXPECT_EQ(factorized_view.data()[idx], t.data()[idx]);
      }
    }
  }
}