// main.cpp
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "microtensor/cpu_kernels.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

// Note: Ensure your kernel functions (add, elementwise_multiply,
// scalar_multiply, naive_matmul) are either included via a header
// here or defined above these tests.
// #include "microtensor/tensor_math.hpp"
using namespace tensors;

// Helper to print shapes
void print_shape(const std::string &name, const std::vector<size_t> &shape) {
  std::cout << name << " shape: [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
  }
  std::cout << "]\n";
}

void test_metadata_and_layout() {
  std::cout << "--- Test 1: Tensor Metadata & Layout ---\n";
  Tensor<float> t({2uz, 3uz, 4uz});
  assert((t.ndim() == 3));
  assert((t.numel() == 24));
  assert((!t.empty()));

  // Strides for [2, 3, 4] should be [12, 4, 1]
  auto strides = t.stride();
  assert((strides[0] == 12));
  assert((strides[1] == 4));
  assert((strides[2] == 1));

  std::cout << "Metadata and stride logic passed.\n\n";
}

void test_broadcasting_rules() {
  std::cout << "--- Test 2: Shape Inference & Stride Rules ---\n";

  // Complex PyTorch example
  Tensor<float> A({5uz, 1uz, 4uz, 1uz});
  Tensor<float> B({3uz, 1uz, 1uz});

  assert((are_broadcastable(A, B) == true));

  auto target_shape = get_broadcast_shape(A, B);
  assert((target_shape.size() == 4));
  assert((target_shape[0] == 5 && target_shape[1] == 3 &&
          target_shape[2] == 4 && target_shape[3] == 1));

  // Test the variadic tuple API
  auto [bA, bB] = broadcast_tensors(target_shape, A, B);

  // Check if the stretches and padding applied 0-strides correctly
  // A: original [5, 1, 4, 1] -> strides [4, 4, 1, 1]
  // A target:  [5, 3, 4, 1] -> strides [4, 0, 1, 1] (dim 1 stretched)
  assert((bA.stride()[0] == 4));
  assert((bA.stride()[1] == 0)); // Stretched!
  assert((bA.stride()[2] == 1));

  // B: original [3, 1, 1] -> strides [1, 1, 1]
  // B target:  [5, 3, 4, 1] -> strides [0, 1, 0, 0] (dim 0 padded, dim 2,3
  // stretched)
  assert((bB.stride()[0] == 0)); // Padded!
  assert((bB.stride()[1] == 1));
  assert((bB.stride()[2] == 0)); // Stretched!

  // Test Exception for bad shapes
  Tensor<float> P({5uz, 2uz, 4uz, 1uz});
  Tensor<float> Q({3uz, 1uz, 1uz});

  assert((!are_broadcastable(P, Q)));
  bool caught = false;
  try {
    get_broadcast_shape(P, Q);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  assert((caught));

  std::cout << "Shape inference and zero-stride mapping passed.\n\n";
}

void test_2d_auto_broadcast() {
  std::cout << "--- Test 3: 2D Auto-Broadcasting (C = A + B) ---\n";
  // A is 3x1 (column vector)
  Tensor<float> A({3uz, 1uz});
  A[0uz, 0uz] = 1.0f;
  A[1uz, 0uz] = 2.0f;
  A[2uz, 0uz] = 3.0f;

  // B is 2 (1D vector) -> acts as 1x2 row vector during broadcast
  Tensor<float> B({2uz});
  B[0uz] = 10.0f;
  B[1uz] = 20.0f;

  // 1. Get shape and broadcast inputs
  auto target_shape = get_broadcast_shape(A, B); // Should be [3, 2]
  auto [bA, bB] = broadcast_tensors(target_shape, A, B);

  // 2. Allocate output
  Tensor<float> C(target_shape);

  // 3. Iterate and compute
  TensorIterator<float, float, float> it(C, bA, bB);
  while (it.has_next()) {
    auto [c, a, b] = it.next();
    c = a + b;
  }

  // Expected C:
  // [11, 21]
  // [12, 22]
  // [13, 23]
  assert((C[0uz, 0uz] == 11.0f && C[0uz, 1uz] == 21.0f));
  assert((C[1uz, 0uz] == 12.0f && C[1uz, 1uz] == 22.0f));
  assert((C[2uz, 0uz] == 13.0f && C[2uz, 1uz] == 23.0f));

  std::cout << "2D auto-broadcasting execution passed.\n\n";
}

void test_n_ary_high_dimensional_broadcast() {
  std::cout << "--- Test 4: N-ary 3D Broadcasting (D = A + B + C) ---\n";

  Tensor<float> A({2uz, 1uz, 3uz}); // 2 batches, 1 row, 3 cols
  Tensor<float> B({3uz});           // 1D vector of size 3
  Tensor<float> C({1uz});           // Scalar-like 1D tensor

  // Init data
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      A[i, 0uz, j] = (i + 1) * 100.0f + (j + 1);
    }
  }
  B[0uz] = 10.0f;
  B[1uz] = 20.0f;
  B[2uz] = 30.0f;
  C[0uz] = 1000.0f;

  auto out_shape = get_broadcast_shape(A, B, C); // Expected: [2, 1, 3]
  auto [bA, bB, bC] = broadcast_tensors(out_shape, A, B, C);
  Tensor<float> D(out_shape);

  TensorIterator<float, float, float, float> it(D, bA, bB, bC);

  size_t step_count = 0;
  while (it.has_next()) {
    auto [d, a, b, c] = it.next();
    d = a + b + c;
    step_count++;
  }

  assert((step_count == 6)); // 2 * 1 * 3 elements

  // Batch 0
  assert((D[0uz, 0uz, 0uz] == 101.0f + 10.0f + 1000.0f));
  assert((D[0uz, 0uz, 2uz] == 103.0f + 30.0f + 1000.0f));
  // Batch 1
  assert((D[1uz, 0uz, 0uz] == 201.0f + 10.0f + 1000.0f));
  assert((D[1uz, 0uz, 2uz] == 203.0f + 30.0f + 1000.0f));

  std::cout << "Complex N-ary high-dimensional broadcasting passed.\n\n";
}

void test_elementwise_kernels() {
  std::cout << "--- Test 5: Elementwise Kernels (Add, Mul, Scalar) ---\n";

  // A is 2x1
  Tensor<float> A({2uz, 1uz});
  A[0uz, 0uz] = 1.0f;
  A[1uz, 0uz] = 2.0f;

  // B is 1x2
  Tensor<float> B({1uz, 2uz});
  B[0uz, 0uz] = 3.0f;
  B[0uz, 1uz] = 4.0f;

  // 1. Test add kernel (expected broadcast to 2x2)
  auto C_add = add(A, B);
  assert((C_add.shape() == std::vector<size_t>{2uz, 2uz}));
  assert((C_add[0uz, 0uz] == 4.0f)); // 1 + 3
  assert((C_add[0uz, 1uz] == 5.0f)); // 1 + 4
  assert((C_add[1uz, 0uz] == 5.0f)); // 2 + 3
  assert((C_add[1uz, 1uz] == 6.0f)); // 2 + 4

  // 2. Test elementwise_multiply kernel
  auto C_mul = elementwise_multiply(A, B);
  assert((C_mul.shape() == std::vector<size_t>{2uz, 2uz}));
  assert((C_mul[0uz, 0uz] == 3.0f)); // 1 * 3
  assert((C_mul[0uz, 1uz] == 4.0f)); // 1 * 4
  assert((C_mul[1uz, 0uz] == 6.0f)); // 2 * 3
  assert((C_mul[1uz, 1uz] == 8.0f)); // 2 * 4

  // 3. Test scalar_multiply kernel
  auto C_scalar = scalar_multiply(A, 10.0f);
  assert((C_scalar.shape() == std::vector<size_t>{2uz, 1uz}));
  assert((C_scalar[0uz, 0uz] == 10.0f));
  assert((C_scalar[1uz, 0uz] == 20.0f));

  std::cout << "Element-wise mathematical kernels passed.\n\n";
}

void test_matmul_kernel() {
  std::cout << "--- Test 6: Matrix Multiplication (naive_matmul) ---\n";

  // A: 2x3 matrix
  Tensor<float> A({2uz, 3uz});
  A[0uz, 0uz] = 1.0f;
  A[0uz, 1uz] = 2.0f;
  A[0uz, 2uz] = 3.0f;
  A[1uz, 0uz] = 4.0f;
  A[1uz, 1uz] = 5.0f;
  A[1uz, 2uz] = 6.0f;

  // B: 3x2 matrix
  Tensor<float> B({3uz, 2uz});
  B[0uz, 0uz] = 7.0f;
  B[0uz, 1uz] = 8.0f;
  B[1uz, 0uz] = 9.0f;
  B[1uz, 1uz] = 10.0f;
  B[2uz, 0uz] = 11.0f;
  B[2uz, 1uz] = 12.0f;

  // Compute C = A @ B
  auto C = naive_matmul(A, B);

  // Expected Shape: 2x2
  assert((C.shape() == std::vector<size_t>{2uz, 2uz}));

  // C[0, 0] = (1*7) + (2*9) + (3*11) = 7 + 18 + 33 = 58
  assert((C[0uz, 0uz] == 58.0f));

  // C[0, 1] = (1*8) + (2*10) + (3*12) = 8 + 20 + 36 = 64
  assert((C[0uz, 1uz] == 64.0f));

  // C[1, 0] = (4*7) + (5*9) + (6*11) = 28 + 45 + 66 = 139
  assert((C[1uz, 0uz] == 139.0f));

  // C[1, 1] = (4*8) + (5*10) + (6*12) = 32 + 50 + 72 = 154
  assert((C[1uz, 1uz] == 154.0f));

  std::cout << "Matrix multiplication kernel passed.\n\n";
}

int main() {
  std::cout << "====================================\n";
  std::cout << "   microtensor Core Stress Tests     \n";
  std::cout << "====================================\n\n";

  test_metadata_and_layout();
  test_broadcasting_rules();
  test_2d_auto_broadcast();
  test_n_ary_high_dimensional_broadcast();

  // New kernel tests
  test_elementwise_kernels();
  test_matmul_kernel();

  std::cout << "ALL TESTS PASSED SUCCESSFULLY.\n";
  return 0;
}