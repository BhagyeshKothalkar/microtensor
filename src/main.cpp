#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "microtensor/broadcasting.hpp"
#include "microtensor/cpu_kernels.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/tensor.hpp"
#include "microtensor/tensor_iterator.hpp"

using namespace tensors;

void print_shape(const std::string& name, const std::vector<size_t>& shape) {
  std::cout << name << " shape: [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
  }
  std::cout << "]\n";
}

void test_metadata_and_layout() {
  std::cout << "--- Test 1: Tensor Metadata & Layout ---\n";
  Tensor t({2uz, 3uz, 4uz});
  assert((t.ndim() == 3));
  assert((t.numel() == 24));
  assert((!t.empty()));

  auto strides = t.stride();
  assert((strides[0] == 12));
  assert((strides[1] == 4));
  assert((strides[2] == 1));

  std::cout << "Metadata and stride logic passed.\n\n";
}

void test_broadcasting_rules() {
  std::cout << "--- Test 2: Shape Inference & Stride Rules ---\n";

  Tensor A({5uz, 1uz, 4uz, 1uz});
  Tensor B({3uz, 1uz, 1uz});

  assert((are_broadcastable(A, B) == true));

  auto target_shape = get_broadcast_shape(A, B);
  assert((target_shape.size() == 4));
  assert((target_shape[0] == 5 && target_shape[1] == 3 &&
          target_shape[2] == 4 && target_shape[3] == 1));

  auto [bA, bB] = broadcast_tensors(target_shape, A, B);

  assert((bA.stride()[0] == 4));
  assert((bA.stride()[1] == 0));
  assert((bA.stride()[2] == 1));

  assert((bB.stride()[0] == 0));
  assert((bB.stride()[1] == 1));
  assert((bB.stride()[2] == 0));

  Tensor P({5uz, 2uz, 4uz, 1uz});
  Tensor Q({3uz, 1uz, 1uz});

  assert((!are_broadcastable(P, Q)));
  bool caught = false;
  try {
    get_broadcast_shape(P, Q);
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert((caught));

  std::cout << "Shape inference and zero-stride mapping passed.\n\n";
}

void test_2d_auto_broadcast() {
  std::cout << "--- Test 3: 2D Auto-Broadcasting (C = A + B) ---\n";

  Tensor A({3uz, 1uz});
  A[0uz, 0uz] = 1.0f;
  A[1uz, 0uz] = 2.0f;
  A[2uz, 0uz] = 3.0f;

  Tensor B({2uz});
  B[0uz] = 10.0f;
  B[1uz] = 20.0f;

  auto target_shape = get_broadcast_shape(A, B);
  auto [bA, bB] = broadcast_tensors(target_shape, A, B);

  Tensor C(target_shape);

  TensorIterator<float, const float, const float> it(C, bA, bB);
  while (it.has_next()) {
    auto [c, a, b] = it.next();
    c = a + b;
  }

  assert((C[0uz, 0uz] == 11.0f && C[0uz, 1uz] == 21.0f));
  assert((C[1uz, 0uz] == 12.0f && C[1uz, 1uz] == 22.0f));
  assert((C[2uz, 0uz] == 13.0f && C[2uz, 1uz] == 23.0f));

  std::cout << "2D auto-broadcasting execution passed.\n\n";
}

void test_n_ary_high_dimensional_broadcast() {
  std::cout << "--- Test 4: N-ary 3D Broadcasting (D = A + B + C) ---\n";

  Tensor A({2uz, 1uz, 3uz});
  Tensor B({3uz});
  Tensor C({1uz});

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      A[i, 0uz, j] = (i + 1) * 100.0f + (j + 1);
    }
  }
  B[0uz] = 10.0f;
  B[1uz] = 20.0f;
  B[2uz] = 30.0f;
  C[0uz] = 1000.0f;

  auto out_shape = get_broadcast_shape(A, B, C);
  auto [bA, bB, bC] = broadcast_tensors(out_shape, A, B, C);
  Tensor D(out_shape);

  TensorIterator<float, const float, const float, const float> it(D, bA, bB,
                                                                  bC);

  size_t step_count = 0;
  while (it.has_next()) {
    auto [d, a, b, c] = it.next();
    d = a + b + c;
    step_count++;
  }

  assert((step_count == 6));

  assert((D[0uz, 0uz, 0uz] == 101.0f + 10.0f + 1000.0f));
  assert((D[0uz, 0uz, 2uz] == 103.0f + 30.0f + 1000.0f));

  assert((D[1uz, 0uz, 0uz] == 201.0f + 10.0f + 1000.0f));
  assert((D[1uz, 0uz, 2uz] == 203.0f + 30.0f + 1000.0f));

  std::cout << "Complex N-ary high-dimensional broadcasting passed.\n\n";
}

void test_elementwise_kernels() {
  std::cout << "--- Test 5: Elementwise Kernels (Add, Mul, Scalar) ---\n";

  Tensor A({2uz, 1uz});
  A[0uz, 0uz] = 1.0f;
  A[1uz, 0uz] = 2.0f;

  Tensor B({1uz, 2uz});
  B[0uz, 0uz] = 3.0f;
  B[0uz, 1uz] = 4.0f;

  auto C_add = add(A, B);
  assert((C_add.shape() == std::vector<size_t>{2uz, 2uz}));
  assert((C_add[0uz, 0uz] == 4.0f));
  assert((C_add[0uz, 1uz] == 5.0f));
  assert((C_add[1uz, 0uz] == 5.0f));
  assert((C_add[1uz, 1uz] == 6.0f));

  auto C_mul = elementwise_multiply(A, B);
  assert((C_mul.shape() == std::vector<size_t>{2uz, 2uz}));
  assert((C_mul[0uz, 0uz] == 3.0f));
  assert((C_mul[0uz, 1uz] == 4.0f));
  assert((C_mul[1uz, 0uz] == 6.0f));
  assert((C_mul[1uz, 1uz] == 8.0f));

  auto C_scalar = scalar_multiply(A, 10.0f);
  assert((C_scalar.shape() == std::vector<size_t>{2uz, 1uz}));
  assert((C_scalar[0uz, 0uz] == 10.0f));
  assert((C_scalar[1uz, 0uz] == 20.0f));

  std::cout << "Element-wise mathematical kernels passed.\n\n";
}

void test_matmul_kernel() {
  std::cout << "--- Test 6: Matrix Multiplication (naive_matmul) ---\n";

  Tensor A({2uz, 3uz});
  A[0uz, 0uz] = 1.0f;
  A[0uz, 1uz] = 2.0f;
  A[0uz, 2uz] = 3.0f;
  A[1uz, 0uz] = 4.0f;
  A[1uz, 1uz] = 5.0f;
  A[1uz, 2uz] = 6.0f;

  Tensor B({3uz, 2uz});
  B[0uz, 0uz] = 7.0f;
  B[0uz, 1uz] = 8.0f;
  B[1uz, 0uz] = 9.0f;
  B[1uz, 1uz] = 10.0f;
  B[2uz, 0uz] = 11.0f;
  B[2uz, 1uz] = 12.0f;

  auto C = naive_matmul(A, B);

  assert((C.shape() == std::vector<size_t>{2uz, 2uz}));

  assert((C[0uz, 0uz] == 58.0f));

  assert((C[0uz, 1uz] == 64.0f));

  assert((C[1uz, 0uz] == 139.0f));

  assert((C[1uz, 1uz] == 154.0f));

  std::cout << "Matrix multiplication kernel passed.\n\n";
}

void test_nn() {
  using namespace tensors;
  using namespace tensors::nn;

  // Combine them into a sequential container
  Sequential net({Linear(2, 4), Linear(4, 1)});

  // Create an input tensor with shape [in_dim, batch_size] -> {2, 1}
  Tensor input({2, 1}, {1.0f, -2.0f});

  // Run forward pass
  Tensor output = net.forward(input);

  std::cout << "Forward pass completed successfully! Output numel: "
            << output.numel() << std::endl;
}

int main() {
  std::cout << "====================================\n";
  std::cout << "   microtensor Core Stress Tests     \n";
  std::cout << "====================================\n\n";

  test_metadata_and_layout();
  test_broadcasting_rules();
  test_2d_auto_broadcast();
  test_n_ary_high_dimensional_broadcast();

  test_elementwise_kernels();
  test_matmul_kernel();

  test_nn();

  std::cout << "ALL TESTS PASSED SUCCESSFULLY.\n";
  return 0;
}
