#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "microtensor/cpu.hpp"
#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optim.hpp"
#include "microtensor/shape.hpp"
#include "microtensor/tensor.hpp"

using namespace microtensor;

namespace {

size_t total_tests = 0;
size_t failed_tests = 0;

void check(bool condition, const std::string& name) {
  ++total_tests;

  if (condition) {
    std::cout << "[PASS] " << name << "\n";
  } else {
    ++failed_tests;
    std::cout << "[FAIL] " << name << "\n";
  }
}

void check_close(float actual, float expected, const std::string& name,
                 float eps = 1e-5f) {
  check(std::fabs(actual - expected) < eps, name);
}

Tensor make_vector(std::initializer_list<float> values) {
  std::array<size_t, 1> shape{values.size()};

  Tensor result(shape);

  size_t index = 0;

  for (float value : values) {
    result.data()[index++] = value;
  }

  return result;
}

Tensor make_matrix(size_t rows, size_t cols) {
  Tensor result = Tensor::zeros(std::array<size_t, 2>{rows, cols});

  for (size_t i = 0; i < result.numel(); ++i) {
    result.data()[i] = 1.0f;
  }

  return result;
}

}  // namespace

void test_tensor() {
  std::cout << "\nTensor\n";

  Tensor x(std::array<size_t, 2>{2, 3});

  check(x.ndim() == 2, "rank");

  check(x.numel() == 6, "numel");

  check(x.shape()[0] == 2 && x.shape()[1] == 3, "shape");

  Tensor scalar = Tensor::zeros(std::array<size_t, 1>{1});

  check(scalar.numel() == 1, "scalar");
}

void test_cpu() {
  std::cout << "\nCPU\n";

  Tensor input = make_vector({1, 2, 3});

  Tensor output = Tensor::zeros(std::array<size_t, 1>{3});

  cpu::unary(output, input, [](float x) { return x * x; });

  check_close(output.data()[2], 9, "unary");

  Tensor binary = Tensor::zeros(std::array<size_t, 1>{3});

  cpu::binary(binary, input, output, [](float a, float b) { return a + b; });

  check_close(binary.data()[1], 6, "binary");

  Tensor reduced = Tensor::zeros(std::array<size_t, 1>{1});

  cpu::sum(reduced, input, {});

  check_close(reduced.data()[0], 6, "sum");
}

void test_broadcast() {
  std::cout << "\nBroadcast\n";

  Tensor a = Tensor::zeros(std::array<size_t, 2>{2, 1});

  Tensor b = Tensor::zeros(std::array<size_t, 2>{1, 3});

  std::array<const Tensor*, 2> inputs{&a, &b};

  auto result = broadcast_shape(inputs);

  check(result[0] == 2 && result[1] == 3, "broadcast shape");
}

void test_functional() {
  std::cout << "\nFunctional\n";

  Tensor a = make_vector({1, 2, 3});

  Tensor b = make_vector({4, 5, 6});

  Tensor c = add(a, b);

  check_close(c.data()[0], 5, "add");

  Tensor d = mul(a, b);

  check_close(d.data()[2], 18, "mul");

  Tensor e = relu(neg(a));

  check_close(e.data()[0], 0, "relu");
}

void test_autograd() {
  std::cout << "\nAutograd\n";

  Tensor x = make_vector({1, 2, 3});

  x.requires_grad(true);

  Tensor y = mul(x, x);

  Tensor loss = sum(y);

  loss.backward();

  auto* gradient = x.grad();

  check(gradient != nullptr, "gradient exists");

  if (gradient) {
    check_close(gradient->data()[0], 2, "x2 gradient first");

    check_close(gradient->data()[2], 6, "x2 gradient last");
  }
}

void test_nn() {
  std::cout << "\nNN\n";

  Linear layer(3, 2);

  Tensor input = Tensor::zeros(std::array<size_t, 2>{1, 3});

  Tensor output = layer(input);

  check(output.shape()[0] == 1 && output.shape()[1] == 2, "linear shape");

  check(layer.parameters().size() > 0, "linear parameters");
}

void test_optimizer() {
  std::cout << "\nOptimizer\n";

  Tensor weight = make_vector({10});

  weight.requires_grad(true);

  Tensor loss = mul(weight, weight);

  loss.backward();

  float before = weight.data()[0];

  SGD optimizer({&weight}, 0.1f);

  optimizer.step();

  check(weight.data()[0] < before, "sgd update");

  optimizer.zero_grad();

  check(weight.grad() != nullptr, "zero grad");
}

int main() {
  test_tensor();

  test_cpu();

  test_broadcast();

  test_functional();

  test_autograd();

  test_nn();

  test_optimizer();

  std::cout << "\n"
            << total_tests << " tests, " << failed_tests << " failures\n";

  return failed_tests == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
