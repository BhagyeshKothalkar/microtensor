#include <algorithm>
#include <cmath>
#include <iostream>

#include "microtensor/tensor_iterator.hpp"

using namespace tensors;

namespace {

int failures = 0;

void expect_close(float got, float expected, const char* name) {
  if (std::fabs(got - expected) > 1e-5f) {
    std::cerr << "[FAIL] " << name << " got=" << got << " expected=" << expected
              << "\n";

    failures++;
  } else {
    std::cout << "[PASS] " << name << "\n";
  }
}

void test_sum_reduce_last_dimension_lvalue() {
  Tensor src({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor dst({2});

  ReductionIterator<float, float> iter(std::vector<int>{1}, dst, src);

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float sum = 0;

    reduce.for_each([&](const float& x) { sum += x; });

    out = sum;
  });

  expect_close(dst[0], 6, "sum reduce dim1 row0");
  expect_close(dst[1], 15, "sum reduce dim1 row1");
}

void test_sum_reduce_first_dimension_lvalue() {
  Tensor src({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor dst({3});

  ReductionIterator<float, float> iter(std::vector<int>{0}, dst, src);

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float sum = 0;

    reduce.for_each([&](const float& x) { sum += x; });

    out = sum;
  });

  expect_close(dst[0], 5, "sum reduce dim0 col0");
  expect_close(dst[1], 7, "sum reduce dim0 col1");
  expect_close(dst[2], 9, "sum reduce dim0 col2");
}

void test_max_reduce() {
  Tensor src({2, 3}, {1, 8, 3, 4, 5, 9});

  Tensor dst({2});

  ReductionIterator<float, float> iter(std::vector<int>{1}, dst, src);

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float mx = -INFINITY;

    reduce.for_each([&](const float& x) { mx = std::max(mx, x); });

    out = mx;
  });

  expect_close(dst[0], 8, "max reduce row0");
  expect_close(dst[1], 9, "max reduce row1");
}

void test_mean_reduce() {
  Tensor src({2, 2}, {2, 4, 6, 8});

  Tensor dst({2});

  ReductionIterator<float, float> iter(std::vector<int>{1}, dst, src);

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float sum = 0;
    int count = 0;

    reduce.for_each([&](const float& x) {
      sum += x;
      count++;
    });

    out = sum / count;
  });

  expect_close(dst[0], 3, "mean row0");
  expect_close(dst[1], 7, "mean row1");
}

void test_constant_rvalue_source() {
  Tensor dst({2});

  ReductionIterator<float, float> iter(std::vector<int>{1}, dst,
                                       Tensor({2, 2}, {1, 2, 3, 4}));

  iter.for_each([](float& out, TensorIterator<const float> reduce) {
    float sum = 0;

    reduce.for_each([&](const float& x) { sum += x; });

    out = sum;
  });

  expect_close(dst[0], 3, "rvalue source row0");
  expect_close(dst[1], 7, "rvalue source row1");
}

void test_reduction_iterator() {
  test_sum_reduce_last_dimension_lvalue();
  test_sum_reduce_first_dimension_lvalue();
  test_max_reduce();
  test_mean_reduce();
  test_constant_rvalue_source();

  if (failures) {
    std::cerr << failures << " failures\n";
  } else {
    std::cout << "all reduction iterator tests passed\n";
  }
}

}  // namespace

int main() { test_reduction_iterator(); }
