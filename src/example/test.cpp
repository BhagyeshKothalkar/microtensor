#include <iostream>
#include <random>
#include <vector>

#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optimizer.hpp"

using namespace tensors;
using namespace tensors::nn;
using namespace tensors::optim;

static float scalar_value(const Tensor& x) { return x.data()[0]; }

static void init_linear(Linear& layer, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  layer.weight().set_requires_grad(true);
  layer.bias().set_requires_grad(true);

  for (size_t i = 0; i < layer.weight().numel(); ++i) {
    layer.weight().data()[i] = dist(rng);
  }

  for (size_t i = 0; i < layer.bias().numel(); ++i) {
    layer.bias().data()[i] = 0.0f;
  }
}
std::mt19937 rng(123);

int main() {
  // Focused feature harness; these APIs are intentionally added by this change.
  Tensor indexed({2, 3}, {1, 2, 3, 4, 5, 6});
  if (indexed[-1, -1] != 6.0f) return 1;
  auto permuted = indexed.permute({1, 0});
  if (permuted[-1, -1] != 6.0f) return 2;
  auto reduced = functional::sum(indexed, {-1});
  if (reduced.shape() != std::vector<size_t>{2} || reduced[0] != 6.0f) return 3;
  auto normalized = functional::rmsnorm(indexed, {-1});
  if (normalized.shape() != indexed.shape()) return 4;
  auto batched = functional::matmul(Tensor::ones({2, 3, 4}), Tensor::ones({2, 4, 5}));
  if (batched.shape() != std::vector<size_t>{2, 3, 5} || batched[0, 0, 0] != 4.0f) return 5;
  return 0;

  class mymodule : public Module {
    Linear l1;
    Linear l2;

   public:
    mymodule() : l1(4, 16), l2(16, 1) {
      init_linear(l1, rng);
      init_linear(l2, rng);
      register_children({{"l1", &l1}, {"l2", &l2}});
    }

    Tensor forward(const Tensor& x) override {
      return l2.forward(functional::relu(l1.forward(x)));
    }
  };

  mymodule model;

  for (const auto& [name, param] : model.named_parameters_recursive()) {
    std::cout << name << '\n';
  }

  // std::vector<Tensor*> params = model.

  optim::Adam optimizer(model.parameters_recursive(), 0.01f);

  // Dummy regression data.
  Tensor x({4}, {
                    0.5f,
                    -1.0f,
                    2.0f,
                    0.25f,
                });

  Tensor target({1}, {
                         0.75f,
                     });

  for (int step = 0; step < 100; ++step) {
    // Forward.
    Tensor prediction = model.forward(x);

    // MSE.
    Tensor diff = functional::sub(prediction, target);
    Tensor loss = functional::mean(functional::mul(diff, diff));

    float loss_value = scalar_value(loss);

    // Backward.
    loss.backward();

    // // SGD.
    optimizer.step();
    optimizer.zero_grad();

    std::cout << "step " << step << " loss = " << loss_value
              << " prediction = " << prediction.data()[0] << '\n';
  }
}
