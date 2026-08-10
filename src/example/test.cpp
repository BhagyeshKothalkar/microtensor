#include <iostream>
#include <random>
#include <vector>

#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optimizer.hpp"

using namespace tensors;
using namespace tensors::nn;
using namespace tensors::optim;

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

static float scalar_value(const Tensor& x) { return x.data()[0]; }

int main() {
  std::mt19937 rng(123);

  // 5 trainable Linear layers:
  //
  // 4 -> 16 -> 16 -> 16 -> 8 -> 1
  //
  // with ReLU between each pair.
  Linear l1(4, 16);
  Linear l2(16, 16);
  Linear l3(16, 16);
  Linear l4(16, 8);
  Linear l5(8, 1);

  init_linear(l1, rng);
  init_linear(l2, rng);
  init_linear(l3, rng);
  init_linear(l4, rng);
  init_linear(l5, rng);

  std::vector<Tensor*> params = {
      &l1.weight(), &l1.bias(),   &l2.weight(), &l2.bias(),   &l3.weight(),
      &l3.bias(),   &l4.weight(), &l4.bias(),   &l5.weight(), &l5.bias(),
  };

  optim::SGD optimizer(params, 0.01f);

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
    Tensor h1 = functional::relu(l1.forward(x));
    Tensor h2 = functional::relu(l2.forward(h1));
    Tensor h3 = functional::relu(l3.forward(h2));
    Tensor h4 = functional::relu(l4.forward(h3));
    Tensor prediction = l5.forward(h4);

    // MSE.
    Tensor diff = functional::sub(prediction, target);
    Tensor loss = functional::mean(functional::mul(diff, diff));

    float loss_value = scalar_value(loss);

    // Backward.
    loss.backward();

    // SGD.
    optimizer.step();
    optimizer.zero_grad();

    std::cout << "step " << step << " loss = " << loss_value
              << " prediction = " << prediction.data()[0] << '\n';
  }
}