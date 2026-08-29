# microtensor

`microtensor` is a small, CPU-backed tensor and neural-network library written
in modern C++. It provides the building blocks for experimenting with tensor
programming, automatic differentiation, and compact neural-network models.

The public API lives in the `tensors` namespace. The project currently targets
C++26 and builds a static library named `microtensor_lib`.

## Features tour

- **Tensor storage and indexing** – Create tensors from shapes or initializer
  lists, access elements with variadic indices, inspect shape/stride metadata,
  and use `zeros`, `ones`, `full`, `linspace`, `rand`, and `clone` helpers.
- **Views and layout operations** – Create reshaped views with `view`, swap
  dimensions with `transpose`, reorder dimensions with `permute`, materialize
  contiguous storage with `contiguous`, and divide tensors with `split` or
  `chunk`.
- **Broadcasting** – Elementwise operations support NumPy-style broadcasting;
  explicit helpers are available for checking and constructing broadcasted
  shapes.
- **Functional tensor operations** – Arithmetic, unary math, reductions,
  normalization, matrix multiplication, indexing, masking, concatenation,
  softmax, log-softmax, and cross-entropy are available through
  `tensors::functional` (also aliased as `tensors::F`).
- **Automatic differentiation** – Mark tensors with
  `set_requires_grad(true)`, compute a scalar result, call `backward()`, and
  read or clear gradients with `grad()`, `mutable_grad()`, and `zero_grad()`.
  `NoGradGuard` disables graph construction for updates and inference helpers.
- **Neural-network modules** – Build models from `Module`, `Sequential`,
  `Linear`, `ReLU`, `GELU`, `Dropout`, `Embedding`, `LayerNorm`, and
  `MultiHeadAttention`. Modules expose recursive parameter traversal and
  propagate training/evaluation mode to children.
- **Optimizers** – Update registered parameters with `optim::SGD` or
  `optim::Adam`, using `zero_grad()` between training steps.
- **CPU implementation** – Low-level elementwise, reduction, activation, and
  matrix-multiplication kernels are exposed for library internals and advanced
  users.

## Building

### Prerequisites

- CMake 3.14 or newer
- Ninja
- A C++ compiler with C++26 support (GCC or Clang recommended)

From the project root, run:

```bash
./scripts/build.sh
```

This configures a Debug Ninja build in `build/` with testing disabled and
builds the library and examples. For a manual configuration:

```bash
cmake -S . -B build -G Ninja -DENABLE_TESTING=OFF
ninja -C build
```

The repository's maintained executable checks are under `src/example`.

## Running the examples

Executables are placed in `build/bin/`:

```bash
./build/bin/testing
./build/bin/regtesting
./build/bin/gpt2
```

`testing` exercises functional operations, reductions, autograd, modules, and
attention. `regtesting` runs broader tensor and small-model regression checks.
`gpt2` demonstrates a compact transformer-style model with embeddings,
attention, normalization, cross-entropy, and Adam updates.

## Quick example

```cpp
#include "microtensor/functional.hpp"
#include "microtensor/tensor.hpp"

using namespace tensors;

int main() {
  Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
  x.set_requires_grad(true);

  Tensor loss = functional::sum(functional::mul(x, x));
  loss.backward();

  // x.grad() contains the gradient of the scalar loss with respect to x.
  return 0;
}
```

When compiling an application against a build-tree library, add the project
root (which contains `src/microtensor`) to the include path and link against
`build/src/microtensor/libmicrotensor_lib.a` or the corresponding CMake target
`microtensor_lib`.

## Core headers

| Header | Main API | Purpose |
| --- | --- | --- |
| `microtensor/tensor.hpp` | `Tensor`, `compute_strides`, `compute_size` | Tensor storage, indexing, metadata, initialization, views, layout operations, and gradient access. |
| `microtensor/functional.hpp` | `functional::add/sub/mul/div`, `sum`, `max`, `mean`, `matmul`, `softmax`, `logsoftmax`, `layer_norm`, `rmsnorm`, `cross_entropy`, `cat`, `index_select`, `masked_fill` | Autograd-aware tensor operations. Arithmetic operators are also provided for common scalar and tensor arithmetic. |
| `microtensor/autograd.hpp` | `AutogradContext`, `NoGradGuard`, `IGradNode`, `make_grad_node` | Graph construction controls and the internal gradient-node primitives used by `Tensor::backward()`. |
| `microtensor/broadcasting.hpp` | `are_broadcastable`, `get_broadcast_shape`, `broadcast_to_shape`, `broadcast_tensors` | Validate broadcasting and create tensors with a common broadcast shape. |
| `microtensor/tensor_iterator.hpp` | `TensorIterator`, `ReductionIterator` | Stride-aware elementwise and reduction iteration for implementing tensor kernels. |
| `microtensor/cpu_kernels.hpp` | `cpu_kernels::add/sub/mul/div`, activations, `naive_matmul`, `sum`, `clone` | In-place CPU kernels used by the functional layer and optimizers. |
| `microtensor/nn.hpp` | `nn::Module`, `Sequential`, `Linear`, `ReLU`, `GELU`, `Dropout`, `Embedding`, `LayerNorm`, `MultiHeadAttention` | Neural-network modules, model composition, parameter registration, and train/eval mode. |
| `microtensor/optimizer.hpp` | `optim::Optimizer`, `SGD`, `Adam` | Gradient clearing and parameter updates. |

## Typical training loop

```cpp
#include "microtensor/functional.hpp"
#include "microtensor/nn.hpp"
#include "microtensor/optimizer.hpp"

using namespace tensors;

nn::Sequential model;
model.emplace<nn::Linear>(4, 8);
model.emplace<nn::GELU>();
model.emplace<nn::Linear>(8, 2);

optim::Adam optimizer(model.parameters_recursive(), 1e-3f);

Tensor logits = model.forward(input);
Tensor loss = functional::cross_entropy(logits, targets);
loss.backward();
optimizer.step();
optimizer.zero_grad();
```

Use `model.eval()` for inference and `model.train()` to return to training
mode. `Dropout` observes the module mode.

## Project layout

- `src/microtensor/` – library implementation and public headers
- `src/example/` – maintained executable examples and feature checks
- `scripts/` – build, test, and formatting entry points
- `CMakeLists.txt` – project configuration and build options

To format C++ sources and headers using the repository configuration:

```bash
./scripts/format.sh
```
