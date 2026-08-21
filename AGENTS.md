# AGENTS.md

## 1. Executive Project Summary
- **Project Purpose & Domain:** `microtensor` is a lightweight, header-centric C++ tensor framework and neural network module runtime designed for minimal dependencies, transparent memory layouts, zero-copy strided views, and lock-step elementwise iteration kernels.
- **Language Standard:** C++26 (`set(CMAKE_CXX_STANDARD 26)` enforced in `CMakeLists.txt`).
- **External Dependencies:** GoogleTest (fetched dynamically via `FetchContent` during test builds). No third-party BLAS or tensor runtime dependencies are used.

## 2. Project Architecture & Directory Layout

### Visual Directory Tree
```
microtensor/
├── .clang-format           # Code formatting profile (BasedOnStyle: Google)
├── CMakeLists.txt          # Root build configuration (enforces C++26, Debug flags)
├── README.md               # Quickstart and known test failure log
├── scripts/                # Utility execution scripts
│   ├── build.sh            # Production/library release CMake + Ninja build trigger
│   ├── build_test.sh       # CMake + Ninja test build configuration trigger
│   ├── format.sh           # In-place code formatter wrapper using clang-format
│   └── run_tests.sh        # CTest output test executor wrapper
├── src/
│   └── microtensor/        # Core library header/source implementation
│       ├── autograd.hpp        # Zero-boilerplate reverse-mode autograd node closures & state
│       ├── autograd.cpp        # Topological DFS execution engine & gradient shape reduction
│       ├── broadcasting.hpp    # NumPy-style tensor broadcast validation & view creation
│       ├── cpu_kernels.hpp     # Low-level CPU execution kernel declarations
│       ├── cpu_kernels.cpp     # Elementwise, matmul, and activation kernel implementations
│       ├── functional.hpp      # High-level functional wrappers (in-place & out-of-place ops)
│       ├── nn.hpp              # Neural network Module, Linear layer, and Sequential container
│       ├── tensor.hpp          # Core Tensor class, striding, indexing, and storage views
│       └── tensor_iterator.hpp # Lock-step N-tensor iterator template over strided layouts
└── tests/
    ├── common/
    │   └── test_tensor.h   # GoogleTest fixture class (TensorTests) with RNG helpers
    └── microtensor/        # Unit test suites matching core headers
        ├── autograd.cpp        # Unit tests for Autograd engine & gradient verification
        ├── broadcasting.cpp    # Unit tests for shape broadcasting
        ├── cpu_kernels.cpp     # Unit tests for low-level CPU kernels
        ├── functional.cpp     # Unit tests for functional operators
        ├── nn.cpp              # Unit tests for neural net modules
        ├── tensor_iterator.cpp # Unit tests for TensorIterator step traversal
        └── tensors.cpp         # Unit tests for Tensor creation, indexing, views, & clone
```

### Key Architectural Patterns
- **Memory Management & Storage Sharing:**
  - Dynamic tensor payloads are backed by `std::shared_ptr<float[]>`.
  - Transposition, slicing, and broadcasting create new `Tensor` instances sharing the underlying `std::shared_ptr<float[]>` storage with updated `shape_`, `stride_`, and `offset_` metadata without copying elements.
  - Ownership transfer is explicitly avoided; raw pointers to `Tensor` / `Module` objects registered in `Module::register_parameters` and `Module::register_children` rely on caller-managed lifetimes.
- **Zero-Boilerplate Autograd Engine:**
  - `AutogradMeta` holds `std::unique_ptr<Tensor> grad_` and `std::shared_ptr<IGradNode> grad_fn_`, lazily allocated on `Tensor` to prevent memory overhead on non-differentiable tensors.
  - Computational graph nodes use templated `GradNode<Parents, BackwardFn>` captured via `make_parents(...)` and generic lambdas. `IGradNode::get_parents()` enables post-order DFS topological sorting in `Tensor::backward()` without recursion stack overflow or `std::function` allocation overhead.
  - Operation node registration evaluates `if (AutogradContext::is_enabled() && (a.requires_grad() || b.requires_grad()))` to bypass node construction for non-differentiable operations.
  - `Tensor::add_grad(const Tensor& g)` automatically reduces and sums out expanded broadcast dimensions via `reduce_sum_to` if `g.shape()` differs from `this->shape()`.
- **Lock-Step Traversal Kernel Architecture:**
  - `TensorIterator<Dest, Src...>` manages multi-tensor traversal using per-tensor stride steps and offsets across N-dimensional shapes.
  - Kernels (`cpu_kernels::add`, `cpu_kernels::relu`, `cpu_kernels::softmax`) leverage `TensorIterator` for uniform iteration without explicit nested indexing loops.
- **Thread-Safety Guidelines:**
  - `Tensor` metadata (`shape_`, `stride_`, `offset_`) is value-copied per view instance.
  - Accessing disjoint memory locations across threads via separate `Tensor` views sharing a standard buffer is thread-safe; simultaneous mutable access to the same flat index requires explicit external synchronization.

## 3. Build, Test, & Execution Commands

### Environment Requirements
- **CMake:** Version 3.14 or newer.
- **Generator:** Ninja (`cmake -G Ninja`).
- **Compiler:** GCC 13+/Clang 16+ supporting C++26 standard features (`-std=c++26`).

### Exact Terminal Commands
- **Configure & Build Library Only:**
  ```bash
  cmake -S . -B build -G Ninja
  ninja -C build
  ```
  *(Or execute `./scripts/build.sh` on Unix-like environments).*

- **Configure & Build Unit Tests:**
  ```bash
  cmake -S . -B build -G Ninja -DENABLE_TESTING=ON
  ninja -C build
  ```
  *(Or execute `./scripts/build_test.sh` on Unix-like environments).*

- **Execute Unit Test Suite:**
  ```bash
  ctest --test-dir build --output-on-failure
  ```
  *(Or execute `./scripts/run_tests.sh` on Unix-like environments).*

- **Format Source Code:**
  ```bash
  ./scripts/format.sh
  ```
  *(Runs `clang-format -i --style=file` on all `.cpp`, `.h`, `.cc`, `.hpp` files).*

## 4. C++ Code Style & Engineering Constraints

### Explicit Naming Conventions
- **Classes / Structs:** `PascalCase` (e.g., `Tensor`, `TensorIterator`, `Module`, `Linear`, `Sequential`).
- **Functions & Methods:** `snake_case` (e.g., `compute_strides()`, `is_contiguous()`, `as_strided()`, `forward()`).
  - In-place mutation operators append a trailing underscore: `add_()`, `relu_()`, `softmax_()`.
- **Member Variables:** `snake_case_` with a trailing underscore (e.g., `shape_`, `stride_`, `offset_`, `storage_`, `data_`, `named_children_`).
- **Namespaces:** `snake_case` (e.g., `tensors`, `tensors::cpu_kernels`, `tensors::functional`, `tensors::nn`).

### Autograd Op Creation & Node Conventions
- **Zero-Boilerplate Node Factories:** Use `make_parents(...)` and `make_grad_node(parents, backward_fn)` inside out-of-place functional operations.
- **Scope & Condition Evaluation:** Only construct graph nodes when `AutogradContext::is_enabled()` and at least one input tensor satisfies `requires_grad()`.
- **Inner Backward Isolation:** Wrap math executed inside backward pass closures with `NoGradGuard guard;` to prevent recursive autograd graph tracking during gradient evaluation.
- **Gradient Accumulation:** Call `lhs.add_grad(grad)` / `rhs.add_grad(...)` inside backward pass closures. `Tensor::add_grad` handles automatic un-broadcasting/reduction for shape mismatches.

### Error Handling & Invariants
- **Assertions:** Use standard `<cassert>` `assert(...)` for low-level internal invariants and shape boundaries in developer builds.
- **Exceptions:** Throw `std::invalid_argument` for incompatible broadcast shapes (e.g., in `get_broadcast_shape()`). Throw `std::runtime_error` for rank/dimension mismatches (e.g., matrix multiplication incompatibility or invalid shape views).
- **Return Types:** Use reference returns for in-place chainable operations (`Tensor&`) and direct object returns for created views/clones.

### Header Policy & Inclusion Hierarchy
- **Header Guards:** Every header file (`.hpp` / `.h`) MUST begin with `#pragma once`.
- **Inclusion Rules:**
  1. Standard C++ headers (alphabetical order).
  2. Third-party dependency headers (e.g., `gtest/gtest.h`).
  3. Microtensor library headers using quote paths relative to include root (e.g., `#include "microtensor/tensor.hpp"`).

### Formatting Rules
- Formatting strictly follows Google C++ Style via `.clang-format` (`BasedOnStyle: Google`).
- 2-space indent, 80-column line limit recommendation, open braces on the same line for functions and control structures.

## 5. Karpathy Operating Rules for Agents

- **Rule 1 (Directness):** Implement fixes directly without adding unnecessary wrapper classes, abstraction layers, or heavy template metaprogramming.
- **Rule 2 (Loop Verification):** Compile (`cmake -S . -B build -G Ninja -DENABLE_TESTING=ON && ninja -C build`) and run test suites (`ctest --test-dir build --output-on-failure`) after every atomic code change.
- **Rule 3 (Zero Assumptions):** Always inspect exact header definitions (`.hpp`) before using types, functions, or class members. Verify metadata assumptions against real data buffers.
- **Rule 4 (Read First):** Fully trace header/source pairs completely (e.g., `tensor.hpp`, `tensor_iterator.hpp`, `cpu_kernels.cpp`) before modifying or extending implementation logic.
- **Rule 5 (Simplicity):** Keep C++ constructs clear and raw. Use standard C++26 language features (`std::span`, `std::views::zip`, concepts, RAII) while avoiding obscure custom metaprogramming constructs.
- **Rule 6 (Style Alignment):** Match established project conventions: use `snake_case_` for member variables, `#pragma once` header guards, `BasedOnStyle: Google` formatting, and shared pointer view semantics for tensor allocations.
