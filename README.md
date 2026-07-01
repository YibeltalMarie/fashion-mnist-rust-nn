# Fashion-MNIST Neural Network — Pure Rust, Zero Dependencies

A fully-connected neural network built **entirely from scratch in Rust**, using only the standard library — no `ndarray`, no `rand`, no `serde`, no ML framework of any kind. Every matrix operation, the PRNG, backpropagation, optimizers, and the training loop are hand-implemented.

This project was built for a competition evaluated on **Accuracy**, **Complexity**, and **Neatness**.

---

## Highlights

- **Zero external crates** — matrix math, random number generation, backpropagation, and optimizers are all implemented from first principles using only `std`.
- **235,146 trainable parameters** across 3 fully-connected layers.
- **54,000 training images / 6,000 validation images / 10,000 held-out test images** (Fashion-MNIST, 28×28 grayscale, 10 classes).
- **Two optimizers implemented from scratch**: SGD with momentum, and Adam (with bias correction).
- **Mathematically verified backpropagation** via a numerical gradient-check test suite (finite differences vs. analytical gradients, relative error < 1e-4).
- **~10x training speedup** achieved by batching samples into single matrix operations and parallelizing matrix multiplication across CPU cores using `std::thread`.
- **Dropout regularization** (inverted dropout, 15% rate) and **learning-rate decay** to improve generalization and convergence.
- **103 automated tests** (100 unit tests + 3 integration tests), all passing — including a dedicated numerical gradient-check suite.
- **Custom binary-free weight serialization format** for saving/loading trained models to disk.
- **Interactive terminal demo** — draw a garment shape on an ASCII grid and get a live prediction from the trained model, without retraining.

---

## Why Fashion-MNIST

Fashion-MNIST was chosen over the classic digit-MNIST dataset specifically because it is **harder for a plain feedforward network**: clothing categories (shirt vs. pullover vs. coat) share overlapping silhouettes and textures that a non-convolutional architecture cannot fully separate. This was a deliberate choice to demonstrate a stronger, more honest benchmark rather than an easier dataset that would inflate the reported accuracy.

---

## Architecture

```
Input (784)                         # 28x28 flattened, normalized to [0, 1]
  │
  ▼
Dense (784 → 256)  + Sigmoid
  │
  ▼
Dropout (rate = 0.15)               # active during training only
  │
  ▼
Dense (256 → 128)  + Sigmoid
  │
  ▼
Dropout (rate = 0.15)
  │
  ▼
Dense (128 → 10)   + Softmax        # output layer, probability distribution
```

### Parameter count (complexity)

| Layer                  | Weights           | Biases | Subtotal    |
|-------------------------|-------------------|--------|-------------|
| Dense 1 (784 → 256)      | 784 × 256 = 200,704 | 256    | 200,960     |
| Dense 2 (256 → 128)      | 256 × 128 = 32,768  | 128    | 32,896      |
| Dense 3 (128 → 10)       | 128 × 10 = 1,280    | 10     | 1,290       |
| **Total**                |                    |        | **235,146** |

All 235,146 parameters are trained from scratch on CPU, using hand-written matrix multiplication — no BLAS, no linear algebra library.

---

## Dataset

| Split       | Samples | Source                          | Used for                                |
|-------------|---------|----------------------------------|------------------------------------------|
| Train       | 54,000  | `fashion-mnist_train.csv` (90%)  | Weight updates (forward + backward pass) |
| Validation  | 6,000   | `fashion-mnist_train.csv` (10%)  | Monitoring generalization during training, never used to update weights |
| Test        | 10,000  | `fashion-mnist_test.csv`         | Final, single-use accuracy report        |

Each row is `label, pixel_1, pixel_2, ..., pixel_784` — parsed by hand (no CSV crate) and normalized by dividing every pixel by 255.0.

The test set is touched **exactly once**, at the very end, after all training and hyperparameter decisions are finalized — ensuring the reported accuracy is unbiased.

---

## Results

> Fill in with your final run's numbers from `results.log` before submitting.

| Metric                     | Value       |
|------------------------------|-------------|
| Final validation accuracy    | `__.__ %`   |
| Final test accuracy          | `__.__ %`   |
| Training time (30 epochs)    | `__m __s`   |
| Optimizer                    | Adam        |
| Learning rate                | 0.01 (decays ×0.5 every 10 epochs) |
| Batch size                   | 128         |

A full confusion matrix and per-class accuracy breakdown are printed automatically at the end of training (see `metrics.rs`), showing exactly which garment categories the model tends to confuse — a known limitation of feedforward (non-convolutional) architectures on this dataset.

---

## Engineering features implemented

### Core (from-scratch implementations)
- **`matrix.rs`** — a flat-array-backed `Matrix` type (row-major `Vec<f64>`) supporting matrix multiplication, transpose, element-wise operations, and scalar operations. Matrix multiplication is parallelized across CPU cores using `std::thread`.
- **`rng.rs`** — a hand-rolled `xorshift64` pseudo-random number generator with a Box–Muller transform for Gaussian sampling, used for weight initialization and dropout masks. Fully deterministic given a seed (reproducible runs).
- **`init.rs`** — centralized weight initialization dispatched by activation type (Xavier/Glorot for Sigmoid and Softmax, He initialization for ReLU).
- **`activation.rs`** — Sigmoid, ReLU, and numerically-stable Softmax (max-subtraction trick to prevent overflow), plus their derivatives.
- **`layer.rs`** — a `Layer` trait implemented by:
  - `Dense` — fully-connected layer with forward/backward passes and batched matrix operations
  - `Dropout` — inverted dropout with independent training/inference behavior
- **`loss.rs`** — cross-entropy loss, paired with Softmax so their combined gradient simplifies to `predicted − actual` (avoiding unnecessary derivative chaining).
- **`optimizer.rs`** — an `Optimizer` trait implemented by:
  - `SGD` — with configurable momentum
  - `Adam` — full implementation with first/second moment estimates and bias correction
- **`network.rs`** — orchestrates the full training loop: batching, Fisher–Yates shuffling (hand-implemented, no `rand` crate), forward/backward passes, gradient averaging, optimizer steps, dropout mode switching, and learning-rate decay.
- **`data.rs`** — manual CSV parsing and train/validation splitting.
- **`metrics.rs`** — confusion matrix construction and per-class accuracy reporting.
- **`io_utils.rs`** — custom plain-text weight serialization format (no `serde`/`bincode`) with full round-trip and architecture-mismatch validation.
- **`demo.rs`** — interactive terminal demo: draw a shape on an ASCII grid, get a live prediction from the saved model.

### Correctness & rigor
- **Numerical gradient checking** (`tests/gradient_check.rs`): every analytical gradient produced by backpropagation is cross-checked against an independently computed finite-difference gradient, across multiple layers, weights, biases, and input samples. This mathematically proves the backward pass is implemented correctly, rather than relying solely on "the accuracy went up."
- **100 unit tests + 3 integration tests (103 total, all passing)** covering matrix operations (including parallel vs. sequential matmul equivalence), RNG determinism, activation functions, layer forward/backward correctness, loss computation, optimizer behavior (SGD momentum, Adam bias correction), dropout train/inference modes, save/load round-trips, confusion matrix logic, and the interactive demo's grid-to-input conversion.

### Performance
- Initial (naive, single-sample) training took **~70 minutes for 5 epochs**. After introducing **batched matrix operations** (whole batches processed as a single matrix multiplication instead of per-sample loops) and **parallelizing matrix multiplication across CPU cores** with `std::thread`, training time dropped to **~7 minutes for 5 epochs** — roughly a **10x speedup**, achieved entirely with the standard library.

---

## Project structure

```
fashion_mnist_rust/
├── Cargo.toml
├── README.md
├── results.log                 # append-only log of every training run
├── data/
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
├── src/
│   ├── main.rs                 # entry point — train mode & demo mode
│   ├── lib.rs                  # library root (enables integration tests)
│   ├── matrix.rs
│   ├── rng.rs
│   ├── init.rs
│   ├── activation.rs
│   ├── layer.rs
│   ├── loss.rs
│   ├── optimizer.rs
│   ├── network.rs
│   ├── data.rs
│   ├── metrics.rs
│   ├── io_utils.rs
│   └── demo.rs
└── tests/
    └── gradient_check.rs        # integration test: numerical gradient check
```

---

## Fashion-MNIST classes

| Index | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Status

- [x] Matrix struct (flat Vec<f64>, parallel matmul)
- [x] Hand-rolled PRNG (xorshift64 + Box-Muller)
- [x] CSV data loader + normalization + train/val split
- [x] Dense layer (forward + backward, batched)
- [x] Dropout regularization (inverted, train/inference toggle)
- [x] Sigmoid, ReLU, Softmax activations + derivatives
- [x] Cross-entropy loss + batched gradient
- [x] SGD (momentum) + Adam optimizers
- [x] Mini-batch training with Fisher-Yates shuffle
- [x] Learning rate decay (step schedule)
- [x] Gradient check (numerical vs analytical, < 1e-4 relative error)
- [x] Confusion matrix + per-class accuracy
- [x] Save/load trained weights (custom plain-text format)
- [x] Live ASCII training curve
- [x] Interactive terminal draw-and-predict demo
- [x] Full test suite (unit + integration)

## Usage

### Train the network
```bash
cargo run --release -- data/fashion-mnist_train.csv data/fashion-mnist_test.csv
```
Trains for 30 epochs, prints a live ASCII progress bar with loss/accuracy per epoch, evaluates on the held-out test set exactly once, saves the trained model to `trained.weights`, and appends the run's results to `results.log`.

### Run the interactive demo (no retraining required)
```bash
cargo run --release -- demo
```
Loads `trained.weights` and launches a terminal grid where you can sketch a garment shape and get a live prediction with full class probabilities.

### Run the test suite
```bash
cargo test --release
```
Runs all 100 unit tests plus the 3 numerical gradient-check integration tests (103 total).

---

## Design notes

- **Why pure `std`, no crates**: the assignment specifically required implementing every component — including matrix math, randomness, and serialization — without relying on existing libraries, to demonstrate genuine understanding of the underlying mechanics rather than API usage.
- **Why Sigmoid for hidden layers**: chosen to align with the curriculum's focus on sigmoid activations, with the known trade-off (vanishing gradients in deep networks) mitigated by keeping the network at two hidden layers and using Xavier initialization tuned specifically for sigmoid.
- **Why batching + threading**: a naive single-sample training loop was correct but too slow for iterative experimentation within the project timeline. Batching and parallel matrix multiplication were added as a deliberate performance engineering step, not just a correctness fix — a concrete demonstration of Rust's suitability for numerically intensive workloads.