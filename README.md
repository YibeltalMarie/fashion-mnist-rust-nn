# Fashion-MNIST Neural Network — Pure Rust (No External Crates)

A feedforward neural network built entirely from scratch in Rust's standard
library — no ndarray, no rand, no serde, no ML frameworks of any kind.

## Why this project
- Dataset: Fashion-MNIST (60,000 train / 10,000 test, 28x28 grayscale, 10 classes)
- Goal: demonstrate real understanding of NN internals (matrix math, backprop,
  optimizers) rather than relying on library abstractions.

## Architecture
(to be filled in as built — see src/ for module breakdown)

## Results
(to be filled in Day 4-5: accuracy, confusion matrix, SGD vs Adam comparison,
training time benchmarks)

## Status
- [ ] Matrix struct
- [ ] PRNG + init
- [ ] Data loader
- [ ] Dense layer + forward/backward
- [ ] Training loop
- [ ] Adam optimizer
- [ ] Gradient check test
- [ ] Confusion matrix
- [ ] Save/load weights
- [ ] Terminal demo
