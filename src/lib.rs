// =====================================================================
// lib.rs
//
// Makes this project accessible as a library crate in addition to
// being a binary (main.rs). Required so integration tests in tests/
// can import modules via `use fashion_mnist_rust::...`.
//
// Every module listed here with `pub mod` becomes accessible from
// outside the crate. main.rs continues to work unchanged -- it
// declares the same modules independently as a binary entry point.
// =====================================================================

pub mod matrix;
pub mod rng;
pub mod init;
pub mod activation;
pub mod layer;
pub mod loss;
pub mod optimizer;
pub mod network;
pub mod data;
pub mod metrics;
pub mod io_utils;