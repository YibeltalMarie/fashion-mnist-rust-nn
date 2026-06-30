// =====================================================================
// init.rs
//
// Centralizes weight initialization strategy, dispatched by
// ActivationType. Keeps layer.rs free of "which formula applies to
// which activation" logic, and keeps rng.rs a pure, activation-
// agnostic random number generator.
//
// WHY DIFFERENT ACTIVATIONS NEED DIFFERENT INIT:
// Both Xavier and He init aim to keep the VARIANCE of activations
// roughly stable as data flows through layers -- preventing values
// from exploding or vanishing as the network gets deeper. The exact
// formula differs because sigmoid and ReLU have different output
// statistics.
//
//   Xavier/Glorot (sigmoid, tanh): std_dev = sqrt(1 / n_inputs)
//   He            (ReLU):          std_dev = sqrt(2 / n_inputs)
//     -- the extra factor of 2 compensates for ReLU zeroing out
//        roughly half its inputs (everything negative).
// =====================================================================

use crate::matrix::Matrix;
use crate::activation::ActivationType;
use crate::rng::Rng;

/// Returns the correct standard deviation for Gaussian weight
/// initialization, given which activation function this layer uses.
pub fn std_dev_for(activation: ActivationType, n_inputs: usize) -> f64 {
    match activation {
        ActivationType::Sigmoid => (1.0 / n_inputs as f64).sqrt(),
        ActivationType::ReLU => (2.0 / n_inputs as f64).sqrt(),
    }
}

/// Builds a (rows x cols) weight Matrix using the correct
/// initialization scheme for the given activation.
/// Convention: rows = output_size, cols = input_size, so n_inputs = cols.
pub fn init_weights(rows: usize, cols: usize, activation: ActivationType, rng: &mut Rng) -> Matrix {
    let std_dev = std_dev_for(activation, cols);
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| rng.gaussian_scaled(std_dev))
        .collect();
    Matrix::from_vec(rows, cols, data)
}

/// Biases always start at zero -- standard practice, true regardless
/// of activation choice. Centralized here too for consistency: every
/// layer's initialization, of every kind, goes through this one file.
pub fn init_biases(size: usize) -> Matrix {
    Matrix::zeros(size, 1)
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_std_dev_formula() {
        // n_inputs = 4 -> sqrt(1/4) = 0.5
        let std_dev = std_dev_for(ActivationType::Sigmoid, 4);
        assert!((std_dev - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_relu_std_dev_formula() {
        // n_inputs = 2 -> sqrt(2/2) = 1.0
        let std_dev = std_dev_for(ActivationType::ReLU, 2);
        assert!((std_dev - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_relu_std_dev_larger_than_sigmoid_for_same_inputs() {
        // He should always produce a larger std_dev than Xavier,
        // for the same n_inputs (the factor-of-2 difference).
        let n = 100;
        let xavier = std_dev_for(ActivationType::Sigmoid, n);
        let he = std_dev_for(ActivationType::ReLU, n);
        assert!(he > xavier);
    }

    #[test]
    fn test_init_weights_correct_shape() {
        let mut rng = Rng::new(1);
        let w = init_weights(3, 4, ActivationType::Sigmoid, &mut rng);
        assert_eq!(w.rows, 3);
        assert_eq!(w.cols, 4);
    }

    #[test]
    fn test_init_weights_not_all_zero() {
        let mut rng = Rng::new(2);
        let w = init_weights(10, 10, ActivationType::Sigmoid, &mut rng);
        let nonzero_count = w.data.iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "weights should not all be zero");
    }

    #[test]
    fn test_init_biases_all_zero() {
        let b = init_biases(5);
        assert_eq!(b.data, vec![0.0; 5]);
    }

    #[test]
    fn test_same_seed_gives_same_init() {
        // Determinism: same seed -> identical initial weights.
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let w1 = init_weights(5, 5, ActivationType::ReLU, &mut rng1);
        let w2 = init_weights(5, 5, ActivationType::ReLU, &mut rng2);
        assert_eq!(w1.data, w2.data);
    }
}