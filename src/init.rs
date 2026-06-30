// =====================================================================
// init.rs
//
// Centralizes weight initialization strategy, dispatched by
// ActivationType. Keeps layer.rs free of "which formula applies to
// which activation" logic.
//
// Xavier/Glorot (Sigmoid, OutputSoftmax): std_dev = sqrt(1 / n_inputs)
// He                (ReLU):               std_dev = sqrt(2 / n_inputs)
//
// OutputSoftmax uses Xavier because the linear combination before
// softmax has the same variance properties as before sigmoid.
// =====================================================================

use crate::matrix::Matrix;
use crate::activation::ActivationType;
use crate::rng::Rng;

/// Returns the correct standard deviation for weight initialization
/// based on which activation function this layer uses.
pub fn std_dev_for(activation: ActivationType, n_inputs: usize) -> f64 {
    match activation {
        ActivationType::Sigmoid       => (1.0 / n_inputs as f64).sqrt(),
        ActivationType::ReLU          => (2.0 / n_inputs as f64).sqrt(),
        ActivationType::OutputSoftmax => (1.0 / n_inputs as f64).sqrt(), // Xavier
    }
}

/// Builds a (rows x cols) weight Matrix using the correct
/// initialization scheme for the given activation.
/// rows = output_size, cols = input_size, so n_inputs = cols.
pub fn init_weights(rows: usize, cols: usize, activation: ActivationType, rng: &mut Rng) -> Matrix {
    let std_dev = std_dev_for(activation, cols);
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| rng.gaussian_scaled(std_dev))
        .collect();
    Matrix::from_vec(rows, cols, data)
}

/// Biases always start at zero, regardless of activation choice.
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
    fn test_output_softmax_uses_xavier() {
        // OutputSoftmax should use same formula as Sigmoid (Xavier).
        let n = 100;
        let sigmoid_std = std_dev_for(ActivationType::Sigmoid, n);
        let softmax_std = std_dev_for(ActivationType::OutputSoftmax, n);
        assert!((sigmoid_std - softmax_std).abs() < 1e-9,
            "OutputSoftmax should use same std_dev as Sigmoid (Xavier)");
    }

    #[test]
    fn test_relu_std_dev_larger_than_sigmoid() {
        // He should always produce larger std_dev than Xavier.
        let n = 100;
        let xavier = std_dev_for(ActivationType::Sigmoid, n);
        let he     = std_dev_for(ActivationType::ReLU, n);
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
        let nonzero = w.data.iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero > 0, "weights should not all be zero");
    }

    #[test]
    fn test_init_biases_all_zero() {
        let b = init_biases(5);
        assert_eq!(b.data, vec![0.0; 5]);
    }

    #[test]
    fn test_same_seed_gives_same_init() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let w1 = init_weights(5, 5, ActivationType::ReLU, &mut rng1);
        let w2 = init_weights(5, 5, ActivationType::ReLU, &mut rng2);
        assert_eq!(w1.data, w2.data);
    }

    #[test]
    fn test_init_weights_output_softmax_not_all_zero() {
        let mut rng = Rng::new(7);
        let w = init_weights(10, 64, ActivationType::OutputSoftmax, &mut rng);
        let nonzero = w.data.iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero > 0);
    }
}