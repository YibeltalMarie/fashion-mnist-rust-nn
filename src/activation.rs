// TODO (Day 2): Sigmoid, ReLU, Softmax + derivatives. Pure functions over Matrix/Vec<f64>.

// =====================================================================
// activation.rs
//
// Activation functions inject non-linearity into the network --
// without them, stacking layers would mathematically collapse into
// one big linear function, and the network couldn't learn complex
// patterns.
//
// SIGMOID: used for hidden layers (per mentor's curriculum).
//   sigmoid(x) = 1 / (1 + e^(-x))   -- squashes to (0, 1)
//
// SOFTMAX: used for the OUTPUT layer regardless of hidden-layer choice,
// since we need a probability distribution over 10 classes.
//
// RELU: included as a comparison option (not mentor-required, but lets
// us benchmark sigmoid vs ReLU as an extra result to present).
// =====================================================================

use crate::matrix::Matrix;

/// Which activation function a layer should use. An enum represents
/// "exactly one of these named options" -- similar to Python's Enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
}

// -----------------------------------------------------------------
// SIGMOID
// -----------------------------------------------------------------

/// Applies sigmoid to every element of a Matrix, returning a new Matrix.
/// x.exp() is a built-in f64 method computing e^x.
pub fn sigmoid(input: &Matrix) -> Matrix {
    let data: Vec<f64> = input.data.iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

/// Derivative of sigmoid, computed from the sigmoid OUTPUT (not the
/// original input) -- this is why we cache forward-pass outputs in
/// layer.rs, so backward doesn't need to recompute exp() again.
pub fn sigmoid_derivative(sigmoid_output: &Matrix) -> Matrix {
    let data: Vec<f64> = sigmoid_output.data.iter()
        .map(|s| s * (1.0 - s))
        .collect();
    Matrix::from_vec(sigmoid_output.rows, sigmoid_output.cols, data)
}

// -----------------------------------------------------------------
// RELU (comparison option)
// -----------------------------------------------------------------

pub fn relu(input: &Matrix) -> Matrix {
    let data: Vec<f64> = input.data.iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

/// ReLU's derivative is 1 where the ORIGINAL INPUT was positive, 0
/// otherwise -- note this needs the original input, unlike sigmoid
/// which uses its own output.
pub fn relu_derivative(original_input: &Matrix) -> Matrix {
    let data: Vec<f64> = original_input.data.iter()
        .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Matrix::from_vec(original_input.rows, original_input.cols, data)
}

// -----------------------------------------------------------------
// SOFTMAX (output layer only)
//
// Includes the "subtract max" numerical stability trick: prevents
// e^x from overflowing when x is large, without changing the result.
// -----------------------------------------------------------------

pub fn softmax(input: &Matrix) -> Matrix {
    // Find the maximum value in the input. iter() borrows each element,
    // fold(...) walks through accumulating a running result -- here,
    // the running maximum so far.
    let max_val = input.data.iter().fold(f64::MIN, |acc, &x| if x > acc { x } else { acc });

    // Subtract max, then exponentiate each element.
    let exps: Vec<f64> = input.data.iter()
        .map(|x| (x - max_val).exp())
        .collect();

    // Sum all the exponentiated values, to use as the normalizing
    // denominator.
    let sum: f64 = exps.iter().sum();

    let data: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

// -----------------------------------------------------------------
// DISPATCH HELPERS
//
// Given an ActivationType, apply the right function. This is where
// the enum becomes useful -- layer.rs can just store an
// ActivationType and call these, without needing its own if/else.
// -----------------------------------------------------------------

pub fn apply(activation: ActivationType, input: &Matrix) -> Matrix {
    match activation {
        ActivationType::Sigmoid => sigmoid(input),
        ActivationType::ReLU => relu(input),
    }
}

/// NOTE: for Sigmoid, pass the CACHED OUTPUT here.
/// For ReLU, pass the ORIGINAL INPUT here.
/// (layer.rs will be written to respect this distinction.)
pub fn apply_derivative(activation: ActivationType, cached_value: &Matrix) -> Matrix {
    match activation {
        ActivationType::Sigmoid => sigmoid_derivative(cached_value),
        ActivationType::ReLU => relu_derivative(cached_value),
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_at_zero_is_half() {
        let m = Matrix::from_vec(1, 1, vec![0.0]);
        let result = sigmoid(&m);
        assert!((result.get(0, 0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_sigmoid_large_positive_approaches_one() {
        let m = Matrix::from_vec(1, 1, vec![20.0]);
        let result = sigmoid(&m);
        assert!(result.get(0, 0) > 0.999);
    }

    #[test]
    fn test_sigmoid_large_negative_approaches_zero() {
        let m = Matrix::from_vec(1, 1, vec![-20.0]);
        let result = sigmoid(&m);
        assert!(result.get(0, 0) < 0.001);
    }

    #[test]
    fn test_sigmoid_derivative_at_half() {
        // sigmoid output of 0.5 -> derivative = 0.5 * (1 - 0.5) = 0.25
        let output = Matrix::from_vec(1, 1, vec![0.5]);
        let deriv = sigmoid_derivative(&output);
        assert!((deriv.get(0, 0) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_relu_negative_becomes_zero() {
        let m = Matrix::from_vec(1, 3, vec![-5.0, 0.0, 5.0]);
        let result = relu(&m);
        assert_eq!(result.data, vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_relu_derivative() {
        let m = Matrix::from_vec(1, 3, vec![-5.0, 0.0, 5.0]);
        let deriv = relu_derivative(&m);
        assert_eq!(deriv.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let m = Matrix::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let result = softmax(&m);
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_handles_large_values_without_overflow() {
        // Without the "subtract max" trick, this would overflow to inf/NaN.
        let m = Matrix::from_vec(1, 3, vec![1000.0, 1001.0, 1002.0]);
        let result = softmax(&m);
        for &v in result.data.iter() {
            assert!(v.is_finite(), "softmax produced a non-finite value: {}", v);
        }
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_highest_input_gets_highest_probability() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 5.0, 2.0]);
        let result = softmax(&m);
        // index 1 had the highest input (5.0), should have highest probability
        assert!(result.get(0, 1) > result.get(0, 0));
        assert!(result.get(0, 1) > result.get(0, 2));
    }

    #[test]
    fn test_apply_dispatch_sigmoid() {
        let m = Matrix::from_vec(1, 1, vec![0.0]);
        let result = apply(ActivationType::Sigmoid, &m);
        assert!((result.get(0, 0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_apply_dispatch_relu() {
        let m = Matrix::from_vec(1, 1, vec![-3.0]);
        let result = apply(ActivationType::ReLU, &m);
        assert_eq!(result.get(0, 0), 0.0);
    }
}