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
// RELU: included as a comparison option -- lets us benchmark sigmoid
//   vs ReLU as a differentiating result to present.
//
// SOFTMAX: used for the OUTPUT layer only, since we need a
//   probability distribution summing to 1.0 across all 10 classes.
//
// OutputSoftmax: a special marker for the output layer. During
//   backward(), the cross-entropy + softmax gradients simplify
//   together to just (predicted - one_hot), so the activation
//   derivative step is skipped (returns identity = all 1.0s).
//   This avoids double-applying the softmax derivative.
// =====================================================================

use crate::matrix::Matrix;

/// Which activation function a layer uses.
/// OutputSoftmax is specifically for the final layer -- it signals
/// that loss.rs already folded the softmax derivative into the
/// cross-entropy gradient, so backward() should not apply it again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    OutputSoftmax,
}

// -----------------------------------------------------------------
// SIGMOID
// -----------------------------------------------------------------

/// Applies sigmoid element-wise to a Matrix.
/// sigmoid(x) = 1 / (1 + e^(-x))
/// Output range: (0.0, 1.0) -- never exactly 0 or 1.
pub fn sigmoid(input: &Matrix) -> Matrix {
    let data: Vec<f64> = input.data.iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

/// Derivative of sigmoid, computed from the sigmoid OUTPUT.
/// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
///
/// IMPORTANT: pass the CACHED OUTPUT here (not the original input)
/// -- layer.rs caches output during forward() so backward() can
/// reuse it here without recomputing exp() again.
pub fn sigmoid_derivative(sigmoid_output: &Matrix) -> Matrix {
    let data: Vec<f64> = sigmoid_output.data.iter()
        .map(|s| s * (1.0 - s))
        .collect();
    Matrix::from_vec(sigmoid_output.rows, sigmoid_output.cols, data)
}

// -----------------------------------------------------------------
// RELU
// -----------------------------------------------------------------

/// ReLU(x) = max(0, x) -- passes positive values, zeroes negatives.
pub fn relu(input: &Matrix) -> Matrix {
    let data: Vec<f64> = input.data.iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

/// ReLU derivative: 1 where original input was positive, 0 otherwise.
/// IMPORTANT: pass the ORIGINAL INPUT here (not sigmoid output) --
/// unlike sigmoid, ReLU's derivative needs the pre-activation value.
pub fn relu_derivative(original_input: &Matrix) -> Matrix {
    let data: Vec<f64> = original_input.data.iter()
        .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Matrix::from_vec(original_input.rows, original_input.cols, data)
}

// -----------------------------------------------------------------
// SOFTMAX (output layer only)
// -----------------------------------------------------------------

/// Converts raw scores into a probability distribution summing to 1.0.
/// Uses "subtract max" trick for numerical stability -- prevents
/// e^x overflowing to infinity when x is large. Mathematically
/// identical to the naive formula, just numerically safer.
pub fn softmax(input: &Matrix) -> Matrix {
    let max_val = input.data.iter()
        .fold(f64::MIN, |acc, &x| if x > acc { x } else { acc });

    let exps: Vec<f64> = input.data.iter()
        .map(|x| (x - max_val).exp())
        .collect();

    let sum: f64 = exps.iter().sum();

    let data: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    Matrix::from_vec(input.rows, input.cols, data)
}

// ----------------------------------------------------------------
// DISPATCH: given an ActivationType, apply the right function.
// layer.rs calls these instead of calling sigmoid/relu/softmax
// directly -- Dense doesn't need to know which formula is used.
// -----------------------------------------------------------------

pub fn apply(activation: ActivationType, input: &Matrix) -> Matrix {
    match activation {
        ActivationType::Sigmoid      => sigmoid(input),
        ActivationType::ReLU         => relu(input),
        ActivationType::OutputSoftmax => softmax(input),
    }
}

/// Returns the activation derivative for backward().
///
/// OutputSoftmax returns a matrix of 1.0s (identity) -- because
/// loss.rs's cross_entropy_derivative() already incorporates the
/// softmax derivative via the (predicted - one_hot) simplification.
/// Multiplying by 1.0 passes the gradient through unchanged.
///
/// For Sigmoid: pass the CACHED OUTPUT.
/// For ReLU:    pass the ORIGINAL INPUT (before activation).
/// For OutputSoftmax: pass anything -- result is always 1.0s.
pub fn apply_derivative(activation: ActivationType, cached_value: &Matrix) -> Matrix {
    match activation {
        ActivationType::Sigmoid => sigmoid_derivative(cached_value),
        ActivationType::ReLU    => relu_derivative(cached_value),
        ActivationType::OutputSoftmax => {
            // Identity: 1.0 everywhere -- gradient passes through
            // unchanged since derivative was already applied in loss.rs.
            let data = vec![1.0; cached_value.rows * cached_value.cols];
            Matrix::from_vec(cached_value.rows, cached_value.cols, data)
        }
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
        let m = Matrix::from_vec(1, 3, vec![1000.0, 1001.0, 1002.0]);
        let result = softmax(&m);
        for &v in result.data.iter() {
            assert!(v.is_finite(),
                "softmax produced non-finite value: {}", v);
        }
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_highest_input_gets_highest_probability() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 5.0, 2.0]);
        let result = softmax(&m);
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

    #[test]
    fn test_apply_dispatch_output_softmax() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let result = apply(ActivationType::OutputSoftmax, &m);
        let sum: f64 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_output_softmax_derivative_is_identity() {
        // OutputSoftmax derivative should return all 1.0s --
        // gradient passes through unchanged.
        let cached = Matrix::from_vec(1, 3, vec![0.2, 0.5, 0.3]);
        let deriv = apply_derivative(ActivationType::OutputSoftmax, &cached);
        assert_eq!(deriv.data, vec![1.0, 1.0, 1.0]);
    }
}