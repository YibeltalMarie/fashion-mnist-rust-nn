// TODO (Day 2): Cross-entropy loss + derivative (simplifies to predicted - actual
// when paired with softmax).

// =====================================================================
// loss.rs
//
// Cross-entropy loss for multi-class classification.
// Works together with softmax output (activation.rs) as a paired
// system -- their combined gradient simplifies to just:
//   dL/dz = predicted - one_hot(actual_label)
// which is what we return from `cross_entropy_derivative()` and
// pass into the last layer's backward() call.
//
// RESPONSIBILITY BOUNDARY:
// This file computes the SCALAR loss (for logging/monitoring) and
// its GRADIENT (to start the backward pass). It does NOT call
// backward() or know anything about layers -- that's network.rs.
// =====================================================================

use crate::matrix::Matrix;

/// Computes cross-entropy loss for ONE sample.
///
/// `predicted` -- softmax output, shape (10 x 1), values sum to 1.0
/// `actual`    -- the correct class label (0-9), as usize
///
/// Formula: loss = -log(predicted[actual])
/// We only care about the probability assigned to the correct class.
pub fn cross_entropy(predicted: &Matrix, actual: usize) -> f64 {
    // Clip to 1e-15 to prevent log(0) = -infinity, which would
    // break the backward pass entirely.
    let p = predicted.get(actual, 0).max(1e-15);
    -p.ln() // .ln() is Rust's natural log (log base e)
}

/// Average cross-entropy loss over a batch.
///
/// `predictions` -- Vec of softmax outputs, each shape (10 x 1)
/// `actuals`     -- Vec of correct class labels (usize), same length
pub fn batch_cross_entropy(predictions: &[Matrix], actuals: &[usize]) -> f64 {
    assert_eq!(
        predictions.len(), actuals.len(),
        "batch_cross_entropy: predictions and actuals must have the same length"
    );

    let total: f64 = predictions.iter()
        .zip(actuals.iter())
        .map(|(pred, &actual)| cross_entropy(pred, actual))
        .sum();

    // Average over the batch.
    total / predictions.len() as f64
}

/// Gradient of cross-entropy + softmax combined, for ONE sample.
///
/// Because cross-entropy and softmax are paired, their combined
/// gradient simplifies to: predicted - one_hot(actual_label)
///
/// This is the STARTING gradient for the backward pass -- it gets
/// passed into the last layer's backward() as `output_grad`.
///
/// `predicted` -- softmax output, shape (10 x 1)
/// `actual`    -- correct class label (0-9)
/// `n_classes` -- number of output classes (10 for Fashion-MNIST)
pub fn cross_entropy_derivative(predicted: &Matrix, actual: usize, n_classes: usize) -> Matrix {
    // Build a copy of predicted, then subtract 1.0 at the correct
    // class position -- that's all "predicted - one_hot(actual)" means.
    let mut grad = predicted.clone();
    let current = grad.get(actual, 0);
    grad.set(actual, 0, current - 1.0);
    grad
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_prediction_gives_near_zero_loss() {
        // If the network assigns 100% probability to the correct class,
        // loss should be -log(1.0) = 0.0.
        let mut predicted = Matrix::zeros(10, 1);
        predicted.set(3, 0, 1.0); // 100% confident in class 3
        let loss = cross_entropy(&predicted, 3);
        assert!(loss < 1e-9, "perfect prediction should give ~0 loss, got {}", loss);
    }

    #[test]
    fn test_wrong_prediction_gives_high_loss() {
        // If the network assigns near-zero probability to the correct
        // class, loss should be very large.
        let mut predicted = Matrix::zeros(10, 1);
        predicted.set(7, 0, 0.999); // confident in WRONG class
        predicted.set(3, 0, 0.001); // almost zero probability for correct class
        let loss = cross_entropy(&predicted, 3); // correct class is 3
        assert!(loss > 4.0, "confident wrong prediction should give high loss, got {}", loss);
    }

    #[test]
    fn test_zero_probability_does_not_produce_infinity() {
        // Without the 1e-15 clip, log(0) = -infinity.
        // Confirms the clip prevents this.
        let predicted = Matrix::zeros(10, 1); // all zeros, including class 3
        let loss = cross_entropy(&predicted, 3);
        assert!(loss.is_finite(), "loss should be finite even with zero probability");
    }

    #[test]
    fn test_batch_loss_is_average() {
        // Manually compute expected average and compare.
        let mut p1 = Matrix::zeros(10, 1);
        p1.set(2, 0, 1.0); // perfect for class 2 -> loss = 0

        let mut p2 = Matrix::zeros(10, 1);
        p2.set(5, 0, 0.5); // 50% for correct class 5 -> loss = -log(0.5) ≈ 0.693

        let predictions = vec![p1, p2];
        let actuals = vec![2, 5];

        let batch_loss = batch_cross_entropy(&predictions, &actuals);
        let expected = (0.0 + (-0.5_f64.ln())) / 2.0;
        assert!((batch_loss - expected).abs() < 1e-9);
    }

    #[test]
    fn test_derivative_subtracts_one_at_correct_class() {
        // For correct class 3, gradient = predicted - one_hot(3)
        // meaning only position 3 should change (by -1.0).
        let mut predicted = Matrix::zeros(10, 1);
        predicted.set(3, 0, 0.7);  // predicted prob for class 3
        predicted.set(5, 0, 0.3);  // predicted prob for class 5

        let grad = cross_entropy_derivative(&predicted, 3, 10);

        // Position 3: 0.7 - 1.0 = -0.3
        assert!((grad.get(3, 0) - (-0.3)).abs() < 1e-9);
        // Position 5: unchanged (0.3 - 0.0 = 0.3)
        assert!((grad.get(5, 0) - 0.3).abs() < 1e-9);
        // All other positions: still 0.0
        for i in 0..10 {
            if i != 3 && i != 5 {
                assert_eq!(grad.get(i, 0), 0.0);
            }
        }
    }

    #[test]
    fn test_derivative_shape_matches_predicted() {
        let predicted = Matrix::zeros(10, 1);
        let grad = cross_entropy_derivative(&predicted, 0, 10);
        assert_eq!(grad.rows, 10);
        assert_eq!(grad.cols, 1);
    }
}