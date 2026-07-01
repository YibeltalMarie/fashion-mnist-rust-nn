// =====================================================================
// layer.rs
//
// Defines the Layer trait and two implementations:
//   Dense   -- fully-connected layer (weights + biases + activation)
//   Dropout -- regularization layer (randomly zeros neurons during
//              training to prevent overfitting)
//
// BATCHING: forward()/backward() operate on whole batches.
// `input` is (size x batch_size), one column per sample.
//
// RESPONSIBILITY BOUNDARY:
// This file computes forward output and gradients per layer only.
// Weight init -> init.rs. Gradient application -> optimizer.rs.
// Training orchestration -> network.rs.
// =====================================================================

use crate::matrix::Matrix;
use crate::activation::{self, ActivationType};
use crate::rng::Rng;
use crate::init;

// -----------------------------------------------------------------
// THE Layer TRAIT
// -----------------------------------------------------------------
pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_grad: &Matrix) -> Matrix;

    /// Default returns None -- only Dense overrides this.
    fn as_dense_mut(&mut self) -> Option<&mut Dense> {
        None
    }

    fn as_dense(&self) -> Option<&Dense> {
        None
    }

    /// Number of output neurons this layer produces.
    /// Used by network.rs to size gradients dynamically.
    fn output_size(&self) -> usize;

    /// Switch between training mode (dropout active) and inference
    /// mode (dropout disabled). Default is a no-op -- only Dropout
    /// overrides this. network.rs calls set_training(false) before
    /// evaluate() and set_training(true) before training batches.
    fn set_training(&mut self, _training: bool) {}
}

// =====================================================================
// DENSE LAYER
// =====================================================================
pub struct Dense {
    pub weights: Matrix,       // (output_size x input_size)
    pub biases: Matrix,         // (output_size x 1)
    pub activation: ActivationType,

    cached_input: Option<Matrix>,
    cached_output: Option<Matrix>,

    pub weight_grad: Option<Matrix>,
    pub bias_grad: Option<Matrix>,
}

impl Dense {
    pub fn new(
        input_size:  usize,
        output_size: usize,
        activation:  ActivationType,
        rng:         &mut Rng,
    ) -> Self {
        Dense {
            weights:      init::init_weights(output_size, input_size, activation, rng),
            biases:       init::init_biases(output_size),
            activation,
            cached_input:  None,
            cached_output: None,
            weight_grad:   None,
            bias_grad:     None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        // z = W . input + bias (broadcast across batch columns)
        let z = self.weights.matmul_parallel(input)
            .add_bias_broadcast(&self.biases);
        let output = activation::apply(self.activation, &z);

        self.cached_input  = Some(input.clone());
        self.cached_output = Some(output.clone());

        output
    }

    fn backward(&mut self, output_grad: &Matrix) -> Matrix {
        let cached_input = self.cached_input.as_ref()
            .expect("backward() called before forward()");
        let cached_output = self.cached_output.as_ref()
            .expect("backward() called before forward()");

        // Gradient through activation (elementwise, works on batches).
        let activation_deriv = activation::apply_derivative(
            self.activation, cached_output
        );
        let dz = output_grad.hadamard(&activation_deriv);

        // weight_grad = dz . input^T  (sums over batch automatically)
        let weight_grad = dz.matmul_parallel(&cached_input.transpose());

        // bias_grad = sum dz across batch columns -> (output x 1)
        let bias_grad = dz.sum_cols();

        // input_grad = W^T . dz  (passes gradient to previous layer)
        let input_grad = self.weights.transpose().matmul_parallel(&dz);

        self.weight_grad = Some(weight_grad);
        self.bias_grad   = Some(bias_grad);

        input_grad
    }

    fn output_size(&self) -> usize {
        // weights is (output x input), rows = output neurons.
        self.weights.rows
    }

    fn as_dense_mut(&mut self) -> Option<&mut Dense> {
        Some(self)
    }

    fn as_dense(&self) -> Option<&Dense> {
        Some(self)
    }
}

// =====================================================================
// DROPOUT LAYER
//
// Randomly zeros a fraction `rate` of neurons during training.
// Uses "inverted dropout": surviving neurons are scaled UP by
// 1/(1-rate) during training so inference needs no adjustment at all.
//
// FORWARD (training):
//   mask[i] = 1.0 if random > rate, else 0.0
//   output  = input * mask * (1 / (1-rate))
//
// FORWARD (inference):
//   output = input  (pass through unchanged, mask ignored)
//
// BACKWARD:
//   gradient only flows through neurons that were active (mask=1).
//   input_grad = output_grad * mask * (1 / (1-rate))
//   (same mask and scale applied to gradient as was applied to input)
// =====================================================================
pub struct Dropout {
    // Fraction of neurons to zero out (e.g. 0.2 = 20% dropped).
    // Keep rate low for sigmoid networks (0.1-0.2) -- sigmoid already
    // has vanishing gradient issues; high dropout makes it worse.
    rate: f64,

    // true during training (dropout active), false during evaluate().
    // network.rs toggles this via set_training().
    training: bool,

    // Cached during forward(), used by backward() to route gradients.
    // Shape: same as forward input (size x batch_size).
    // None until first forward() call.
    mask: Option<Matrix>,

    // The input size -- needed for output_size() since Dropout doesn't
    // change the shape of its input.
    size: usize,

    // Rng for generating the random mask each forward pass.
    rng: Rng,
}

impl Dropout {
    /// rate: fraction of neurons to drop (0.0-1.0).
    ///   0.0 = no dropout (identity), 1.0 = drop everything (don't use).
    ///   Recommended: 0.1-0.2 for sigmoid networks.
    /// size: number of neurons this layer operates on (= previous
    ///   layer's output_size, so Dropout knows its own output_size).
    /// seed: for the internal Rng (different from the network's main Rng
    ///   so dropout randomness is independent of weight init randomness).
    pub fn new(rate: f64, size: usize, seed: u64) -> Self {
        assert!(
            rate >= 0.0 && rate < 1.0,
            "Dropout rate must be in [0, 1), got {}", rate
        );
        Dropout {
            rate,
            training: true, // default to training mode
            mask: None,
            size,
            rng: Rng::new(seed),
        }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        if !self.training || self.rate == 0.0 {
            // Inference mode or no dropout -- pass through unchanged.
            // We still update `mask` to a all-ones matrix so backward()
            // can be called safely if needed (e.g. in gradient check).
            self.mask = Some(Matrix::from_vec(
                input.rows, input.cols,
                vec![1.0; input.rows * input.cols],
            ));
            return input.clone();
        }

        // Scale factor for inverted dropout -- applied during training
        // so inference (scale=1.0) needs no adjustment.
        let scale = 1.0 / (1.0 - self.rate);

        // Build the random binary mask for this batch.
        // Each element is independently 0.0 (dropped) or scale (kept).
        // Applying scale here (not separately) means ONE hadamard in
        // both forward and backward instead of two operations.
        let mask_data: Vec<f64> = (0..input.rows * input.cols)
            .map(|_| {
                if self.rng.next_f64() > self.rate {
                    scale  // neuron survives, scaled up
                } else {
                    0.0    // neuron dropped
                }
            })
            .collect();

        let mask = Matrix::from_vec(input.rows, input.cols, mask_data);
        let output = input.hadamard(&mask);

        self.mask = Some(mask);
        output
    }

    fn backward(&mut self, output_grad: &Matrix) -> Matrix {
        // Gradient flows only through neurons that were active --
        // apply the same mask (with the same scale baked in) to the
        // incoming gradient. Dropped neurons get zero gradient.
        let mask = self.mask.as_ref()
            .expect("Dropout backward() called before forward()");
        output_grad.hadamard(mask)
    }

    fn output_size(&self) -> usize {
        // Dropout never changes tensor shape -- output = input size.
        self.size
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // DENSE TESTS
    // -----------------------------------------------------------------

    #[test]
    fn test_forward_output_shape_single_sample() {
        let mut rng   = Rng::new(1);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input     = Matrix::zeros(4, 1);
        let output    = layer.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 1);
    }

    #[test]
    fn test_forward_output_shape_batch() {
        let mut rng   = Rng::new(2);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input     = Matrix::zeros(4, 5);
        let output    = layer.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 5);
    }

    #[test]
    fn test_forward_sigmoid_output_in_range() {
        let mut rng   = Rng::new(2);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input     = Matrix::from_vec(4, 2, vec![
            1.0, -2.0, 0.5, 3.0,
            0.1, -0.1, 2.0, 1.0,
        ]);
        let output = layer.forward(&input);
        for &v in output.data.iter() {
            assert!(v > 0.0 && v < 1.0,
                "sigmoid output {} out of (0,1) range", v);
        }
    }

    #[test]
    fn test_forward_output_softmax_columns_sum_to_one() {
        let mut rng   = Rng::new(3);
        let mut layer = Dense::new(4, 5, ActivationType::OutputSoftmax, &mut rng);
        let input     = Matrix::from_vec(4, 3, vec![
            0.1, 0.9, 0.2,
            0.5, 0.1, 0.7,
            0.3, 0.4, 0.1,
            0.8, 0.2, 0.5,
        ]);
        let output = layer.forward(&input);
        for col in 0..3 {
            let col_sum: f64 = (0..5).map(|row| output.get(row, col)).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-6,
                "column {} should sum to 1.0, got {}", col, col_sum
            );
        }
    }

    #[test]
    #[should_panic(expected = "backward() called before forward()")]
    fn test_backward_before_forward_panics() {
        let mut rng   = Rng::new(4);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let fake_grad = Matrix::zeros(3, 1);
        layer.backward(&fake_grad);
    }

    #[test]
    fn test_backward_produces_correct_shapes_batch() {
        let mut rng   = Rng::new(5);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input     = Matrix::from_vec(4, 2, vec![
            1.0, 0.5, 0.5, -1.0,
            -1.0, 2.0, 2.0, 0.1,
        ]);
        layer.forward(&input);

        let output_grad = Matrix::from_vec(3, 2, vec![
            0.1, 0.2, 0.2, 0.1, 0.3, 0.05
        ]);
        let input_grad = layer.backward(&output_grad);

        assert_eq!(input_grad.rows, 4);
        assert_eq!(input_grad.cols, 2);

        let wg = layer.weight_grad.as_ref().unwrap();
        assert_eq!(wg.rows, 3);
        assert_eq!(wg.cols, 4);

        let bg = layer.bias_grad.as_ref().unwrap();
        assert_eq!(bg.rows, 3);
        assert_eq!(bg.cols, 1);
    }

    #[test]
    fn test_output_size() {
        let mut rng   = Rng::new(6);
        let layer     = Dense::new(784, 128, ActivationType::Sigmoid, &mut rng);
        assert_eq!(layer.output_size(), 128);
    }

    #[test]
    fn test_as_dense_mut_returns_some() {
        let mut rng       = Rng::new(7);
        let mut layer     = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let trait_obj: &mut dyn Layer = &mut layer;
        assert!(trait_obj.as_dense_mut().is_some());
    }

    // -----------------------------------------------------------------
    // DROPOUT TESTS
    // -----------------------------------------------------------------

    #[test]
    fn test_dropout_inference_mode_passes_through_unchanged() {
        // In inference mode, dropout must be completely transparent --
        // output identical to input.
        let mut dropout = Dropout::new(0.5, 4, 1);
        dropout.set_training(false);

        let input  = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let output = dropout.forward(&input);

        assert_eq!(input.data, output.data,
            "inference mode must pass input through unchanged");
    }

    #[test]
    fn test_dropout_training_mode_zeros_some_neurons() {
        // In training mode with rate=0.5, roughly 50% of neurons
        // should be zeroed. With enough neurons, at least one should
        // be zero and at least one should be nonzero.
        let mut dropout = Dropout::new(0.5, 100, 42);
        dropout.set_training(true);

        let input  = Matrix::from_vec(100, 1, vec![1.0; 100]);
        let output = dropout.forward(&input);

        let zeros    = output.data.iter().filter(|&&v| v == 0.0).count();
        let nonzeros = output.data.iter().filter(|&&v| v != 0.0).count();

        assert!(zeros > 0,    "at least some neurons should be dropped");
        assert!(nonzeros > 0, "at least some neurons should survive");
    }

    #[test]
    fn test_dropout_inverted_scaling_preserves_expected_value() {
        // With inverted dropout, the EXPECTED VALUE of a surviving
        // neuron's output should equal the input value -- scaling by
        // 1/(1-rate) compensates for the dropped neurons.
        // Over many samples, the average output should ≈ input value.
        let rate        = 0.3;
        let n           = 10_000;
        let input_val   = 2.0;
        let mut dropout = Dropout::new(rate, n, 77);
        dropout.set_training(true);

        let input   = Matrix::from_vec(n, 1, vec![input_val; n]);
        let output  = dropout.forward(&input);
        let mean    = output.data.iter().sum::<f64>() / n as f64;

        // Expected mean = input_val (inverted dropout preserves it).
        // Allow 5% relative tolerance for randomness.
        assert!(
            (mean - input_val).abs() < input_val * 0.05,
            "expected mean ≈ {}, got {:.4}", input_val, mean
        );
    }

    #[test]
    fn test_dropout_backward_zeros_dropped_neurons() {
        // Gradient must be zero for dropped neurons (mask=0) and
        // nonzero (scaled) for surviving neurons.
        let mut dropout = Dropout::new(0.5, 10, 99);
        dropout.set_training(true);

        let input  = Matrix::from_vec(10, 1, vec![1.0; 10]);
        let output = dropout.forward(&input);

        // incoming gradient = all ones
        let grad_in  = Matrix::from_vec(10, 1, vec![1.0; 10]);
        let grad_out = dropout.backward(&grad_in);

        // For each neuron: if output was 0 (dropped), grad must be 0.
        //                  if output was nonzero (kept), grad must be nonzero.
        for i in 0..10 {
            if output.get(i, 0) == 0.0 {
                assert_eq!(grad_out.get(i, 0), 0.0,
                    "dropped neuron {} should have zero gradient", i);
            } else {
                assert!(grad_out.get(i, 0) != 0.0,
                    "surviving neuron {} should have nonzero gradient", i);
            }
        }
    }

    #[test]
    fn test_dropout_output_size_unchanged() {
        // Dropout must never change tensor shape.
        let dropout = Dropout::new(0.3, 128, 1);
        assert_eq!(dropout.output_size(), 128);
    }

    #[test]
    fn test_dropout_set_training_toggles_behavior() {
        // Same network, same input -- training vs inference must differ.
        let mut dropout = Dropout::new(0.9, 100, 55); // high rate -> many zeros
        let input       = Matrix::from_vec(100, 1, vec![1.0; 100]);

        // Training: many zeros expected
        dropout.set_training(true);
        let train_output = dropout.forward(&input);
        let train_zeros  = train_output.data.iter().filter(|&&v| v == 0.0).count();

        // Inference: no zeros expected
        dropout.set_training(false);
        let infer_output = dropout.forward(&input);
        let infer_zeros  = infer_output.data.iter().filter(|&&v| v == 0.0).count();

        assert!(train_zeros > 0,   "training mode should drop some neurons");
        assert_eq!(infer_zeros, 0, "inference mode should drop no neurons");
    }

    #[test]
    fn test_dropout_rate_zero_is_identity() {
        // rate=0.0 means "drop nothing" -- output must equal input
        // exactly in both training and inference mode.
        let mut dropout = Dropout::new(0.0, 5, 1);

        let input = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        dropout.set_training(true);
        let train_out = dropout.forward(&input);
        assert_eq!(train_out.data, input.data);

        dropout.set_training(false);
        let infer_out = dropout.forward(&input);
        assert_eq!(infer_out.data, input.data);
    }
}