// =====================================================================
// layer.rs
//
// Defines the Layer trait and the Dense layer.
//
// BATCHING (new): forward()/backward() now operate on WHOLE BATCHES.
// `input` is (input_size x batch_size) -- one column per sample --
// instead of a single (input_size x 1) column. This is what lets
// network.rs process an entire mini-batch with ONE matmul per layer
// instead of looping per sample.
//
// GRADIENT SEMANTICS (important): because dz.matmul(input^T) sums
// over the batch dimension automatically (that's just what matmul
// does), weight_grad and bias_grad coming out of backward() are the
// SUM of gradients across the batch, NOT the average. network.rs is
// responsible for dividing by batch_size before calling the optimizer.
//
// RESPONSIBILITY BOUNDARY unchanged: this file only computes forward
// output and gradients for a single layer. Weight init is init.rs's
// job; applying gradients to update weights is optimizer.rs's job.
// =====================================================================

use crate::matrix::Matrix;
use crate::activation::{self, ActivationType};
use crate::rng::Rng;
use crate::init;

pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_grad: &Matrix) -> Matrix;

    fn as_dense_mut(&mut self) -> Option<&mut Dense> {
        None
    }

    fn as_dense(&self) -> Option<&Dense> {
        None
    }

    fn output_size(&self) -> usize;
}

pub struct Dense {
    pub weights: Matrix,       // (output_size x input_size)
    pub biases: Matrix,         // (output_size x 1)
    pub activation: ActivationType,

    // Cached during forward(): now BATCHED matrices --
    // (size x batch_size) instead of (size x 1).
    cached_input: Option<Matrix>,
    cached_output: Option<Matrix>,

    // SUMMED (not averaged) across the batch -- network.rs divides
    // by batch_size before passing these to the optimizer.
    pub weight_grad: Option<Matrix>,
    pub bias_grad: Option<Matrix>,
}

impl Dense {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
        rng: &mut Rng,
    ) -> Self {
        Dense {
            weights: init::init_weights(output_size, input_size, activation, rng),
            biases: init::init_biases(output_size),
            activation,
            cached_input: None,
            cached_output: None,
            weight_grad: None,
            bias_grad: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        // input: (input_size x batch_size)
        // z = W . input + bias, bias broadcast across every column
        //   -- ONE matmul handles the whole batch.
        //   -- matmul_parallel splits this across CPU cores by output row.
        let z = self.weights.matmul_parallel(input).add_bias_broadcast(&self.biases);
        let output = activation::apply(self.activation, &z);

        self.cached_input  = Some(input.clone());
        self.cached_output = Some(output.clone());

        output
    }

    fn backward(&mut self, output_grad: &Matrix) -> Matrix {
        // output_grad: (output_size x batch_size)
        let cached_input = self.cached_input.as_ref()
            .expect("backward() called before forward() -- no cached input");
        let cached_output = self.cached_output.as_ref()
            .expect("backward() called before forward() -- no cached output");

        // Activation derivative is elementwise, so it works unchanged
        // whether the matrix is one column or a whole batch.
        let activation_deriv = activation::apply_derivative(self.activation, cached_output);
        let dz = output_grad.hadamard(&activation_deriv);
        // dz: (output_size x batch_size)

        // weight_grad = dz . input^T
        //   (output_size x batch) * (batch x input_size) = (output_size x input_size)
        // This SUMS over the batch dimension automatically -- matmul's
        // inner product does the summing for us, "for free".
        let weight_grad = dz.matmul_parallel(&cached_input.transpose());

        // bias_grad = sum of dz across the batch (bias affects every
        // sample identically, so its gradient is the sum across samples).
        let bias_grad = dz.sum_cols();

        // input_grad = W^T . dz -- passed back to the previous layer,
        // one column of gradient per sample, NOT summed.
        let input_grad = self.weights.transpose().matmul_parallel(&dz);

        self.weight_grad = Some(weight_grad);
        self.bias_grad   = Some(bias_grad);

        input_grad
    }

    fn output_size(&self) -> usize {
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
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_output_shape_single_sample() {
        // batch_size = 1 is just the old single-sample case --
        // confirms batching doesn't break the simple case.
        let mut rng = Rng::new(1);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::zeros(4, 1);
        let output = layer.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 1);
    }

    #[test]
    fn test_forward_output_shape_batch() {
        // batch_size = 5 -- output must have 5 columns, one per sample.
        let mut rng = Rng::new(2);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::zeros(4, 5);
        let output = layer.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 5);
    }

    #[test]
    fn test_forward_sigmoid_output_in_range() {
        let mut rng = Rng::new(2);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::from_vec(4, 2, vec![1.0, -2.0, 0.5, 3.0, 0.1, -0.1, 2.0, 1.0]);
        let output = layer.forward(&input);
        for &v in output.data.iter() {
            assert!(v > 0.0 && v < 1.0, "sigmoid output {} out of (0,1) range", v);
        }
    }

    #[test]
    fn test_forward_output_softmax_columns_sum_to_one() {
        // With a BATCH, EACH COLUMN must independently sum to 1.0 --
        // this is the exact bug batching could introduce if softmax
        // normalized globally instead of per-column.
        let mut rng = Rng::new(3);
        let mut layer = Dense::new(4, 5, ActivationType::OutputSoftmax, &mut rng);
        let input = Matrix::from_vec(4, 3, vec![
            0.1, 0.9, 0.2,
            0.5, 0.1, 0.7,
            0.3, 0.4, 0.1,
            0.8, 0.2, 0.5,
        ]);
        let output = layer.forward(&input);

        for col in 0..3 {
            let mut col_sum = 0.0;
            for row in 0..5 {
                col_sum += output.get(row, col);
            }
            assert!(
                (col_sum - 1.0).abs() < 1e-6,
                "column {} should sum to 1.0, got {}", col, col_sum
            );
        }
    }

    #[test]
    #[should_panic(expected = "backward() called before forward()")]
    fn test_backward_before_forward_panics() {
        let mut rng = Rng::new(4);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let fake_grad = Matrix::zeros(3, 1);
        layer.backward(&fake_grad);
    }

    #[test]
    fn test_backward_produces_correct_shapes_batch() {
        let mut rng = Rng::new(5);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::from_vec(4, 2, vec![
            1.0, 0.5,
            0.5, -1.0,
            -1.0, 2.0,
            2.0, 0.1,
        ]);

        layer.forward(&input);

        let output_grad = Matrix::from_vec(3, 2, vec![0.1, 0.2, 0.2, 0.1, 0.3, 0.05]);
        let input_grad = layer.backward(&output_grad);

        // input_grad must match ORIGINAL input shape: (4 x batch=2).
        assert_eq!(input_grad.rows, 4);
        assert_eq!(input_grad.cols, 2);

        // weight_grad shape is (output x input) regardless of batch
        // size -- batching is summed away by the matmul.
        let wg = layer.weight_grad.as_ref().unwrap();
        assert_eq!(wg.rows, 3);
        assert_eq!(wg.cols, 4);

        // bias_grad must be (output_size x 1) -- summed across batch.
        let bg = layer.bias_grad.as_ref().unwrap();
        assert_eq!(bg.rows, 3);
        assert_eq!(bg.cols, 1);
    }

    #[test]
    fn test_as_dense_mut_returns_some() {
        let mut rng = Rng::new(6);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let trait_obj: &mut dyn Layer = &mut layer;
        assert!(trait_obj.as_dense_mut().is_some());
    }

    #[test]
    fn test_dense_delegates_init_to_init_module() {
        let mut rng_a = Rng::new(99);
        let mut rng_b = Rng::new(99);
        let layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng_a);
        let expected = init::init_weights(3, 4, ActivationType::Sigmoid, &mut rng_b);
        assert_eq!(layer.weights.data, expected.data);
    }

    #[test]
    fn test_xavier_init_reasonable_scale_for_large_layer() {
        let mut rng = Rng::new(7);
        let layer = Dense::new(784, 128, ActivationType::Sigmoid, &mut rng);
        let max_abs = layer.weights.data.iter()
            .fold(0.0_f64, |acc, &x| if x.abs() > acc { x.abs() } else { acc });
        assert!(max_abs > 0.0, "weights should not be all zero");
        assert!(max_abs < 1.0, "weights too large for Xavier init with 784 inputs");
    }
}