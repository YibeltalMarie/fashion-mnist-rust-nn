// =====================================================================
// layer.rs
//
// Defines the Layer trait (a contract every layer type must satisfy)
// and the Dense layer (a fully-connected layer -- every input neuron
// connects to every output neuron).
//
// RESPONSIBILITY BOUNDARY:
// This file ONLY computes forward output and gradients for a single
// layer. It does NOT decide:
//   - how weights get initialized (that's init.rs)
//   - how gradients get applied to update weights (that's optimizer.rs)
// network.rs is responsible for looping over layers and calling the
// optimizer on each one's stored gradients.
//
// FORWARD:  z = W . input + bias ;  output = activation(z)
// BACKWARD: given output_grad (error from the next layer), compute:
//   - weight_grad  (stored, for the optimizer to use)
//   - bias_grad    (stored, for the optimizer to use)
//   - input_grad   (returned, passed backward to the PREVIOUS layer)
// =====================================================================

use crate::matrix::Matrix;
use crate::activation::{self, ActivationType};
use crate::rng::Rng;
use crate::init;

// -----------------------------------------------------------------
// THE Layer TRAIT
//
// Deliberately minimal: forward, backward, and a default
// as_dense_mut() that returns None. Only Dense overrides
// as_dense_mut() to return Some(self), giving network.rs a safe
// way to access Dense-specific fields (weight_grad, bias_grad)
// through a Box<dyn Layer> without a downcast library.
// -----------------------------------------------------------------
pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_grad: &Matrix) -> Matrix;

    /// Default returns None -- only Dense overrides this.
    /// Lets network.rs access Dense fields through Box<dyn Layer>
    /// without needing a full dynamic downcast library.
    fn as_dense_mut(&mut self) -> Option<&mut Dense> {
        None
    }
}

// -----------------------------------------------------------------
// DENSE LAYER
// -----------------------------------------------------------------
pub struct Dense {
    pub weights: Matrix,       // (output_size x input_size)
    pub biases: Matrix,         // (output_size x 1)
    pub activation: ActivationType,

    // Cached during forward(), consumed by backward().
    // Option because they don't exist until first forward() call.
    cached_input: Option<Matrix>,
    cached_output: Option<Matrix>,

    // Computed during backward(), read by optimizer via network.rs.
    // pub so network.rs can pass them to optimizer.step().
    pub weight_grad: Option<Matrix>,
    pub bias_grad: Option<Matrix>,
}

impl Dense {
    /// Creates a new Dense layer. Weight/bias initialization is
    /// fully delegated to init.rs -- Dense does not know or care
    /// which formula was chosen, only that init:: returns a
    /// correctly-shaped, correctly-scaled starting Matrix.
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
        // Step 1: linear combination z = W . input + bias
        let z = self.weights.matmul(input).add(&self.biases);

        // Step 2: apply activation function
        let output = activation::apply(self.activation, &z);

        // Cache for backward() -- .clone() makes an independent copy
        // since input is a borrowed reference that may not remain
        // valid or unchanged by the time backward() runs.
        self.cached_input  = Some(input.clone());
        self.cached_output = Some(output.clone());

        output
    }

    fn backward(&mut self, output_grad: &Matrix) -> Matrix {
        // .as_ref() borrows the inner value without taking ownership.
        // .expect() panics with a clear message if None -- signals a
        // real bug (backward called before forward) rather than silent
        // wrong results.
        let cached_input = self.cached_input.as_ref()
            .expect("backward() called before forward() -- no cached input");
        let cached_output = self.cached_output.as_ref()
            .expect("backward() called before forward() -- no cached output");

        // Step 1: gradient through activation.
        // For Sigmoid: cached_output holds sigmoid(z) -- derivative is s*(1-s).
        // For OutputSoftmax: returns 1.0s (identity) -- gradient passes unchanged
        //   since loss.rs already folded in the softmax derivative.
        let activation_deriv = activation::apply_derivative(
            self.activation, cached_output
        );
        let dz = output_grad.hadamard(&activation_deriv);

        // Step 2: gradient w.r.t weights = dz . input^T
        let weight_grad = dz.matmul(&cached_input.transpose());

        // Step 3: gradient w.r.t biases = dz directly
        let bias_grad = dz.clone();

        // Step 4: gradient to pass to the PREVIOUS layer = W^T . dz
        let input_grad = self.weights.transpose().matmul(&dz);

        // Store for optimizer (network.rs reads these after backward).
        self.weight_grad = Some(weight_grad);
        self.bias_grad   = Some(bias_grad);

        input_grad
    }

    /// Overrides the default None -- gives network.rs access to
    /// Dense-specific fields through a Box<dyn Layer>.
    fn as_dense_mut(&mut self) -> Option<&mut Dense> {
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
    fn test_forward_output_shape() {
        let mut rng = Rng::new(1);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::zeros(4, 1);
        let output = layer.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 1);
    }

    #[test]
    fn test_forward_sigmoid_output_in_range() {
        let mut rng = Rng::new(2);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::from_vec(4, 1, vec![1.0, -2.0, 0.5, 3.0]);
        let output = layer.forward(&input);
        for &v in output.data.iter() {
            assert!(v > 0.0 && v < 1.0,
                "sigmoid output {} out of (0,1) range", v);
        }
    }

    #[test]
    fn test_forward_output_softmax_sums_to_one() {
        let mut rng = Rng::new(3);
        let mut layer = Dense::new(4, 5, ActivationType::OutputSoftmax, &mut rng);
        let input = Matrix::from_vec(4, 1, vec![0.1, 0.5, 0.3, 0.8]);
        let output = layer.forward(&input);
        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "OutputSoftmax output should sum to 1.0, got {}", sum);
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
    fn test_backward_produces_correct_shapes() {
        let mut rng = Rng::new(5);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::from_vec(4, 1, vec![1.0, 0.5, -1.0, 2.0]);

        layer.forward(&input);

        let output_grad = Matrix::from_vec(3, 1, vec![0.1, 0.2, 0.3]);
        let input_grad  = layer.backward(&output_grad);

        // input_grad must match original input shape (4x1).
        assert_eq!(input_grad.rows, 4);
        assert_eq!(input_grad.cols, 1);

        // weight_grad must match weights shape (3x4).
        let wg = layer.weight_grad.as_ref().unwrap();
        assert_eq!(wg.rows, 3);
        assert_eq!(wg.cols, 4);
    }

    #[test]
    fn test_as_dense_mut_returns_some() {
        let mut rng = Rng::new(6);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        // Cast through the trait to confirm as_dense_mut works.
        let trait_obj: &mut dyn Layer = &mut layer;
        assert!(trait_obj.as_dense_mut().is_some());
    }

    #[test]
    fn test_dense_delegates_init_to_init_module() {
        // Confirms Dense::new produces weights identical to calling
        // init::init_weights directly with the same seed -- verifying
        // Dense truly delegates rather than duplicating logic.
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
        assert!(max_abs < 1.0,
            "weights too large for Xavier init with 784 inputs");
    }
}