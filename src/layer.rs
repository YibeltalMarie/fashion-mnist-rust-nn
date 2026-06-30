// TODO (Day 2): Layer trait (forward/backward) + Dense layer struct.
// Dense holds weights: Matrix, biases: Vec<f64>, cached input/output for backward.

// =====================================================================
// layer.rs
//
// Defines the Layer trait (a contract every layer type must satisfy)
// and the Dense layer (a fully-connected layer -- every input neuron
// connects to every output neuron).
//
// RESPONSIBILITY BOUNDARY (important):
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
// -----------------------------------------------------------------
pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_grad: &Matrix) -> Matrix;
}

// -----------------------------------------------------------------
// DENSE LAYER
// -----------------------------------------------------------------
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
    /// Creates a new Dense layer. Weight/bias initialization is fully
    /// delegated to init.rs -- this function doesn't know or care
    /// which formula was used, only that init:: gives back a
    /// correctly-shaped, correctly-scaled starting point.
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType, rng: &mut Rng) -> Self {
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
        let z = self.weights.matmul(input).add(&self.biases);
        let output = activation::apply(self.activation, &z);

        self.cached_input = Some(input.clone());
        self.cached_output = Some(output.clone());

        output
    }

    fn backward(&mut self, output_grad: &Matrix) -> Matrix {
        let cached_input = self.cached_input.as_ref()
            .expect("backward() called before forward() -- no cached input");
        let cached_output = self.cached_output.as_ref()
            .expect("backward() called before forward() -- no cached output");

        let activation_deriv = activation::apply_derivative(self.activation, cached_output);
        let dz = output_grad.hadamard(&activation_deriv);

        let weight_grad = dz.matmul(&cached_input.transpose());
        let bias_grad = dz.clone();
        let input_grad = self.weights.transpose().matmul(&dz);

        self.weight_grad = Some(weight_grad);
        self.bias_grad = Some(bias_grad);

        input_grad
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
            assert!(v > 0.0 && v < 1.0, "sigmoid output {} out of range", v);
        }
    }

    #[test]
    #[should_panic(expected = "backward() called before forward()")]
    fn test_backward_before_forward_panics() {
        let mut rng = Rng::new(3);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let fake_grad = Matrix::zeros(3, 1);
        layer.backward(&fake_grad);
    }

    #[test]
    fn test_backward_produces_correct_shapes() {
        let mut rng = Rng::new(4);
        let mut layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng);
        let input = Matrix::from_vec(4, 1, vec![1.0, 0.5, -1.0, 2.0]);

        layer.forward(&input);

        let output_grad = Matrix::from_vec(3, 1, vec![0.1, 0.2, 0.3]);
        let input_grad = layer.backward(&output_grad);

        assert_eq!(input_grad.rows, 4);
        assert_eq!(input_grad.cols, 1);

        let wg = layer.weight_grad.as_ref().unwrap();
        assert_eq!(wg.rows, 3);
        assert_eq!(wg.cols, 4);
    }

    #[test]
    fn test_dense_uses_init_module_correctly() {
        // Confirms Dense::new produces weights matching init::init_weights
        // directly -- i.e. Dense really did delegate, not duplicate logic.
        let mut rng_a = Rng::new(99);
        let mut rng_b = Rng::new(99);

        let layer = Dense::new(4, 3, ActivationType::Sigmoid, &mut rng_a);
        let expected_weights = init::init_weights(3, 4, ActivationType::Sigmoid, &mut rng_b);

        assert_eq!(layer.weights.data, expected_weights.data);
    }
}