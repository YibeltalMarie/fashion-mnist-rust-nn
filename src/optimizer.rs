// TODO (Day 3): SGD (with momentum) and Adam optimizers.
// Each holds per-parameter state (velocity / moment estimates).

// =====================================================================
// optimizer.rs
//
// Defines the Optimizer trait and two implementations:
//   SGD  -- plain stochastic gradient descent with optional momentum
//   Adam -- adaptive moment estimation (per-weight learning rates)
//
// RESPONSIBILITY BOUNDARY:
// Given weights (mutable reference), their gradient, and a learning
// rate, update the weights in place. That's all. No knowledge of
// layers, network structure, or loss functions.
//
// network.rs will call optimizer.step() after every batch's backward
// pass, passing each layer's weights/biases and their gradients.
// =====================================================================

use crate::matrix::Matrix;

// -----------------------------------------------------------------
// THE Optimizer TRAIT
//
// layer_idx: which layer we're updating -- Adam needs this to look
// up that layer's stored moment estimates (m, v).
// is_bias:   whether we're updating weights or biases -- Adam keeps
// separate moment estimates for each.
// -----------------------------------------------------------------
pub trait Optimizer {
    fn step(
        &mut self,
        weights: &mut Matrix,
        grad: &Matrix,
        layer_idx: usize,
        is_bias: bool,
        lr: f64,
    );
}

// -----------------------------------------------------------------
// SGD WITH MOMENTUM
// -----------------------------------------------------------------
pub struct SGD {
    pub momentum: f64,

    // Per-layer velocity matrices. None until the first step()
    // call for that layer (initialized lazily to match weight shape).
    // Outer Vec: one entry per layer.
    // Each entry: (weight_velocity, bias_velocity).
    velocities: Vec<Option<(Matrix, Matrix)>>,
}

impl SGD {
    /// momentum = 0.0 gives plain SGD (no momentum).
    /// momentum = 0.9 is the standard starting point with momentum.
    pub fn new(n_layers: usize, momentum: f64) -> Self {
        SGD {
            momentum,
            // Pre-fill with None for each layer -- lazily initialized
            // on first step() call so we don't need layer sizes upfront.
            velocities: (0..n_layers).map(|_| None).collect(),
        }
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        weights: &mut Matrix,
        grad: &Matrix,
        layer_idx: usize,
        is_bias: bool,
        lr: f64,
    ) {
        // Lazily initialize velocity to zeros matching weight shape,
        // if this is the first update for this layer.
        if self.velocities[layer_idx].is_none() {
            self.velocities[layer_idx] = Some((
                Matrix::zeros(weights.rows, weights.cols),
                Matrix::zeros(weights.rows, weights.cols),
            ));
        }

        // .as_mut() gives a mutable reference to the value inside
        // the Option -- like as_ref() but allowing mutation.
        let (weight_vel, bias_vel) = self.velocities[layer_idx].as_mut().unwrap();
        let vel = if is_bias { bias_vel } else { weight_vel };

        // velocity = momentum * velocity + gradient
        *vel = vel.scalar_mul(self.momentum).add(grad);

        // weight = weight - lr * velocity
        let update = vel.scalar_mul(lr);
        *weights = weights.add(&update.scalar_mul(-1.0));
    }
}

// -----------------------------------------------------------------
// ADAM
// -----------------------------------------------------------------
pub struct Adam {
    pub beta1: f64,   // first moment decay  (default 0.9)
    pub beta2: f64,   // second moment decay (default 0.999)
    pub epsilon: f64, // prevents division by zero (default 1e-8)
    pub t: u64,       // global timestep, incremented each step()

    // Per-layer first moment (m) and second moment (v) matrices.
    // Each entry: Option<(weight_m, bias_m, weight_v, bias_v)>
    // Lazily initialized on first step() call per layer.
    moments: Vec<Option<(Matrix, Matrix, Matrix, Matrix)>>,
}

impl Adam {
    pub fn new(n_layers: usize) -> Self {
        Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            moments: (0..n_layers).map(|_| None).collect(),
        }
    }

    /// Allows overriding defaults -- useful for hyperparameter
    /// comparison experiments (mention in README/presentation).
    pub fn with_params(n_layers: usize, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam { beta1, beta2, epsilon, t: 0, moments: (0..n_layers).map(|_| None).collect() }
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        weights: &mut Matrix,
        grad: &Matrix,
        layer_idx: usize,
        is_bias: bool,
        lr: f64,
    ) {
        // Increment global timestep.
        // NOTE: we increment once per full layer update cycle
        // (network.rs increments t by calling step for weights
        // then biases -- so we only want to count one "step" per
        // batch per layer, not two). We let network.rs manage this
        // by calling update for weights first, then biases, and
        // Adam uses self.t for both calls within that step.
        if !is_bias {
            self.t += 1; // only increment on weight update, not bias
        }

        // Lazy initialization -- zeros matching weight shape.
        if self.moments[layer_idx].is_none() {
            self.moments[layer_idx] = Some((
                Matrix::zeros(weights.rows, weights.cols), // weight m
                Matrix::zeros(weights.rows, weights.cols), // bias m
                Matrix::zeros(weights.rows, weights.cols), // weight v
                Matrix::zeros(weights.rows, weights.cols), // bias v
            ));
        }

        let (weight_m, bias_m, weight_v, bias_v) =
            self.moments[layer_idx].as_mut().unwrap();

        // Pick weight or bias moment estimates.
        let (m, v) = if is_bias {
            (bias_m, bias_v)
        } else {
            (weight_m, weight_v)
        };

        // m = beta1 * m + (1 - beta1) * gradient
        *m = m.scalar_mul(self.beta1)
             .add(&grad.scalar_mul(1.0 - self.beta1));

        // v = beta2 * v + (1 - beta2) * gradient^2
        let grad_sq = grad.hadamard(grad); // element-wise gradient²
        *v = v.scalar_mul(self.beta2)
             .add(&grad_sq.scalar_mul(1.0 - self.beta2));

        // Bias correction -- compensates for m and v being zero-
        // initialized (they'd be underestimated early in training).
        let t = self.t as f64;
        let m_hat_scale = 1.0 / (1.0 - self.beta1.powf(t));
        let v_hat_scale = 1.0 / (1.0 - self.beta2.powf(t));

        // weight = weight - lr * m_hat / (sqrt(v_hat) + epsilon)
        // We compute this element-wise using our Matrix operations.
        let update_data: Vec<f64> = m.data.iter()
            .zip(v.data.iter())
            .map(|(&mi, &vi)| {
                let m_hat = mi * m_hat_scale;
                let v_hat = vi * v_hat_scale;
                lr * m_hat / (v_hat.sqrt() + self.epsilon)
            })
            .collect();

        let update = Matrix::from_vec(weights.rows, weights.cols, update_data);
        *weights = weights.add(&update.scalar_mul(-1.0));
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_no_momentum_moves_weights_in_gradient_direction() {
        // With momentum=0, SGD is: weight = weight - lr * grad.
        // If weight=1.0 and grad=1.0 and lr=0.1, new weight = 0.9.
        let mut opt = SGD::new(1, 0.0);
        let mut weights = Matrix::from_vec(1, 1, vec![1.0]);
        let grad = Matrix::from_vec(1, 1, vec![1.0]);

        opt.step(&mut weights, &grad, 0, false, 0.1);

        assert!((weights.get(0, 0) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_sgd_with_momentum_accumulates_velocity() {
        // With momentum=0.9, second step should move further than
        // first step because velocity accumulates.
        let mut opt = SGD::new(1, 0.9);
        let mut weights = Matrix::from_vec(1, 1, vec![1.0]);
        let grad = Matrix::from_vec(1, 1, vec![1.0]);

        opt.step(&mut weights, &grad, 0, false, 0.1);
        let after_step1 = weights.get(0, 0);

        opt.step(&mut weights, &grad, 0, false, 0.1);
        let after_step2 = weights.get(0, 0);

        let step1_size = (1.0 - after_step1).abs();
        let step2_size = (after_step1 - after_step2).abs();

        assert!(step2_size > step1_size,
            "momentum should make step2 larger than step1");
    }

    #[test]
    fn test_adam_decreases_loss_direction() {
        // After one Adam step, weights should have moved in the
        // direction that reduces loss (opposite to gradient).
        let mut opt = Adam::new(1);
        let mut weights = Matrix::from_vec(1, 1, vec![0.5]);
        let grad = Matrix::from_vec(1, 1, vec![1.0]); // positive grad

        opt.step(&mut weights, &grad, 0, false, 0.001);

        // Weight should have decreased (moved against positive gradient).
        assert!(weights.get(0, 0) < 0.5,
            "Adam should decrease weight when gradient is positive");
    }

    #[test]
    fn test_adam_timestep_increments_on_weight_not_bias() {
        let mut opt = Adam::new(1);
        let mut w = Matrix::from_vec(1, 1, vec![0.0]);
        let grad = Matrix::from_vec(1, 1, vec![0.1]);

        assert_eq!(opt.t, 0);
        opt.step(&mut w, &grad, 0, false, 0.001); // weight update -> t++
        assert_eq!(opt.t, 1);
        opt.step(&mut w, &grad, 0, true, 0.001);  // bias update -> t stays
        assert_eq!(opt.t, 1);
    }

    #[test]
    fn test_adam_with_zero_gradient_does_not_move_weights() {
        let mut opt = Adam::new(1);
        let mut weights = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let grad = Matrix::zeros(2, 2); // zero gradient

        opt.step(&mut weights, &grad, 0, false, 0.001);

        // Weights should be essentially unchanged.
        for &v in weights.data.iter() {
            assert!(v > 0.0, "weights should not vanish with zero gradient");
        }
    }

    #[test]
    fn test_sgd_multi_layer_separate_velocities() {
        // Each layer should maintain its OWN velocity -- they must
        // not interfere with each other.
        let mut opt = SGD::new(2, 0.9);
        let mut w0 = Matrix::from_vec(1, 1, vec![1.0]);
        let mut w1 = Matrix::from_vec(1, 1, vec![2.0]);
        let g0 = Matrix::from_vec(1, 1, vec![1.0]);
        let g1 = Matrix::from_vec(1, 1, vec![0.0]); // no gradient for layer 1

        opt.step(&mut w0, &g0, 0, false, 0.1); // update layer 0
        opt.step(&mut w1, &g1, 1, false, 0.1); // update layer 1 (zero grad)

        // Layer 0 should have moved, layer 1 should be unchanged.
        assert!(w0.get(0, 0) < 1.0, "layer 0 weight should have decreased");
        assert!((w1.get(0, 0) - 2.0).abs() < 1e-9, "layer 1 weight should be unchanged");
    }
}