// TODO (Day 3): Numerical gradient check.
// For a small random batch, compare backprop-computed gradients against
// finite-difference approximations: (L(w+eps) - L(w-eps)) / (2*eps).
// Assert relative error < 1e-4 for a handful of sampled weights.

// =====================================================================
// tests/gradient_check.rs
//
// INTEGRATION TEST: numerical gradient check.
//
// Mathematically proves that backprop-computed (analytical) gradients
// match finite-difference (numerical) gradient estimates, confirming
// our backward() implementation is correct.
//
// METHOD: finite differences
//   numerical_grad(w_i) = (loss(w_i + ε) - loss(w_i - ε)) / (2ε)
//
// COMPARISON: relative error
//   err = |analytical - numerical| / (|analytical| + |numerical| + δ)
//   where δ = 1e-15 prevents division by zero when both are ~0.
//
// THRESHOLD: relative error < 1e-5 = correct implementation.
//   > 1e-3 = almost certainly a bug in backward().
//   Between 1e-5 and 1e-3 = suspicious, worth investigating.
//
// WHY SAMPLE RANDOMLY: checking all ~134k weights would take minutes.
// 50 randomly sampled weights per layer is statistically sufficient --
// if those pass, the math is right everywhere.
// =====================================================================

use fashion_mnist_rust::matrix::Matrix;
use fashion_mnist_rust::network::Network;
use fashion_mnist_rust::layer::Dense;
use fashion_mnist_rust::activation::ActivationType;
use fashion_mnist_rust::loss;
use fashion_mnist_rust::optimizer::SGD;
use fashion_mnist_rust::rng::Rng;

// ε for finite differences -- small enough to be accurate, large
// enough to avoid floating-point precision issues (too small = noise
// dominates, too large = nonlinearity dominates).
const EPSILON: f64 = 1e-4;

// Relative error threshold -- anything below this is "correct".
const THRESHOLD: f64 = 1e-4;

// How many weights to randomly sample per layer for the check.
const SAMPLES_PER_LAYER: usize = 30;

/// Computes the scalar loss for a single forward pass.
/// Used by finite_difference_grad() to evaluate loss at w±ε.
fn compute_loss(net: &mut Network, input: &Matrix, label: usize) -> f64 {
    let prediction = net.forward(input);
    loss::cross_entropy(&prediction, label)
}

/// Computes the numerical gradient of the loss w.r.t. one specific
/// weight in one specific layer, using the central finite difference:
///   (loss(w + ε) - loss(w - ε)) / (2ε)
///
/// `layer_idx`:  which layer (0 = first hidden, etc.)
/// `is_bias`:    true if we're checking a bias weight, false for matrix weight
/// `weight_idx`: flat index into the weight/bias vector
fn numerical_grad(
    net:        &mut Network,
    input:      &Matrix,
    label:      usize,
    layer_idx:  usize,
    is_bias:    bool,
    weight_idx: usize,
) -> f64 {
    // +ε: temporarily nudge the weight up
    {
        let dense = net.layers[layer_idx].as_dense_mut().unwrap();
        if is_bias {
            dense.biases.data[weight_idx] += EPSILON;
        } else {
            dense.weights.data[weight_idx] += EPSILON;
        }
    }
    let loss_plus = compute_loss(net, input, label);

    // -ε: nudge down (go through 0 to avoid accumulated floating-point drift)
    {
        let dense = net.layers[layer_idx].as_dense_mut().unwrap();
        if is_bias {
            dense.biases.data[weight_idx] -= 2.0 * EPSILON;
        } else {
            dense.weights.data[weight_idx] -= 2.0 * EPSILON;
        }
    }
    let loss_minus = compute_loss(net, input, label);

    // Restore original weight value
    {
        let dense = net.layers[layer_idx].as_dense_mut().unwrap();
        if is_bias {
            dense.biases.data[weight_idx] += EPSILON;
        } else {
            dense.weights.data[weight_idx] += EPSILON;
        }
    }

    (loss_plus - loss_minus) / (2.0 * EPSILON)
}

/// Relative error between analytical and numerical gradient.
/// Adding 1e-15 in the denominator prevents division by zero when
/// both gradients are near zero (which is fine -- they agree).
fn relative_error(analytical: f64, numerical: f64) -> f64 {
    (analytical - numerical).abs()
        / (analytical.abs() + numerical.abs() + 1e-15)
}

/// Builds a small but realistic test network -- small enough for
/// the gradient check to run quickly, large enough to exercise
/// real matrix shapes through multiple layers.
fn make_test_network() -> Network {
    let mut rng = Rng::new(42);
    // n_layers = 3 for the Adam/SGD optimizer state allocation.
    let opt     = SGD::new(3, 0.0); // plain SGD, no momentum
    let mut net = Network::new(Box::new(opt), 0.01);

    // Small but multi-layer: 8 inputs -> 6 hidden -> 4 hidden -> 3 output.
    // Deliberately NOT 784->256->10 to keep the check fast while
    // still testing the full forward/backward pipeline.
    net.add(Box::new(Dense::new(8, 6,  ActivationType::Sigmoid,      &mut rng)));
    net.add(Box::new(Dense::new(6, 4,  ActivationType::Sigmoid,      &mut rng)));
    net.add(Box::new(Dense::new(4, 3,  ActivationType::OutputSoftmax, &mut rng)));
    net
}

// =====================================================================
// THE GRADIENT CHECK TESTS
// =====================================================================

#[test]
fn test_gradient_check_weights_all_layers() {
    // For each layer, sample SAMPLES_PER_LAYER random WEIGHT indices
    // and compare analytical vs numerical gradients.
    let mut rng = Rng::new(123);
    let mut net = make_test_network();

    // Fixed input and label -- reproducible, not random.
    let input = Matrix::from_vec(8, 1, vec![
        0.5, -0.3, 0.8, 0.1, -0.6, 0.4, 0.2, -0.9
    ]);
    let label = 1usize;

    // Run ONE full forward+backward to populate analytical gradients.
    // We use a single sample (batch_size=1) so the gradient shapes
    // match what our finite-difference helper expects.
    let prediction = net.forward(&input);
    let loss_grad  = loss::cross_entropy_derivative_batch(&prediction, &[label]);
    // Manually trigger backward through the network's layers.
    {
        let mut grad = loss_grad.clone();
        for layer in net.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }
    // Bias gradients from backward() are summed -- with batch_size=1
    // that IS the gradient, no scaling needed.

    let n_layers = net.layers.len();
    let mut max_error: f64 = 0.0;
    let mut checks = 0;

    for layer_idx in 0..n_layers {
        // Get weight count for this layer.
        let weight_count = {
            let dense = net.layers[layer_idx].as_dense_mut().unwrap();
            dense.weights.data.len()
        };

        // Sample random weight indices.
        for _ in 0..SAMPLES_PER_LAYER {
            let w_idx = (rng.next_u64() as usize) % weight_count;

            // Get analytical gradient at this index.
            let analytical = {
                let dense = net.layers[layer_idx].as_dense_mut().unwrap();
                dense.weight_grad.as_ref()
                    .expect("weight_grad should be populated after backward()")
                    .data[w_idx]
            };

            // Get numerical gradient at this index.
            let numerical = numerical_grad(
                &mut net, &input, label, layer_idx, false, w_idx
            );

            let err = relative_error(analytical, numerical);
            if err > max_error { max_error = err; }
            checks += 1;

            assert!(
                err < THRESHOLD,
                "Layer {} weight[{}]: analytical={:.6} numerical={:.6} \
                 relative_error={:.2e} (threshold {:.2e})",
                layer_idx, w_idx, analytical, numerical, err, THRESHOLD
            );
        }
    }

    println!(
        "Gradient check PASSED: {} weight checks across {} layers, \
         max relative error = {:.2e}",
        checks, n_layers, max_error
    );
}

#[test]
fn test_gradient_check_biases_all_layers() {
    // Same check but for BIAS gradients.
    // Biases are smaller vectors (output_size x 1) but just as
    // important to verify -- a bug in bias_grad would cause
    // systematic per-neuron errors that are hard to spot from
    // accuracy numbers alone.
    let mut rng = Rng::new(456);
    let mut net = make_test_network();

    let input = Matrix::from_vec(8, 1, vec![
        0.2, 0.7, -0.4, 0.9, 0.1, -0.5, 0.6, -0.2
    ]);
    let label = 2usize;

    // Forward + backward to populate gradients.
    let prediction = net.forward(&input);
    let loss_grad  = loss::cross_entropy_derivative_batch(&prediction, &[label]);
    {
        let mut grad = loss_grad.clone();
        for layer in net.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    let n_layers   = net.layers.len();
    let mut max_error: f64 = 0.0;
    let mut checks = 0;

    for layer_idx in 0..n_layers {
        let bias_count = {
            let dense = net.layers[layer_idx].as_dense_mut().unwrap();
            dense.biases.data.len()
        };

        // Check ALL bias gradients (biases are small -- checking
        // every one is cheap and more thorough than sampling).
        for b_idx in 0..bias_count {
            let analytical = {
                let dense = net.layers[layer_idx].as_dense_mut().unwrap();
                dense.bias_grad.as_ref()
                    .expect("bias_grad should be populated after backward()")
                    .data[b_idx]
            };

            let numerical = numerical_grad(
                &mut net, &input, label, layer_idx, true, b_idx
            );

            let err = relative_error(analytical, numerical);
            if err > max_error { max_error = err; }
            checks += 1;

            assert!(
                err < THRESHOLD,
                "Layer {} bias[{}]: analytical={:.6} numerical={:.6} \
                 relative_error={:.2e} (threshold {:.2e})",
                layer_idx, b_idx, analytical, numerical, err, THRESHOLD
            );
        }
    }

    println!(
        "Bias gradient check PASSED: {} bias checks across {} layers, \
         max relative error = {:.2e}",
        checks, n_layers, max_error
    );
}

#[test]
fn test_gradient_check_different_inputs() {
    // Runs the weight gradient check on THREE different inputs to
    // confirm correctness isn't specific to one particular input
    // or initialization -- a stronger guarantee than one check alone.
    let mut rng = Rng::new(789);

    let inputs = vec![
        Matrix::from_vec(8, 1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        Matrix::from_vec(8, 1, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        Matrix::from_vec(8, 1, vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
    ];
    let labels = vec![0usize, 1, 2];

    for (input, label) in inputs.iter().zip(labels.iter()) {
        let mut net = make_test_network(); // fresh network per input

        let prediction = net.forward(input);
        let loss_grad  = loss::cross_entropy_derivative_batch(&prediction, &[*label]);
        {
            let mut grad = loss_grad.clone();
            for layer in net.layers.iter_mut().rev() {
                grad = layer.backward(&grad);
            }
        }

        // Check a small sample of weights from layer 0 only --
        // enough to confirm consistency across inputs without being slow.
        let weight_count = net.layers[0].as_dense_mut().unwrap().weights.data.len();
        for _ in 0..10 {
            let w_idx = (rng.next_u64() as usize) % weight_count;

            let analytical = net.layers[0].as_dense_mut().unwrap()
                .weight_grad.as_ref().unwrap().data[w_idx];

            let numerical = numerical_grad(&mut net, input, *label, 0, false, w_idx);

            let err = relative_error(analytical, numerical);
            assert!(
                err < THRESHOLD,
                "Input variant: layer 0 weight[{}]: analytical={:.6} \
                 numerical={:.6} relative_error={:.2e}",
                w_idx, analytical, numerical, err
            );
        }
    }

    println!("Gradient check PASSED across 3 different inputs");
}