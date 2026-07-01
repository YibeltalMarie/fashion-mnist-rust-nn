// =====================================================================
// network.rs
//
// The conductor: holds all layers and the optimizer, runs the
// complete training loop, evaluates validation accuracy each epoch.
//
// BATCHED TRAINING: the whole batch is assembled into ONE matrix
// (input_size x batch_size) and pushed through forward()/backward()
// ONCE. Each layer's forward/backward does one big matmul
// (parallelized across CPU cores via matmul_parallel).
//
// GRADIENT AVERAGING: Dense::backward() returns gradients SUMMED
// across the batch. scale_gradients() divides by batch_size before
// the optimizer step, keeping lr independent of batch size.
//
// DROPOUT: set_training_mode(true) before training batches,
// set_training_mode(false) before evaluate(). Only Dropout layers
// respond -- Dense layers ignore this call (default no-op in trait).
//
// LEARNING RATE DECAY: halves lr every `decay_every` epochs.
// Helps fine-tune in later epochs without overshooting the minimum.
//
// ARCHITECTURE (built in main.rs):
//   Input (784)
//     -> Dense(784->256, Sigmoid)
//     -> Dropout(0.15)
//     -> Dense(256->128, Sigmoid)
//     -> Dropout(0.15)
//     -> Dense(128->10,  OutputSoftmax)
// =====================================================================

use crate::matrix::Matrix;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::loss;
use crate::rng::Rng;

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    pub learning_rate: f64,
}

impl Network {
    pub fn new(optimizer: Box<dyn Optimizer>, learning_rate: f64) -> Self {
        Network {
            layers: Vec::new(),
            optimizer,
            learning_rate,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    // -----------------------------------------------------------------
    // TRAINING MODE TOGGLE
    //
    // Walks every layer and calls set_training(). Only Dropout layers
    // actually respond -- Dense layers' default implementation is a
    // no-op (defined in the Layer trait with an empty body).
    // Called with true before each training batch, false before
    // evaluate() so dropout is disabled during inference.
    // -----------------------------------------------------------------
    fn set_training_mode(&mut self, training: bool) {
        for layer in self.layers.iter_mut() {
            layer.set_training(training);
        }
    }

    // -----------------------------------------------------------------
    // FORWARD PASS
    //
    // Works uniformly whether `input` is one sample (cols=1) or a
    // whole batch (cols=batch_size). Each layer's forward() overwrites
    // its own cache -- fine because backward() always immediately
    // follows forward() on the same batch before the next forward().
    // -----------------------------------------------------------------
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in self.layers.iter_mut() {
            current = layer.forward(&current);
        }
        current
    }

    /// Convenience alias for single-sample inference outside the
    /// training loop (used in main.rs for collecting predictions).
    pub fn forward_single(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    // -----------------------------------------------------------------
    // BACKWARD PASS
    //
    // Feeds gradient back through layers in REVERSE order.
    // Called immediately after forward() on the same batch, before
    // the next forward() call overwrites the cache.
    // -----------------------------------------------------------------
    fn backward(&mut self, loss_grad: &Matrix) {
        let mut grad = loss_grad.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    // -----------------------------------------------------------------
    // GRADIENT SCALING
    //
    // Dense::backward() sums gradients across the batch (natural
    // side effect of matmul). This divides every Dense layer's stored
    // gradients by batch_size, turning the sum into an average --
    // keeps the learning rate meaningful regardless of batch size.
    // -----------------------------------------------------------------
    fn scale_gradients(&mut self, scale: f64) {
        for layer in self.layers.iter_mut() {
            if let Some(dense) = layer.as_dense_mut() {
                if let Some(wg) = dense.weight_grad.take() {
                    dense.weight_grad = Some(wg.scalar_mul(scale));
                }
                if let Some(bg) = dense.bias_grad.take() {
                    dense.bias_grad = Some(bg.scalar_mul(scale));
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // OPTIMIZER STEP
    //
    // Reads weight_grad/bias_grad from each Dense layer and asks
    // the optimizer to apply them to the actual weights.
    // Only Dense layers have gradients -- Dropout layers are skipped
    // automatically since as_dense_mut() returns None for them.
    // -----------------------------------------------------------------
    fn optimize(&mut self) {
        let lr = self.learning_rate;
        let mut dense_idx = 0; // index over Dense layers only
        for layer in self.layers.iter_mut() {
            if let Some(dense) = layer.as_dense_mut() {
                if let (Some(wg), Some(bg)) = (
                    dense.weight_grad.clone(),
                    dense.bias_grad.clone(),
                ) {
                    self.optimizer.step(&mut dense.weights, &wg, dense_idx, false, lr);
                    self.optimizer.step(&mut dense.biases,  &bg, dense_idx, true,  lr);
                }
                dense_idx += 1; // only advance for Dense layers
            }
        }
    }

    // -----------------------------------------------------------------
    // FISHER-YATES SHUFFLE
    //
    // Shuffles images and labels together in the same random order
    // so each image stays paired with its correct label.
    // -----------------------------------------------------------------
    fn shuffle(images: &mut Vec<Matrix>, labels: &mut Vec<usize>, rng: &mut Rng) {
        let n = images.len();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            images.swap(i, j);
            labels.swap(i, j);
        }
    }

    // -----------------------------------------------------------------
    // BUILD BATCH MATRIX
    //
    // Packs images[start..end] (each an (input_size x 1) column)
    // into ONE (input_size x batch_size) matrix, one column per
    // sample. This is what makes "one matmul per batch" possible.
    // -----------------------------------------------------------------
    fn build_batch(images: &[Matrix], start: usize, end: usize) -> Matrix {
        let input_size = images[start].rows;
        let batch_size = end - start;
        let mut data   = vec![0.0; input_size * batch_size];

        for (col, idx) in (start..end).enumerate() {
            for row in 0..input_size {
                data[row * batch_size + col] = images[idx].get(row, 0);
            }
        }
        Matrix::from_vec(input_size, batch_size, data)
    }

    // -----------------------------------------------------------------
    // ARGMAX (single column vector, cols=1)
    //
    // Returns index of the highest value -- which class the network
    // is most confident about for one sample.
    // -----------------------------------------------------------------
    pub fn argmax(output: &Matrix) -> usize {
        output.data.iter()
            .enumerate()
            .fold((0, f64::MIN), |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            })
            .0
    }

    // -----------------------------------------------------------------
    // ARGMAX for one column of a BATCHED prediction matrix
    // (n_classes x batch_size) -- used inside the training loop where
    // predictions for the whole batch come back as one matrix.
    // -----------------------------------------------------------------
    pub fn argmax_col(m: &Matrix, col: usize) -> usize {
        let mut best_i = 0;
        let mut best_v = f64::MIN;
        for r in 0..m.rows {
            let v = m.get(r, col);
            if v > best_v {
                best_v = v;
                best_i = r;
            }
        }
        best_i
    }

    // -----------------------------------------------------------------
    // TRAINING LOOP
    //
    // Per epoch:
    //   shuffle training data
    //   set_training_mode(true)   <- enable dropout
    //   for each batch:
    //     build_batch()           <- pack samples into one matrix
    //     forward()               <- ONE call processes whole batch
    //     compute batch loss + accuracy
    //     backward()              <- ONE call computes SUMMED gradients
    //     scale_gradients()       <- turn sum into average
    //     optimize()              <- one optimizer step per layer
    //   set_training_mode(false)  <- disable dropout for validation
    //   evaluate on validation set
    //   print epoch summary
    //   apply lr decay if scheduled
    // -----------------------------------------------------------------
    pub fn train(
        &mut self,
        train_images:  &mut Vec<Matrix>,
        train_labels:  &mut Vec<usize>,
        val_images:    &[Matrix],
        val_labels:    &[usize],
        epochs:        usize,
        batch_size:    usize,
        rng:           &mut Rng,
        decay_every:   usize, // halve lr every N epochs (0 = disabled)
        decay_factor:  f64,   // multiply lr by this on decay (e.g. 0.5)
    ) {
        let n = train_images.len();

        for epoch in 1..=epochs {
            Self::shuffle(train_images, train_labels, rng);

            // Enable dropout for training batches.
            self.set_training_mode(true);

            let mut epoch_loss    = 0.0;
            let mut epoch_correct = 0;
            let mut batches       = 0;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end         = (batch_start + batch_size).min(n);
                let actual_batch_size = batch_end - batch_start;

                // --- BUILD BATCH ---
                let batch_input  = Self::build_batch(
                    train_images, batch_start, batch_end
                );
                let batch_labels = &train_labels[batch_start..batch_end];

                // --- FORWARD (whole batch, one call) ---
                // Dropout is active here (training mode = true).
                let predictions = self.forward(&batch_input);

                // Compute batch loss and accuracy for logging.
                let batch_loss = loss::cross_entropy_batch_loss(
                    &predictions, batch_labels
                );
                let mut batch_correct = 0;
                for (col, &label) in batch_labels.iter().enumerate() {
                    if Self::argmax_col(&predictions, col) == label {
                        batch_correct += 1;
                    }
                }

                // --- BACKWARD (whole batch, one call) ---
                // loss_grad: (n_classes x batch_size), one gradient
                // column per sample -- passed into the last layer's
                // backward() as the starting signal.
                let loss_grad = loss::cross_entropy_derivative_batch(
                    &predictions, batch_labels
                );
                self.backward(&loss_grad);

                // Gradients from backward() are SUMMED across the
                // batch -- divide to get the average before updating.
                self.scale_gradients(1.0 / actual_batch_size as f64);

                // --- OPTIMIZE (one update per Dense layer) ---
                self.optimize();

                epoch_loss    += batch_loss;
                epoch_correct += batch_correct;
                batches       += 1;
            }

            // Disable dropout before validation evaluation.
            self.set_training_mode(false);

            let train_loss = epoch_loss    / batches as f64;
            let train_acc  = epoch_correct as f64 / n as f64 * 100.0;
            let val_acc    = self.evaluate(val_images, val_labels);

            Self::print_progress(epoch, epochs, train_loss, train_acc, val_acc);

            // --- LEARNING RATE DECAY ---
            // Halves lr every `decay_every` epochs (if enabled).
            // Smaller lr in later epochs = finer weight adjustments
            // that settle into the loss minimum rather than oscillating.
            if decay_every > 0 && epoch % decay_every == 0 {
                self.learning_rate *= decay_factor;
                println!("  lr decayed -> {:.6}", self.learning_rate);
            }
        }
    }

    // -----------------------------------------------------------------
    // EVALUATION
    //
    // Forward pass only -- NO gradient computation, NO weight updates.
    // Dropout disabled (set_training_mode(false)) so all neurons are
    // active and outputs are deterministic.
    // Used after each epoch (validation) and once at the end (test).
    // -----------------------------------------------------------------
    pub fn evaluate(&mut self, images: &[Matrix], labels: &[usize]) -> f64 {
        // Always ensure dropout is off during evaluation.
        self.set_training_mode(false);

        let correct = images.iter()
            .zip(labels.iter())
            .filter(|(img, &label)| {
                let pred = self.forward(img);
                Self::argmax(&pred) == label
            })
            .count();
        correct as f64 / images.len() as f64 * 100.0
    }

    // -----------------------------------------------------------------
    // ASCII TRAINING CURVE
    //
    // One line per epoch: progress bar + loss + train/val accuracy.
    // -----------------------------------------------------------------
    fn print_progress(
        epoch:     usize,
        total:     usize,
        loss:      f64,
        train_acc: f64,
        val_acc:   f64,
    ) {
        let bar_width = 20;
        let filled    = (epoch * bar_width) / total;
        let bar: String = (0..bar_width)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();
        println!(
            "Epoch {:>3}/{} [{}] loss: {:.4}  train: {:.2}%  val: {:.2}%",
            epoch, total, bar, loss, train_acc, val_acc
        );
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Dense, Dropout};
    use crate::activation::ActivationType;
    use crate::optimizer::SGD;
    use crate::rng::Rng;

    fn make_small_network() -> Network {
        let mut rng = Rng::new(42);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8,  ActivationType::Sigmoid,      &mut rng)));
        net.add(Box::new(Dense::new(8, 3,  ActivationType::OutputSoftmax, &mut rng)));
        net
    }

    fn make_network_with_dropout() -> Network {
        let mut rng = Rng::new(42);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8,   ActivationType::Sigmoid,      &mut rng)));
        net.add(Box::new(Dropout::new(0.2, 8, 99)));
        net.add(Box::new(Dense::new(8, 3,   ActivationType::OutputSoftmax, &mut rng)));
        net
    }

    #[test]
    fn test_output_size_matches_last_layer() {
        let net = make_small_network();
        assert_eq!(net.layers.last().unwrap().output_size(), 3);
    }

    #[test]
    fn test_forward_output_shape() {
        let mut net = make_small_network();
        let input   = Matrix::zeros(4, 1);
        let output  = net.forward(&input);
        assert_eq!(output.rows, net.layers.last().unwrap().output_size());
        assert_eq!(output.cols, 1);
    }

    #[test]
    fn test_forward_output_sums_to_one() {
        let mut net = make_small_network();
        let input   = Matrix::from_vec(4, 1, vec![0.1, 0.5, 0.3, 0.8]);
        let output  = net.forward(&input);
        let sum: f64 = output.data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax output should sum to 1.0, got {}", sum
        );
    }

    #[test]
    fn test_build_batch_packs_columns_correctly() {
        let images = vec![
            Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]),
            Matrix::from_vec(3, 1, vec![4.0, 5.0, 6.0]),
        ];
        let batch = Network::build_batch(&images, 0, 2);
        assert_eq!(batch.rows, 3);
        assert_eq!(batch.cols, 2);
        assert_eq!(batch.get(0, 0), 1.0); // col 0 = first image
        assert_eq!(batch.get(0, 1), 4.0); // col 1 = second image
        assert_eq!(batch.get(2, 1), 6.0);
    }

    #[test]
    fn test_argmax_col() {
        let m = Matrix::from_vec(3, 2, vec![
            0.1, 0.9,
            0.7, 0.05,
            0.2, 0.05,
        ]);
        assert_eq!(Network::argmax_col(&m, 0), 1); // col 0: max at row 1
        assert_eq!(Network::argmax_col(&m, 1), 0); // col 1: max at row 0
    }

    #[test]
    fn test_argmax_correct() {
        let m = Matrix::from_vec(1, 5, vec![0.1, 0.3, 0.8, 0.2, 0.05]);
        assert_eq!(Network::argmax(&m), 2);
    }

    #[test]
    fn test_dropout_disabled_during_evaluate() {
        // evaluate() must call set_training_mode(false) internally.
        // With training mode off, dropout is transparent -- two
        // evaluate() calls on the same input must give the same result.
        let mut net    = make_network_with_dropout();
        let images     = vec![Matrix::from_vec(4, 1, vec![0.1, 0.2, 0.3, 0.4])];
        let labels     = vec![0usize];

        let acc1 = net.evaluate(&images, &labels);
        let acc2 = net.evaluate(&images, &labels);

        assert_eq!(acc1, acc2,
            "evaluate() must be deterministic (dropout off): {} vs {}", acc1, acc2);
    }

    #[test]
    fn test_set_training_mode_propagates_to_dropout() {
        // Directly confirm set_training_mode reaches the Dropout layer.
        let mut net = make_network_with_dropout();
        let input   = Matrix::from_vec(4, 1, vec![1.0; 4]);

        // Training mode: two forward passes may differ (dropout random).
        net.set_training_mode(true);
        let out1 = net.forward(&input);
        let out2 = net.forward(&input);
        // They MIGHT differ (not guaranteed with small network/lucky seed)
        // -- we just confirm no panic and outputs are valid.
        let sum1: f64 = out1.data.iter().sum();
        let sum2: f64 = out2.data.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);
        assert!((sum2 - 1.0).abs() < 1e-6);

        // Inference mode: must be deterministic.
        net.set_training_mode(false);
        let out3 = net.forward(&input);
        let out4 = net.forward(&input);
        assert_eq!(out3.data, out4.data,
            "inference mode must be deterministic");
    }

    #[test]
    fn test_loss_decreases_after_training_with_dropout() {
        // Confirms the full pipeline (Dense + Dropout) still learns.
        let mut rng = Rng::new(1);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8,  ActivationType::Sigmoid,      &mut rng)));
        net.add(Box::new(Dropout::new(0.1, 8, 77)));
        net.add(Box::new(Dense::new(8, 3,  ActivationType::OutputSoftmax, &mut rng)));

        let mut images: Vec<Matrix> = (0..9)
            .map(|i| Matrix::from_vec(4, 1, vec![
                (i as f64) * 0.1,
                (i as f64) * 0.2,
                (i as f64) * 0.15,
                (i as f64) * 0.05,
            ]))
            .collect();
        let mut labels: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];

        // Loss BEFORE training (inference mode for fair measurement).
        net.set_training_mode(false);
        let initial_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 9.0;

        net.train(
            &mut images, &mut labels,
            &[], &[],
            50, 9, &mut rng,
            0, 1.0, // no decay
        );

        // Loss AFTER training (inference mode).
        net.set_training_mode(false);
        let final_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 9.0;

        assert!(
            final_loss < initial_loss,
            "loss must decrease: initial={:.4} final={:.4}",
            initial_loss, final_loss
        );
    }

    #[test]
    fn test_lr_decay_reduces_learning_rate() {
        // Confirm decay_every + decay_factor actually shrinks lr.
        let mut rng = Rng::new(1);
        let opt     = SGD::new(1, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 3, ActivationType::OutputSoftmax, &mut rng)));

        let initial_lr = net.learning_rate;
        let mut images: Vec<Matrix> = (0..3)
            .map(|_| Matrix::zeros(4, 1))
            .collect();
        let mut labels = vec![0usize, 1, 2];

        // Train for 10 epochs with decay_every=10, decay_factor=0.5.
        // After 10 epochs lr should be halved.
        net.train(
            &mut images, &mut labels,
            &[], &[],
            10, 3, &mut rng,
            10, 0.5, // decay at epoch 10
        );

        assert!(
            (net.learning_rate - initial_lr * 0.5).abs() < 1e-9,
            "lr should be halved after decay: expected {}, got {}",
            initial_lr * 0.5, net.learning_rate
        );
    }

    #[test]
    fn test_evaluate_returns_percentage_in_range() {
        let mut net = make_small_network();
        let images: Vec<Matrix> = (0..10)
            .map(|_| Matrix::zeros(4, 1))
            .collect();
        let labels: Vec<usize> = vec![0; 10];
        let acc = net.evaluate(&images, &labels);
        assert!(
            acc >= 0.0 && acc <= 100.0,
            "accuracy must be 0-100%, got {}", acc
        );
    }
}