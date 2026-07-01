// =====================================================================
// network.rs
//
// The conductor: holds all layers and the optimizer, runs the
// complete training loop, evaluates validation accuracy each epoch.
//
// BATCHED TRAINING (new): instead of looping over each sample in a
// batch individually, the whole batch is assembled into ONE matrix
// (input_size x batch_size) and pushed through forward()/backward()
// ONCE. Each layer's forward/backward internally does one big matmul
// (parallelized across CPU cores via matmul_parallel) instead of many
// small ones. This is a fundamentally different, much faster design
// than the old per-sample loop + gradient accumulation.
//
// GRADIENT AVERAGING: Dense::backward() returns gradients SUMMED
// across the batch (that's what matmul naturally produces). We divide
// by batch_size here (scale_gradients) before calling the optimizer,
// so the effective learning rate stays independent of batch size.
//
// ARCHITECTURE (built in main.rs, passed in here):
//   Input (784)
//     -> Dense(784->256, Sigmoid)
//     -> Dense(256->128, Sigmoid)
//     -> Dense(128->10,  OutputSoftmax)
// =====================================================================

use crate::matrix::Matrix;
use crate::layer::{Layer, Dense};
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
    // FORWARD PASS
    //
    // Works uniformly whether `input` is one sample (cols=1, used by
    // evaluate()) or a whole batch (cols=batch_size, used by train()).
    // Each layer's forward() overwrites its own cache -- fine here
    // because we always immediately follow with backward() on the
    // same batch before calling forward() again.
    // -----------------------------------------------------------------
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in self.layers.iter_mut() {
            current = layer.forward(&current);
        }
        current
    }

    pub fn forward_single(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    // -----------------------------------------------------------------
    // BACKWARD PASS
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
    // Dense::backward() sums gradients across the batch (a natural
    // side effect of matmul). This divides every layer's stored
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
    // -----------------------------------------------------------------
    fn optimize(&mut self) {
        let lr = self.learning_rate;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            if let Some(dense) = layer.as_dense_mut() {
                if let (Some(wg), Some(bg)) = (
                    dense.weight_grad.clone(),
                    dense.bias_grad.clone(),
                ) {
                    self.optimizer.step(&mut dense.weights, &wg, idx, false, lr);
                    self.optimizer.step(&mut dense.biases,  &bg, idx, true,  lr);
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // FISHER-YATES SHUFFLE
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
    // Packs images[start..end] (each an (input_size x 1) column) into
    // ONE (input_size x batch_size) matrix, one column per sample.
    // This is what makes the "one matmul per batch" design possible.
    // -----------------------------------------------------------------
    fn build_batch(images: &[Matrix], start: usize, end: usize) -> Matrix {
        let input_size = images[start].rows;
        let batch_size = end - start;
        let mut data = vec![0.0; input_size * batch_size];

        for (col, idx) in (start..end).enumerate() {
            for row in 0..input_size {
                data[row * batch_size + col] = images[idx].get(row, 0);
            }
        }
        Matrix::from_vec(input_size, batch_size, data)
    }

    // -----------------------------------------------------------------
    // ARGMAX (single column, e.g. predictions matrix with cols=1)
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
    // TRAINING LOOP (batched)
    //
    // Per batch:
    //   1. build_batch()   -- pack samples into one matrix
    //   2. forward()       -- ONE call processes the whole batch
    //   3. compute batch loss + accuracy
    //   4. backward()      -- ONE call computes SUMMED gradients
    //   5. scale_gradients -- turn sum into average
    //   6. optimize()      -- one optimizer step per layer
    // -----------------------------------------------------------------
    pub fn train(
        &mut self,
        train_images: &mut Vec<Matrix>,
        train_labels: &mut Vec<usize>,
        val_images:   &[Matrix],
        val_labels:   &[usize],
        epochs:       usize,
        batch_size:   usize,
        rng:          &mut Rng,
    ) {
        let n = train_images.len();

        for epoch in 1..=epochs {
            Self::shuffle(train_images, train_labels, rng);

            let mut epoch_loss    = 0.0;
            let mut epoch_correct = 0;
            let mut batches       = 0;

            for batch_start in (0..n).step_by(batch_size) {
                let batch_end         = (batch_start + batch_size).min(n);
                let actual_batch_size = batch_end - batch_start;

                // --- BUILD BATCH ---
                let batch_input  = Self::build_batch(train_images, batch_start, batch_end);
                let batch_labels = &train_labels[batch_start..batch_end];

                // --- FORWARD (whole batch, one call) ---
                let predictions = self.forward(&batch_input); // (n_classes x batch)

                let batch_loss = loss::cross_entropy_batch_loss(&predictions, batch_labels);
                let mut batch_correct = 0;
                for (col, &label) in batch_labels.iter().enumerate() {
                    if Self::argmax_col(&predictions, col) == label {
                        batch_correct += 1;
                    }
                }

                // --- BACKWARD (whole batch, one call) ---
                let loss_grad = loss::cross_entropy_derivative_batch(&predictions, batch_labels);
                self.backward(&loss_grad);

                // Gradients from backward() are SUMMED across the
                // batch -- divide to get the average before updating.
                self.scale_gradients(1.0 / actual_batch_size as f64);

                // --- OPTIMIZE (one update per layer) ---
                self.optimize();

                epoch_loss    += batch_loss;
                epoch_correct += batch_correct;
                batches       += 1;
            }

            let train_loss = epoch_loss    / batches as f64;
            let train_acc  = epoch_correct as f64 / n as f64 * 100.0;
            let val_acc    = self.evaluate(val_images, val_labels);

            Self::print_progress(epoch, epochs, train_loss, train_acc, val_acc);
        }
    }

    // -----------------------------------------------------------------
    // EVALUATION
    //
    // Still one sample at a time (cols=1) -- correctness matters more
    // than speed here, and forward() works identically for batch_size=1.
    // -----------------------------------------------------------------
    pub fn evaluate(&mut self, images: &[Matrix], labels: &[usize]) -> f64 {
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
    use crate::layer::Dense;
    use crate::activation::ActivationType;
    use crate::optimizer::SGD;
    use crate::rng::Rng;

    fn make_small_network() -> Network {
        let mut rng = Rng::new(42);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8, ActivationType::Sigmoid,      &mut rng)));
        net.add(Box::new(Dense::new(8, 3, ActivationType::OutputSoftmax, &mut rng)));
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
        assert!((sum - 1.0).abs() < 1e-6, "softmax output should sum to 1.0, got {}", sum);
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
        // column 0 = first image, column 1 = second image
        assert_eq!(batch.get(0, 0), 1.0);
        assert_eq!(batch.get(0, 1), 4.0);
        assert_eq!(batch.get(2, 1), 6.0);
    }

    #[test]
    fn test_argmax_col() {
        let m = Matrix::from_vec(3, 2, vec![
            0.1, 0.9,
            0.7, 0.05,
            0.2, 0.05,
        ]);
        assert_eq!(Network::argmax_col(&m, 0), 1); // col 0: max at row 1 (0.7)
        assert_eq!(Network::argmax_col(&m, 1), 0); // col 1: max at row 0 (0.9)
    }

    #[test]
    fn test_argmax_correct() {
        let m = Matrix::from_vec(1, 5, vec![0.1, 0.3, 0.8, 0.2, 0.05]);
        assert_eq!(Network::argmax(&m), 2);
    }

    #[test]
    fn test_loss_decreases_after_training() {
        // Confirms the BATCHED training loop is still mathematically
        // correct -- loss MUST decrease.
        let mut rng = Rng::new(1);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8, ActivationType::Sigmoid,      &mut rng)));
        net.add(Box::new(Dense::new(8, 3, ActivationType::OutputSoftmax, &mut rng)));

        let mut images: Vec<Matrix> = (0..9)
            .map(|i| Matrix::from_vec(4, 1, vec![
                (i as f64) * 0.1,
                (i as f64) * 0.2,
                (i as f64) * 0.15,
                (i as f64) * 0.05,
            ]))
            .collect();
        let mut labels: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];

        let initial_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 9.0;

        net.train(&mut images, &mut labels, &[], &[], 50, 3, &mut rng); // batch_size=3

        let final_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 9.0;

        assert!(
            final_loss < initial_loss,
            "loss must decrease after training: initial={:.4} final={:.4}",
            initial_loss, final_loss
        );
    }

    #[test]
    fn test_evaluate_returns_percentage_in_range() {
        let mut net = make_small_network();
        let images: Vec<Matrix> = (0..10).map(|_| Matrix::zeros(4, 1)).collect();
        let labels: Vec<usize>  = vec![0; 10];
        let acc = net.evaluate(&images, &labels);
        assert!(acc >= 0.0 && acc <= 100.0, "accuracy must be 0-100%, got {}", acc);
    }
}