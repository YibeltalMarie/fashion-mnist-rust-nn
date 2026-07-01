// =====================================================================
// network.rs
//
// The conductor: holds all layers and the optimizer, runs the
// complete training loop (forward, backward, optimize), and
// evaluates validation accuracy after each epoch.
//
// RESPONSIBILITY BOUNDARY:
// network.rs orchestrates -- it calls layer.forward(), layer.backward(),
// optimizer.step(), and loss.rs functions. It does NOT implement any
// of those operations itself.
//
// ARCHITECTURE (built in main.rs, passed in here):
//   Input (784)
//     -> Dense(784->256, Sigmoid)
//     -> Dense(256->128, Sigmoid)
//     -> Dense(128->10,  OutputSoftmax)
// =====================================================================

use crate::matrix::Matrix;
use crate::layer::{Layer, Dense};
use crate::activation::ActivationType;
use crate::optimizer::Optimizer;
use crate::loss;
use crate::rng::Rng;

pub struct Network {
    // Box<dyn Layer>: heap-allocated, type-erased layers. The Vec
    // can hold any mix of layer types as long as they impl Layer.
    layers: Vec<Box<dyn Layer>>,

    // Box<dyn Optimizer>: same pattern -- SGD or Adam, decided at
    // construction time, swappable without changing this file.
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

    /// Adds a layer to the end of the network.
    /// Box::new() moves the layer onto the heap and gives us a
    /// fixed-size pointer, making it storable in Vec<Box<dyn Layer>>.
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    // -----------------------------------------------------------------
    // FORWARD PASS
    //
    // Feeds input through every layer left to right.
    // Returns the final output (softmax probabilities, shape 10x1).
    // -----------------------------------------------------------------
    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        // We need to thread the output of one layer into the input
        // of the next. `current` holds the most recent output,
        // starting with the raw input.
        let mut current = input.clone();

        for layer in self.layers.iter_mut() {
            // iter_mut() gives &mut Box<dyn Layer> for each element.
            // layer.forward() calls the trait method -- Rust looks up
            // which concrete type this is at runtime ("dynamic dispatch")
            // and calls its forward() implementation.
            current = layer.forward(&current);
        }

        current // final output: softmax probabilities, shape (10 x 1)
    }

    // -----------------------------------------------------------------
    // BACKWARD PASS
    //
    // Feeds the loss gradient back through layers RIGHT TO LEFT
    // (reverse order -- this is what "back" in backpropagation means).
    // -----------------------------------------------------------------
    fn backward(&mut self, loss_grad: &Matrix) {
        let mut grad = loss_grad.clone();

        // .iter_mut().rev() iterates the layers in reverse order.
        // rev() is a standard iterator adapter -- no extra data
        // structures needed, it just walks the iterator backwards.
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    // -----------------------------------------------------------------
    // OPTIMIZER STEP
    //
    // After backward() has populated each layer's weight_grad and
    // bias_grad, this walks every Dense layer and tells the optimizer
    // to apply those gradients to update the actual weights.
    // -----------------------------------------------------------------
    fn optimize(&mut self) {
        let lr = self.learning_rate;

        for (idx, layer) in self.layers.iter_mut().enumerate() {
            // We need to downcast Box<dyn Layer> to &mut Dense to access
            // weight_grad/bias_grad fields. `as_any_mut()` + downcast
            // is the standard Rust pattern for this.
            // To keep this file simple, we use a helper on Dense directly.
            // In practice: network.rs knows it only contains Dense layers
            // for this project, so we use a cast via our DenseLayer trait.
            if let Some(dense) = layer.as_dense_mut() {
                if let (Some(wg), Some(bg)) = (
                    dense.weight_grad.as_ref(),
                    dense.bias_grad.as_ref()
                ) {
                    let wg = wg.clone();
                    let bg = bg.clone();
                    self.optimizer.step(&mut dense.weights, &wg, idx, false, lr);
                    self.optimizer.step(&mut dense.biases, &bg, idx, true, lr);
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // FISHER-YATES SHUFFLE
    //
    // Shuffles two Vecs (images, labels) together in the same order,
    // so each image stays paired with its correct label after shuffling.
    // -----------------------------------------------------------------
    fn shuffle(images: &mut Vec<Matrix>, labels: &mut Vec<usize>, rng: &mut Rng) {
        let n = images.len();
        for i in (1..n).rev() {
            // Random index j in [0, i] inclusive.
            let j = (rng.next_u64() as usize) % (i + 1);
            images.swap(i, j);
            labels.swap(i, j);
        }
    }

    // -----------------------------------------------------------------
    // TRAINING LOOP
    // -----------------------------------------------------------------
    pub fn train(
        &mut self,
        train_images: &mut Vec<Matrix>,
        train_labels: &mut Vec<usize>,
        val_images: &[Matrix],
        val_labels: &[usize],
        epochs: usize,
        batch_size: usize,
        rng: &mut Rng,
    ) {
        let n = train_images.len();

        for epoch in 1..=epochs {
            // Shuffle training data at the start of every epoch --
            // prevents the network from memorizing sample order.
            Self::shuffle(train_images, train_labels, rng);

            let mut epoch_loss = 0.0;
            let mut correct = 0;
            let mut batches = 0;

            // Process one mini-batch at a time.
            // (0..n).step_by(batch_size) gives 0, batch_size,
            // 2*batch_size, ... -- the starting index of each batch.
            for batch_start in (0..n).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n);
                let actual_batch_size = batch_end - batch_start;

                // Accumulate loss gradients across the whole batch,
                // then average before backward -- makes learning rate
                // independent of batch size.
                let output_size = self.layers.last().unwrap().output_size();
                let mut avg_grad = Matrix::zeros(output_size, 1);
                let mut batch_loss = 0.0;

                for i in batch_start..batch_end {
                    let prediction = self.forward(&train_images[i]);
                    let label = train_labels[i];

                    // Accumulate loss for logging.
                    batch_loss += loss::cross_entropy(&prediction, label);

                    // Accumulate gradient.
                    let grad = loss::cross_entropy_derivative(&prediction, label, 10);
                    avg_grad = avg_grad.add(&grad);

                    // Track training accuracy.
                    if Self::argmax(&prediction) == label {
                        correct += 1;
                    }
                }

                // Average gradient over batch size.
                avg_grad = avg_grad.scalar_mul(1.0 / actual_batch_size as f64);
                epoch_loss += batch_loss / actual_batch_size as f64;
                batches += 1;

                // Backward pass with averaged gradient.
                self.backward(&avg_grad);

                // Apply optimizer to update all layer weights.
                self.optimize();
            }

            let train_loss = epoch_loss / batches as f64;
            let train_acc = correct as f64 / n as f64 * 100.0;
            let val_acc = self.evaluate(val_images, val_labels);

            // Print live training curve -- one line per epoch.
            Self::print_progress(epoch, epochs, train_loss, train_acc, val_acc);
        }
    }

    // -----------------------------------------------------------------
    // EVALUATION (validation / test set)
    //
    // Runs forward pass only -- NO gradient computation, NO weight
    // updates. Used after each epoch (validation) and at the very end
    // (test set final accuracy).
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
    // ARGMAX
    //
    // Returns the index of the highest value in a (10x1) Matrix --
    // i.e., which class the network is most confident about.
    // -----------------------------------------------------------------
    pub fn argmax(output: &Matrix) -> usize {
        output.data.iter()
            .enumerate()
            .fold((0, f64::MIN), |(best_idx, best_val), (idx, &val)| {
                if val > best_val { (idx, val) } else { (best_idx, best_val) }
            })
            .0 // .0 extracts the first element of the tuple (the index)
    }

    // -----------------------------------------------------------------
    // ASCII TRAINING CURVE
    //
    // Prints one line per epoch showing loss, train accuracy, and
    // validation accuracy. The progress bar fills as epochs advance.
    // -----------------------------------------------------------------
    fn print_progress(epoch: usize, total: usize, loss: f64, train_acc: f64, val_acc: f64) {
        let bar_width = 20;
        let filled = (epoch * bar_width) / total;
        let bar: String = (0..bar_width)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();

        println!(
            "Epoch {:>3}/{} [{}] loss: {:.4}  train: {:.2}%  val: {:.2}%",
            epoch, total, bar, loss, train_acc, val_acc
        );
    }
}

// -----------------------------------------------------------------
// HELPER TRAIT: lets network.rs access Dense fields through
// Box<dyn Layer> without a full dynamic dispatch / downcast library.
// We add as_dense_mut() to the Layer trait so Dense can opt in.
// -----------------------------------------------------------------
// Add this to layer.rs's Layer trait:
//
//   fn as_dense_mut(&mut self) -> Option<&mut Dense> { None }
//
// And override in Dense's impl Layer:
//
//   fn as_dense_mut(&mut self) -> Option<&mut Dense> { Some(self) }
// -----------------------------------------------------------------

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::SGD;
    use crate::rng::Rng;

    fn make_network() -> Network {
        let mut rng = Rng::new(42);
        let n_layers = 3;
        let opt = SGD::new(n_layers, 0.0);
        let mut net = Network::new(Box::new(opt), 0.01);

        net.add(Box::new(Dense::new(4, 8, ActivationType::Sigmoid, &mut rng)));
        net.add(Box::new(Dense::new(8, 4, ActivationType::Sigmoid, &mut rng)));
        net.add(Box::new(Dense::new(4, 3, ActivationType::OutputSoftmax, &mut rng)));
        net
    }

    #[test]
    fn test_forward_output_shape() {
        let mut net = make_network();
        let input = Matrix::zeros(4, 1);
        let output = net.forward(&input);
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 1);
    }

    #[test]
    fn test_forward_output_sums_to_one() {
        let mut net = make_network();
        let input = Matrix::from_vec(4, 1, vec![0.1, 0.5, 0.3, 0.8]);
        let output = net.forward(&input);
        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6,
            "softmax output should sum to 1.0, got {}", sum);
    }

    #[test]
    fn test_argmax_correct() {
        let m = Matrix::from_vec(1, 5, vec![0.1, 0.3, 0.8, 0.2, 0.05]);
        assert_eq!(Network::argmax(&m), 2);
    }

    #[test]
    fn test_loss_decreases_after_training() {
        // Trains a tiny network for a few steps and confirms loss
        // goes down -- the most fundamental sanity check.
        let mut rng = Rng::new(1);
        let opt = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 8, ActivationType::Sigmoid, &mut rng)));
        net.add(Box::new(Dense::new(8, 3, ActivationType::OutputSoftmax, &mut rng)));

        // Simple dataset: 6 samples, 4 features, 3 classes.
        let mut images: Vec<Matrix> = (0..6)
            .map(|i| Matrix::from_vec(4, 1, vec![
                (i as f64) * 0.1,
                (i as f64) * 0.2,
                (i as f64) * 0.15,
                (i as f64) * 0.05,
            ]))
            .collect();
        let mut labels: Vec<usize> = vec![0, 1, 2, 0, 1, 2];

        // Compute initial loss.
        let initial_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 6.0;

        // Train for 20 steps.
        net.train(&mut images, &mut labels, &[], &[], 20, 6, &mut rng);

        // Compute loss after training.
        let final_loss: f64 = images.iter()
            .zip(labels.iter())
            .map(|(img, &lbl)| {
                let pred = net.forward(img);
                loss::cross_entropy(&pred, lbl)
            })
            .sum::<f64>() / 6.0;

        assert!(final_loss < initial_loss,
            "loss should decrease after training: initial={:.4} final={:.4}",
            initial_loss, final_loss);
    }

    #[test]
    fn test_evaluate_returns_percentage() {
        let mut net = make_network();
        let images: Vec<Matrix> = (0..10)
            .map(|_| Matrix::zeros(4, 1))
            .collect();
        let labels: Vec<usize> = vec![0; 10];
        let acc = net.evaluate(&images, &labels);
        assert!(acc >= 0.0 && acc <= 100.0,
            "accuracy should be percentage 0-100, got {}", acc);
    }
}