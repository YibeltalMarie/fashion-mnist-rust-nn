// =====================================================================
// metrics.rs
//
// Builds a confusion matrix from predictions vs actual labels, and
// reports per-class accuracy (recall). Used after training to give
// a detailed breakdown beyond a single overall accuracy percentage --
// exposes WHICH classes the model confuses, not just how often it's
// wrong overall.
//
// CONFUSION MATRIX LAYOUT:
//   rows    = actual class
//   columns = predicted class
//   cell[actual][predicted] = count of that outcome
//
// Storage: flat Vec<usize> of size n_classes*n_classes, same
// row*cols+col indexing trick as Matrix -- but holds integer counts,
// not f64 values, so it's a separate, purpose-built type.
// =====================================================================

pub struct ConfusionMatrix {
    n_classes: usize,
    counts: Vec<usize>, // flat: counts[actual * n_classes + predicted]
}

impl ConfusionMatrix {
    /// Builds a confusion matrix from parallel slices of predictions
    /// and actual labels (same length, same ordering).
    pub fn build(predictions: &[usize], actuals: &[usize], n_classes: usize) -> Self {
        assert_eq!(
            predictions.len(), actuals.len(),
            "predictions and actuals must be the same length"
        );

        let mut counts = vec![0usize; n_classes * n_classes];

        for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
            counts[actual * n_classes + pred] += 1;
        }

        ConfusionMatrix { n_classes, counts }
    }

    /// Count of samples where actual == actual_class and
    /// predicted == predicted_class.
    pub fn get(&self, actual_class: usize, predicted_class: usize) -> usize {
        self.counts[actual_class * self.n_classes + predicted_class]
    }

    /// Total number of actual samples in a given class (row sum).
    fn row_total(&self, actual_class: usize) -> usize {
        (0..self.n_classes)
            .map(|pred| self.get(actual_class, pred))
            .sum()
    }

    /// Per-class accuracy (recall): of all samples truly in this
    /// class, what fraction did we correctly predict?
    /// Returns 0.0 for a class with zero actual samples (avoids
    /// division by zero) rather than panicking.
    pub fn per_class_accuracy(&self, class: usize) -> f64 {
        let total = self.row_total(class);
        if total == 0 {
            return 0.0;
        }
        self.get(class, class) as f64 / total as f64 * 100.0
    }

    /// Overall accuracy across all classes -- sum of diagonal
    /// divided by total sample count. Should match the accuracy
    /// computed elsewhere (e.g. Network::evaluate) as a sanity check.
    pub fn overall_accuracy(&self) -> f64 {
        let correct: usize = (0..self.n_classes)
            .map(|i| self.get(i, i))
            .sum();
        let total: usize = self.counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64 * 100.0
    }

    /// Prints the confusion matrix as an aligned ASCII table.
    /// Rows = actual class, columns = predicted class.
    /// Class names are truncated/padded to keep columns aligned --
    /// full 10-class Fashion-MNIST names would make the table too wide.
    pub fn print(&self, class_names: &[&str]) {
        println!("\nConfusion Matrix (rows=actual, cols=predicted):");

        // Header row: short column labels (class index).
        print!("{:>14}", "");
        for j in 0..self.n_classes {
            print!("{:>6}", j);
        }
        println!();

        for i in 0..self.n_classes {
            // Row label: truncate class name to fit, right-aligned.
            let label = if i < class_names.len() { class_names[i] } else { "?" };
            let short: String = label.chars().take(13).collect();
            print!("{:>14}", short);

            for j in 0..self.n_classes {
                let count = self.get(i, j);
                if i == j {
                    // Highlight the diagonal (correct predictions)
                    // with brackets so it's easy to scan visually.
                    print!("{:>5}*", count);
                } else {
                    print!("{:>6}", count);
                }
            }
            println!();
        }
        println!();
    }

    /// Prints per-class accuracy, sorted from WORST to BEST --
    /// puts the model's weakest classes at the top, which is the
    /// most useful ordering for spotting problems at a glance.
    pub fn print_per_class_accuracy(&self, class_names: &[&str]) {
        println!("Per-Class Accuracy (sorted worst to best):");

        let mut rows: Vec<(usize, f64)> = (0..self.n_classes)
            .map(|i| (i, self.per_class_accuracy(i)))
            .collect();

        // Sort by accuracy ascending -- worst classes first.
        // partial_cmp is needed (not cmp) because f64 doesn't
        // implement Ord (NaN makes total ordering undefined) --
        // .unwrap() is safe here since accuracy is never NaN.
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (class_idx, acc) in rows {
            let name = if class_idx < class_names.len() {
                class_names[class_idx]
            } else {
                "?"
            };
            println!("  {:<14} {:>6.2}%", name, acc);
        }
        println!();
    }
}

// -----------------------------------------------------------------
// FREE FUNCTIONS -- thin wrappers so main.rs's existing call sites
// (metrics::print_confusion_matrix(...)) keep working without
// needing to construct a ConfusionMatrix explicitly every time.
// -----------------------------------------------------------------

pub fn print_confusion_matrix(
    predictions: &[usize],
    actuals: &[usize],
    class_names: &[&str],
) {
    let matrix = ConfusionMatrix::build(predictions, actuals, class_names.len());
    matrix.print(class_names);
}

pub fn print_per_class_accuracy(
    predictions: &[usize],
    actuals: &[usize],
    class_names: &[&str],
) {
    let matrix = ConfusionMatrix::build(predictions, actuals, class_names.len());
    matrix.print_per_class_accuracy(class_names);
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions_all_diagonal() {
        // Every prediction correct -- confusion matrix should be
        // purely diagonal, overall accuracy 100%.
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let actuals      = vec![0, 1, 2, 0, 1, 2];

        let cm = ConfusionMatrix::build(&predictions, &actuals, 3);

        assert_eq!(cm.get(0, 0), 2);
        assert_eq!(cm.get(1, 1), 2);
        assert_eq!(cm.get(2, 2), 2);
        // Off-diagonal should all be zero.
        assert_eq!(cm.get(0, 1), 0);
        assert_eq!(cm.get(1, 2), 0);

        assert_eq!(cm.overall_accuracy(), 100.0);
    }

    #[test]
    fn test_confusion_matrix_counts_specific_mistakes() {
        // actual=0, predicted=1 twice -- confusion[0][1] should be 2.
        let predictions = vec![1, 1, 0];
        let actuals      = vec![0, 0, 0];

        let cm = ConfusionMatrix::build(&predictions, &actuals, 2);

        assert_eq!(cm.get(0, 1), 2); // actual 0, predicted 1 -> happened twice
        assert_eq!(cm.get(0, 0), 1); // actual 0, predicted 0 -> happened once
    }

    #[test]
    fn test_per_class_accuracy() {
        // Class 0: 3 actual samples, 2 correct -> 66.67%
        // Class 1: 2 actual samples, 2 correct -> 100%
        let predictions = vec![0, 0, 1, 1, 1];
        let actuals      = vec![0, 0, 0, 1, 1];

        let cm = ConfusionMatrix::build(&predictions, &actuals, 2);

        let class_0_acc = cm.per_class_accuracy(0);
        let class_1_acc = cm.per_class_accuracy(1);

        assert!((class_0_acc - 66.666).abs() < 0.01);
        assert!((class_1_acc - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_per_class_accuracy_zero_samples_returns_zero() {
        // Class 2 never appears as an actual label -- should return
        // 0.0 rather than panicking on division by zero.
        let predictions = vec![0, 1];
        let actuals      = vec![0, 1];

        let cm = ConfusionMatrix::build(&predictions, &actuals, 3);

        assert_eq!(cm.per_class_accuracy(2), 0.0);
    }

    #[test]
    fn test_overall_accuracy_matches_manual_calculation() {
        // 3 correct out of 5 total -> 60%.
        let predictions = vec![0, 1, 2, 0, 1];
        let actuals      = vec![0, 1, 1, 1, 1];
        // Correct: index 0 (0==0), index 1 (1==1), index 4 (1==1) -> 3 correct
        // Wrong: index 2 (2 vs 1), index 3 (0 vs 1) -> 2 wrong

        let cm = ConfusionMatrix::build(&predictions, &actuals, 3);

        assert!((cm.overall_accuracy() - 60.0).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths_panics() {
        let predictions = vec![0, 1, 2];
        let actuals      = vec![0, 1];
        ConfusionMatrix::build(&predictions, &actuals, 3);
    }

    #[test]
    fn test_print_does_not_panic() {
        // Smoke test -- just confirms print() runs without error
        // for a realistic 10-class case (doesn't check output text).
        let predictions: Vec<usize> = (0..50).map(|i| i % 10).collect();
        let actuals: Vec<usize>     = (0..50).map(|i| (i + 1) % 10).collect();

        let class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ];

        let cm = ConfusionMatrix::build(&predictions, &actuals, 10);
        cm.print(&class_names);
        cm.print_per_class_accuracy(&class_names);
    }
}