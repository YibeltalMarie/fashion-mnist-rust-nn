// TODO (Day 1): Manual CSV parser for fashion-mnist_{train,test}.csv.
// Parse label,pixel1..784 -> normalize to [0,1] -> train/val split (90/10).

// =====================================================================
// data.rs
//
// Loads fashion-mnist_{train,test}.csv from disk, parses each row by
// hand (label,pixel1,...,pixel784), normalizes pixel values to [0,1],
// and provides a train/validation split.
//
// FILE FORMAT (one header row, then 60000 or 10000 data rows):
//   label,pixel1,pixel2,...,pixel784
//   2,0,0,0,3,...
// =====================================================================

use crate::matrix::Matrix;
// `crate::` means "starting from the root of our own project" -- this
// is how files import things from OTHER files in the same project.

pub struct Dataset {
    pub images: Vec<Matrix>, // each entry: a (784 x 1) Matrix, values in [0,1]
    pub labels: Vec<usize>,   // each entry: the correct class, 0-9
}

impl Dataset {
    /// Loads and parses a Fashion-MNIST CSV file from the given path.
    /// Returns a Result so the CALLER decides what to do if the file
    /// is missing or malformed, rather than this function deciding
    /// for them by panicking unconditionally.
    pub fn load_csv(path: &str) -> Result<Dataset, String> {
        // std::fs::read_to_string returns Result<String, io::Error>.
        // We use `match` to convert any io::Error into our own simple
        // String error message (keeps our error type simple and uniform).
        let contents = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => return Err(format!("Could not read {}: {}", path, e)),
        };

        let mut images = Vec::new();
        let mut labels = Vec::new();

        // .lines() splits the text into an iterator over each line,
        // without copying the whole file -- each line is a &str slice
        // pointing into `contents`.
        //
        // .skip(1) skips the header row (label,pixel1,pixel2,...).
        for (line_num, line) in contents.lines().skip(1).enumerate() {
            if line.trim().is_empty() {
                continue; // skip any blank trailing lines
            }

            // .split(',') gives an iterator over comma-separated &str
            // pieces. .collect() gathers them into a Vec<&str>.
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() != 785 {
                return Err(format!(
                    "Row {} has {} fields, expected 785 (1 label + 784 pixels)",
                    line_num + 2, // +2: +1 for header, +1 for 0-indexing
                    fields.len()
                ));
            }

            // Parse the label (first field). .parse::<usize>() returns
            // a Result -- we convert any parse failure into our error type.
            let label: usize = match fields[0].parse() {
                Ok(v) => v,
                Err(_) => return Err(format!("Row {}: invalid label '{}'", line_num + 2, fields[0])),
            };

            // Parse the remaining 784 fields as pixel values, normalizing
            // each by dividing by 255.0 to land in [0.0, 1.0].
            let mut pixels: Vec<f64> = Vec::with_capacity(784);
            for (i, raw) in fields[1..].iter().enumerate() {
                let pixel: f64 = match raw.parse::<f64>() {
                    Ok(v) => v / 255.0,
                    Err(_) => {
                        return Err(format!(
                            "Row {}: invalid pixel value '{}' at position {}",
                            line_num + 2, raw, i
                        ))
                    }
                };
                pixels.push(pixel);
            }

            // Store this image as a (784 x 1) column-vector Matrix --
            // matches the shape our Dense layer's matmul expects.
            images.push(Matrix::from_vec(784, 1, pixels));
            labels.push(label);
        }

        Ok(Dataset { images, labels })
    }

    /// Number of samples in this dataset.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Splits this dataset into (train, validation) using a 90/10 split.
    /// `self` is consumed (moved in) since we're reorganizing its data
    /// into two brand new Datasets, not just reading it.
    pub fn split_train_val(self, val_fraction: f64) -> (Dataset, Dataset) {
        let total = self.images.len();
        let val_size = (total as f64 * val_fraction).round() as usize;
        let train_size = total - val_size;

        // Vec has .split_off(index): splits ONE Vec into two, in place.
        // self.images becomes the first `train_size` elements, and
        // the returned Vec holds everything from `train_size` onward.
        let mut images = self.images;
        let mut labels = self.labels;

        let val_images = images.split_off(train_size);
        let val_labels = labels.split_off(train_size);

        let train_set = Dataset { images, labels };
        let val_set = Dataset { images: val_images, labels: val_labels };

        (train_set, val_set)
    }
}

// =====================================================================
// TESTS
//
// We can't easily test load_csv() against the real 133MB file in a
// fast unit test, so we test against small, hand-written CSV strings
// instead -- this is a common Rust testing pattern: test the LOGIC
// with tiny fake inputs, separately from the real data.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: writes a small temp CSV file and returns its path,
    /// so our tests can call the real load_csv() function end-to-end.
    fn write_temp_csv(filename: &str, contents: &str) -> String {
        let path = format!("/tmp/{}", filename);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_load_valid_csv() {
        let header = "label,".to_string() + &(1..=784).map(|i| format!("pixel{}", i)).collect::<Vec<_>>().join(",");
        let row: String = "3,".to_string() + &vec!["0"; 784].join(",");
        let csv = format!("{}\n{}\n", header, row);

        let path = write_temp_csv("test_valid.csv", &csv);
        let dataset = Dataset::load_csv(&path).unwrap();

        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset.labels[0], 3);
        assert_eq!(dataset.images[0].rows, 784);
        assert_eq!(dataset.images[0].cols, 1);
    }

    #[test]
    fn test_pixel_normalization() {
        let header = "label,".to_string() + &(1..=784).map(|i| format!("pixel{}", i)).collect::<Vec<_>>().join(",");
        let mut pixel_vals = vec!["0"; 784];
        pixel_vals[0] = "255"; // first pixel at max value
        let row = "0,".to_string() + &pixel_vals.join(",");
        let csv = format!("{}\n{}\n", header, row);

        let path = write_temp_csv("test_norm.csv", &csv);
        let dataset = Dataset::load_csv(&path).unwrap();

        // 255 / 255.0 should normalize to exactly 1.0
        assert_eq!(dataset.images[0].get(0, 0), 1.0);
    }

    #[test]
    fn test_missing_file_returns_err() {
        let result = Dataset::load_csv("/tmp/this_file_does_not_exist.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_field_count_returns_err() {
        let csv = "label,pixel1,pixel2\n1,5,10\n"; // only 2 pixels, not 784
        let path = write_temp_csv("test_bad_fields.csv", csv);
        let result = Dataset::load_csv(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_train_val() {
        // Build a tiny fake dataset directly (no CSV needed) to test
        // the splitting logic in isolation.
        let images: Vec<Matrix> = (0..10).map(|_| Matrix::zeros(784, 1)).collect();
        let labels: Vec<usize> = (0..10).collect();
        let dataset = Dataset { images, labels };

        let (train, val) = dataset.split_train_val(0.2); // 20% val

        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }
}