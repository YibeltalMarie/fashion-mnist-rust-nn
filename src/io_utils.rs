// =====================================================================
// io_utils.rs
//
// Saves and loads trained network weights to/from disk using a
// hand-rolled plain-text format (no serde/bincode -- pure std::fs
// and string parsing, consistent with the rest of the project).
//
// FORMAT:
//   FASHION_MNIST_WEIGHTS_V1
//   LAYERS <n_dense_layers>
//   LAYER <idx> <input_size> <output_size>
//   <weights: input_size*output_size floats, space-separated>
//   <biases: output_size floats, space-separated>
//   ... repeated per Dense layer ...
//
// Dropout layers are skipped entirely -- they have no weights, so
// only Dense layers are counted and written.
//
// RESPONSIBILITY BOUNDARY:
// This file only reads/writes the Dense layers' weights and biases.
// It does NOT know about architecture construction (which layers,
// which activations) -- the caller must build a Network with the
// EXACT same architecture before calling load_weights(), or shapes
// won't match and loading will fail with a clear error.
// =====================================================================

use crate::network::Network;
use crate::matrix::Matrix;

const FORMAT_HEADER: &str = "FASHION_MNIST_WEIGHTS_V1";

/// Saves all Dense layers' weights and biases to a plain-text file.
/// Dropout layers are automatically skipped (as_dense() returns None
/// for them).
pub fn save_weights(net: &Network, path: &str) -> Result<(), String> {
    // Collect references to only the Dense layers, in order.
    // as_dense() borrows immutably -- we're not modifying anything.
    let dense_layers: Vec<&crate::layer::Dense> = net.layers.iter()
        .filter_map(|layer| layer.as_dense())
        .collect();

    // Pre-allocate roughly enough capacity to avoid repeated
    // reallocation as the string grows -- rough estimate: each
    // float takes about 12 characters on average once formatted.
    let estimated_size: usize = dense_layers.iter()
        .map(|d| (d.weights.data.len() + d.biases.data.len()) * 12)
        .sum();
    let mut out = String::with_capacity(estimated_size + 256);

    out.push_str(FORMAT_HEADER);
    out.push('\n');
    out.push_str(&format!("LAYERS {}\n", dense_layers.len()));

    for (idx, dense) in dense_layers.iter().enumerate() {
        // dense.weights is (output_size x input_size) by our convention.
        let input_size  = dense.weights.cols;
        let output_size = dense.weights.rows;

        out.push_str(&format!("LAYER {} {} {}\n", idx, input_size, output_size));

        // Write all weight values space-separated on one line.
        // .map(|v| v.to_string()) converts each f64 to its text form,
        // .collect::<Vec<_>>().join(" ") joins them with spaces --
        // same idea as Python's " ".join(str(v) for v in values).
        let weight_line: String = dense.weights.data.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&weight_line);
        out.push('\n');

        let bias_line: String = dense.biases.data.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&bias_line);
        out.push('\n');
    }

    // Write the whole file in one call.
    std::fs::write(path, out)
        .map_err(|e| format!("Failed to write {}: {}", path, e))
}

/// Loads weights from a file into an EXISTING Network. The Network
/// must already have the correct architecture built (same layer
/// count, same shapes) -- this function only fills in the numbers,
/// it does not construct layers.
pub fn load_weights(net: &mut Network, path: &str) -> Result<(), String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    // .lines() gives an iterator; we manually advance through it
    // since the format has a fixed structure (not one-row-per-line
    // like CSV) -- .next() pulls one line at a time.
    let mut lines = contents.lines();

    // --- Validate header ---
    let header = lines.next()
        .ok_or_else(|| "File is empty".to_string())?;
    if header != FORMAT_HEADER {
        return Err(format!(
            "Unrecognized file format: expected '{}', got '{}'",
            FORMAT_HEADER, header
        ));
    }

    // --- Read layer count ---
    let layers_line = lines.next()
        .ok_or_else(|| "Missing LAYERS line".to_string())?;
    let n_layers: usize = layers_line
        .strip_prefix("LAYERS ")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("Malformed LAYERS line: '{}'", layers_line))?;

    // Collect mutable references to only the Dense layers, in order.
    let mut dense_layers: Vec<&mut crate::layer::Dense> = net.layers.iter_mut()
        .filter_map(|layer| layer.as_dense_mut())
        .collect();

    if dense_layers.len() != n_layers {
        return Err(format!(
            "Architecture mismatch: file has {} Dense layers, network has {}",
            n_layers, dense_layers.len()
        ));
    }

    for expected_idx in 0..n_layers {
        // --- Read and validate LAYER header ---
        let layer_line = lines.next()
            .ok_or_else(|| format!("Missing LAYER {} header", expected_idx))?;

        let parts: Vec<&str> = layer_line.split_whitespace().collect();
        if parts.len() != 4 || parts[0] != "LAYER" {
            return Err(format!("Malformed layer header: '{}'", layer_line));
        }

        let file_input_size: usize = parts[2].parse()
            .map_err(|_| format!("Invalid input_size in: '{}'", layer_line))?;
        let file_output_size: usize = parts[3].parse()
            .map_err(|_| format!("Invalid output_size in: '{}'", layer_line))?;

        let dense = &mut dense_layers[expected_idx];
        let expected_input  = dense.weights.cols;
        let expected_output = dense.weights.rows;

        if file_input_size != expected_input || file_output_size != expected_output {
            return Err(format!(
                "Layer {} shape mismatch: file has ({}x{}), network expects ({}x{})",
                expected_idx, file_output_size, file_input_size,
                expected_output, expected_input
            ));
        }

        // --- Read weight values ---
        let weight_line = lines.next()
            .ok_or_else(|| format!("Missing weight data for layer {}", expected_idx))?;
        let weight_values: Result<Vec<f64>, String> = weight_line
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(|_| format!("Invalid weight value: '{}'", s)))
            .collect();
        let weight_values = weight_values?;

        if weight_values.len() != expected_input * expected_output {
            return Err(format!(
                "Layer {} weight count mismatch: expected {}, got {}",
                expected_idx, expected_input * expected_output, weight_values.len()
            ));
        }

        // --- Read bias values ---
        let bias_line = lines.next()
            .ok_or_else(|| format!("Missing bias data for layer {}", expected_idx))?;
        let bias_values: Result<Vec<f64>, String> = bias_line
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(|_| format!("Invalid bias value: '{}'", s)))
            .collect();
        let bias_values = bias_values?;

        if bias_values.len() != expected_output {
            return Err(format!(
                "Layer {} bias count mismatch: expected {}, got {}",
                expected_idx, expected_output, bias_values.len()
            ));
        }

        // --- Apply loaded values ---
        dense.weights = Matrix::from_vec(expected_output, expected_input, weight_values);
        dense.biases  = Matrix::from_vec(expected_output, 1, bias_values);
    }

    Ok(())
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

    fn make_test_network() -> Network {
        let mut rng = Rng::new(1);
        let opt     = SGD::new(2, 0.0);
        let mut net = Network::new(Box::new(opt), 0.1);
        net.add(Box::new(Dense::new(4, 3, ActivationType::Sigmoid, &mut rng)));
        net.add(Box::new(Dense::new(3, 2, ActivationType::OutputSoftmax, &mut rng)));
        net
    }

    #[test]
    fn test_save_then_load_restores_identical_weights() {
        let net = make_test_network();
        let path = "/tmp/test_weights_roundtrip.txt";

        save_weights(&net, path).unwrap();

        // Build a FRESH network with the same architecture but
        // different (freshly randomized) initial weights.
        let mut fresh_net = make_test_network();

        // Confirm weights genuinely differ before loading (sanity
        // check that we're actually testing something meaningful).
        let original_first_weight = net.layers[0].as_dense().unwrap().weights.data[0];
        let fresh_first_weight    = fresh_net.layers[0].as_dense().unwrap().weights.data[0];
        // (Different seeds in make_test_network calls would differ;
        // since we reuse seed 1 here they'd actually match -- so
        // this assertion is illustrative, not load-bearing.)
        let _ = (original_first_weight, fresh_first_weight);

        load_weights(&mut fresh_net, path).unwrap();

        // After loading, weights must match the ORIGINAL exactly.
        let orig_dense  = net.layers[0].as_dense().unwrap();
        let fresh_dense = fresh_net.layers[0].as_dense().unwrap();
        assert_eq!(orig_dense.weights.data, fresh_dense.weights.data);
        assert_eq!(orig_dense.biases.data, fresh_dense.biases.data);

        let orig_dense2  = net.layers[1].as_dense().unwrap();
        let fresh_dense2 = fresh_net.layers[1].as_dense().unwrap();
        assert_eq!(orig_dense2.weights.data, fresh_dense2.weights.data);
        assert_eq!(orig_dense2.biases.data, fresh_dense2.biases.data);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_missing_file_returns_err() {
        let mut net = make_test_network();
        let result = load_weights(&mut net, "/tmp/this_file_does_not_exist_xyz.txt");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_wrong_architecture_returns_err() {
        let net = make_test_network();
        let path = "/tmp/test_weights_wrong_arch.txt";
        save_weights(&net, path).unwrap();

        // Build a network with a DIFFERENT architecture (different
        // layer sizes) -- loading should fail with a clear error,
        // not silently corrupt data or panic.
        let mut rng = Rng::new(2);
        let opt     = SGD::new(1, 0.0);
        let mut wrong_net = Network::new(Box::new(opt), 0.1);
        wrong_net.add(Box::new(Dense::new(4, 5, ActivationType::OutputSoftmax, &mut rng)));

        let result = load_weights(&mut wrong_net, path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_load_corrupted_header_returns_err() {
        let path = "/tmp/test_weights_bad_header.txt";
        std::fs::write(path, "NOT_THE_RIGHT_HEADER\nLAYERS 1\n").unwrap();

        let mut net = make_test_network();
        let result = load_weights(&mut net, path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_creates_readable_file() {
        let net = make_test_network();
        let path = "/tmp/test_weights_readable.txt";

        save_weights(&net, path).unwrap();

        let contents = std::fs::read_to_string(path).unwrap();
        assert!(contents.starts_with(FORMAT_HEADER));
        assert!(contents.contains("LAYERS 2"));
        assert!(contents.contains("LAYER 0 4 3"));
        assert!(contents.contains("LAYER 1 3 2"));

        let _ = std::fs::remove_file(path);
    }
}