// =====================================================================
// demo.rs
//
// Interactive terminal demo: load a trained model, let the user draw
// a digit/garment on a coarse ASCII grid, then predict its class.
//
// GRID DESIGN:
// User draws on a 14x14 grid (196 cells) for fast interaction. Each
// coarse cell is upscaled to a 2x2 block of real pixels, producing
// the full 28x28 = 784 pixel input the network expects -- same shape
// and normalization (0.0-1.0) as training data.
//
// RESPONSIBILITY BOUNDARY:
// This file only handles terminal I/O and grid-to-Matrix conversion.
// It calls io_utils::load_weights() to restore a trained model and
// Network::forward_single() to predict -- no training logic here.
// =====================================================================

use crate::matrix::Matrix;
use crate::network::Network;
use crate::io_utils;
use std::io::{self, Write};

const COARSE_SIZE: usize = 14; // 14x14 user-facing grid
const FULL_SIZE:   usize = 28; // 28x28 = 784, what the network expects
const SCALE:       usize = FULL_SIZE / COARSE_SIZE; // each coarse cell -> 2x2 block

/// Runs the interactive draw-and-predict demo. Loads weights from
/// `weights_path` into `net` (net must already have the correct
/// architecture built, matching how it was trained).
pub fn run_demo(net: &mut Network, weights_path: &str, class_names: &[&str]) {
    println!("\n================================================");
    println!(" INTERACTIVE DEMO: Draw and Predict");
    println!("================================================\n");

    match io_utils::load_weights(net, weights_path) {
        Ok(_)  => println!("Loaded trained weights from {}\n", weights_path),
        Err(e) => {
            println!("Could not load weights: {}", e);
            println!("Run training first to produce {}.", weights_path);
            return;
        }
    }

    loop {
        // grid[row][col] = true means "filled in" (pixel on).
        let mut grid = vec![vec![false; COARSE_SIZE]; COARSE_SIZE];

        println!("Draw a garment shape on the {}x{} grid below.", COARSE_SIZE, COARSE_SIZE);
        println!("Commands:");
        println!("  toggle <row> <col>   -- flip a cell on/off (0-indexed)");
        println!("  fill <row> <col>     -- turn a cell on");
        println!("  clear                -- reset the grid");
        println!("  predict              -- run the network on your drawing");
        println!("  quit                 -- exit the demo\n");

        loop {
            print_grid(&grid);
            print!("> ");
            io::stdout().flush().unwrap(); // ensure "> " prints before input blocks

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                println!("Failed to read input.");
                continue;
            }

            let input = input.trim();
            let parts: Vec<&str> = input.split_whitespace().collect();

            match parts.as_slice() {
                ["toggle", r, c] => {
                    if let (Ok(row), Ok(col)) = (r.parse::<usize>(), c.parse::<usize>()) {
                        if row < COARSE_SIZE && col < COARSE_SIZE {
                            grid[row][col] = !grid[row][col];
                        } else {
                            println!("Coordinates out of range (0-{}).", COARSE_SIZE - 1);
                        }
                    } else {
                        println!("Usage: toggle <row> <col>");
                    }
                }
                ["fill", r, c] => {
                    if let (Ok(row), Ok(col)) = (r.parse::<usize>(), c.parse::<usize>()) {
                        if row < COARSE_SIZE && col < COARSE_SIZE {
                            grid[row][col] = true;
                        } else {
                            println!("Coordinates out of range (0-{}).", COARSE_SIZE - 1);
                        }
                    } else {
                        println!("Usage: fill <row> <col>");
                    }
                }
                ["clear"] => {
                    grid = vec![vec![false; COARSE_SIZE]; COARSE_SIZE];
                    println!("Grid cleared.");
                }
                ["predict"] => {
                    let input_matrix = grid_to_matrix(&grid);
                    let prediction = net.forward_single(&input_matrix);
                    print_prediction(&prediction, class_names);
                }
                ["quit"] => {
                    println!("Exiting demo.");
                    return;
                }
                _ => {
                    println!("Unrecognized command: '{}'", input);
                }
            }
        }
    }
}

/// Prints the coarse grid to the terminal, filled cells as '#',
/// empty cells as '.', with row/column index labels for reference.
fn print_grid(grid: &[Vec<bool>]) {
    print!("    ");
    for c in 0..COARSE_SIZE {
        print!("{:>2}", c);
    }
    println!();

    for (r, row) in grid.iter().enumerate() {
        print!("{:>3} ", r);
        for &cell in row {
            print!(" {}", if cell { '#' } else { '.' });
        }
        println!();
    }
}

/// Converts the coarse boolean grid into a (784 x 1) Matrix matching
/// the network's expected input shape and normalization.
///
/// Upscaling: each coarse cell becomes a SCALE x SCALE block of full
/// pixels (all same value) -- e.g. a 14x14 grid with SCALE=2 produces
/// a 28x28 image where every 2x2 block is uniformly on or off.
fn grid_to_matrix(grid: &[Vec<bool>]) -> Matrix {
    let mut pixels = vec![0.0_f64; FULL_SIZE * FULL_SIZE];

    for coarse_r in 0..COARSE_SIZE {
        for coarse_c in 0..COARSE_SIZE {
            let value = if grid[coarse_r][coarse_c] { 1.0 } else { 0.0 };

            // Fill the SCALE x SCALE block of full-resolution pixels
            // that this coarse cell expands into.
            for dr in 0..SCALE {
                for dc in 0..SCALE {
                    let full_r = coarse_r * SCALE + dr;
                    let full_c = coarse_c * SCALE + dc;
                    pixels[full_r * FULL_SIZE + full_c] = value;
                }
            }
        }
    }

    // Network expects (784 x 1) -- same shape as training images.
    Matrix::from_vec(FULL_SIZE * FULL_SIZE, 1, pixels)
}

/// Prints the network's prediction: top class plus confidence, and
/// the full probability distribution sorted descending -- lets the
/// user see what the network considered as runner-up guesses too.
fn print_prediction(prediction: &Matrix, class_names: &[&str]) {
    let predicted_class = Network::argmax(prediction);

    println!("\n--- Prediction ---");
    println!(
        "Predicted: {} ({:.1}% confidence)\n",
        class_names.get(predicted_class).unwrap_or(&"?"),
        prediction.get(predicted_class, 0) * 100.0
    );

    // Sort all classes by probability, descending, show full ranking.
    let mut ranked: Vec<(usize, f64)> = (0..prediction.rows)
        .map(|i| (i, prediction.get(i, 0)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("All class probabilities:");
    for (class_idx, prob) in ranked {
        let name = class_names.get(class_idx).unwrap_or(&"?");
        println!("  {:<14} {:>5.1}%", name, prob * 100.0);
    }
    println!();
}

// =====================================================================
// TESTS
//
// Interactive I/O (stdin loops) is not practical to unit test directly
// -- instead we test the pure logic pieces: grid-to-matrix conversion
// and prediction formatting inputs, which is where actual bugs could
// hide (off-by-one in upscaling, wrong argmax, etc.).
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_to_matrix_shape() {
        let grid = vec![vec![false; COARSE_SIZE]; COARSE_SIZE];
        let m = grid_to_matrix(&grid);
        assert_eq!(m.rows, 784);
        assert_eq!(m.cols, 1);
    }

    #[test]
    fn test_empty_grid_produces_all_zero_pixels() {
        let grid = vec![vec![false; COARSE_SIZE]; COARSE_SIZE];
        let m = grid_to_matrix(&grid);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_single_filled_cell_upscales_to_block() {
        // Fill coarse cell (0,0) -- should produce a SCALE x SCALE
        // block of 1.0s at the top-left of the full image, and
        // zeros everywhere else.
        let mut grid = vec![vec![false; COARSE_SIZE]; COARSE_SIZE];
        grid[0][0] = true;

        let m = grid_to_matrix(&grid);

        // Check the top-left SCALE x SCALE block is all 1.0.
        for r in 0..SCALE {
            for c in 0..SCALE {
                assert_eq!(m.get(r, 0) as usize, m.get(r, 0) as usize); // shape sanity
                let idx = r * FULL_SIZE + c;
                assert_eq!(m.data[idx], 1.0,
                    "pixel ({},{}) should be filled from coarse cell (0,0)", r, c);
            }
        }

        // A pixel well outside that block should remain 0.0.
        let far_idx = (FULL_SIZE - 1) * FULL_SIZE + (FULL_SIZE - 1);
        assert_eq!(m.data[far_idx], 0.0);
    }

    #[test]
    fn test_full_grid_produces_all_ones() {
        let grid = vec![vec![true; COARSE_SIZE]; COARSE_SIZE];
        let m = grid_to_matrix(&grid);
        assert!(m.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_prediction_argmax_matches_highest_probability() {
        let prediction = Matrix::from_vec(5, 1, vec![0.1, 0.05, 0.6, 0.15, 0.1]);
        assert_eq!(Network::argmax(&prediction), 2);
    }
}