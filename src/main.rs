// =====================================================================
// main.rs
//
// Entry point. Has TWO modes, chosen by the first command-line arg:
//
//   TRAIN MODE (default):
//     cargo run --release -- data/fashion-mnist_train.csv data/fashion-mnist_test.csv
//     Loads data, trains the network, evaluates, saves weights to
//     trained.weights, prints confusion matrix + per-class accuracy.
//
//   DEMO MODE:
//     cargo run --release -- demo
//     Skips training entirely. Builds an empty network with the same
//     architecture, loads trained.weights into it, and launches the
//     interactive draw-and-predict terminal demo.
//
// This separation means you can demo the model as many times as you
// want (e.g. rehearsing for your mentor) without ever retraining.
// =====================================================================

mod matrix;
mod rng;
mod init;
mod activation;
mod layer;
mod loss;
mod optimizer;
mod network;
mod data;
mod metrics;
mod io_utils;
mod demo;

use activation::ActivationType;
use layer::{Dense, Dropout};
use network::Network;
use optimizer::Adam;
use data::Dataset;

// Fashion-MNIST class names -- used by both training output and demo.
const CLASS_NAMES: [&str; 10] = [
    "T-shirt/top", "Trouser",   "Pullover", "Dress",  "Coat",
    "Sandal",      "Shirt",     "Sneaker",  "Bag",    "Ankle boot",
];

fn main() {
    // Check the first argument to decide which mode to run.
    // `cargo run --release -- demo` -> args: ["binary_name", "demo"]
    // .as_deref() converts Option<String> to Option<&str> so we can
    // compare it against a string literal with ==.
    let first_arg = std::env::args().nth(1);

    if first_arg.as_deref() == Some("demo") {
        run_demo_mode();
        return; // exit early -- skip all training code below
    }

    run_training_mode(first_arg);
}

// -----------------------------------------------------------------
// BUILD ARCHITECTURE
//
// Shared by both modes -- guarantees the demo network has the EXACT
// same shape as the trained one, so load_weights() succeeds.
// If you ever change the architecture, only this function needs
// updating -- both training and demo modes stay in sync automatically.
// -----------------------------------------------------------------
fn build_network(seed: u64, learning_rate: f64) -> Network {
    let mut rng = rng::Rng::new(seed);
    let n_layers = 3; // Dense layers only -- optimizer state count

    let optimizer = Adam::new(n_layers);
    let mut net = Network::new(Box::new(optimizer), learning_rate);

    net.add(Box::new(Dense::new(784, 256, ActivationType::Sigmoid, &mut rng)));
    net.add(Box::new(Dropout::new(0.15, 256, 1337)));
    net.add(Box::new(Dense::new(256, 128, ActivationType::Sigmoid, &mut rng)));
    net.add(Box::new(Dropout::new(0.15, 128, 2674)));
    net.add(Box::new(Dense::new(128, 10, ActivationType::OutputSoftmax, &mut rng)));

    net
}

// -----------------------------------------------------------------
// DEMO MODE
//
// No data loading, no training -- just restore a trained model from
// disk and hand control to the interactive terminal demo.
// -----------------------------------------------------------------
fn run_demo_mode() {
    println!("================================================");
    println!(" DEMO MODE -- loading trained weights, no training");
    println!("================================================\n");

    // Seed/lr here don't matter -- load_weights() overwrites every
    // weight and bias immediately after construction.
    let mut net = build_network(0, 0.01);

    demo::run_demo(&mut net, "trained.weights", &CLASS_NAMES);
}

// -----------------------------------------------------------------
// TRAINING MODE
//
// Everything the program did before -- unchanged, just moved into
// its own function so main() can branch cleanly between modes.
// -----------------------------------------------------------------
fn run_training_mode(first_arg: Option<String>) {
    let train_path = first_arg
        .unwrap_or_else(|| "data/fashion-mnist_train.csv".to_string());
    let test_path = std::env::args().nth(2)
        .unwrap_or_else(|| "data/fashion-mnist_test.csv".to_string());

    println!("================================================");
    println!(" Fashion-MNIST Neural Network -- Pure Rust");
    println!(" No external crates. Built from scratch.");
    println!("================================================\n");

    print!("Loading training data from {}... ", train_path);
    let train_full = match Dataset::load_csv(&train_path) {
        Ok(d)  => { println!("OK ({} samples)", d.len()); d }
        Err(e) => { eprintln!("ERROR: {}", e); std::process::exit(1); }
    };
    let (mut train_data, val_data) = train_full.split_train_val(0.1);
    println!("Split: {} train / {} validation\n", train_data.len(), val_data.len());

    print!("Loading test data from {}... ", test_path);
    let test_data = match Dataset::load_csv(&test_path) {
        Ok(d)  => { println!("OK ({} samples)\n", d.len()); d }
        Err(e) => { eprintln!("ERROR: {}", e); std::process::exit(1); }
    };

    let seed         = 42;
    let epochs       = 30;
    let batch_size   = 128;
    let lr           = 0.01;
    let decay_every  = 10;
    let decay_factor = 0.5;

    println!("Configuration:");
    println!("  Architecture : 784 -> 256 -> [Dropout 0.15] -> 128 -> [Dropout 0.15] -> 10");
    println!("  Activation   : Sigmoid (hidden), Softmax (output)");
    println!("  Optimizer    : Adam");
    println!("  Learning rate: {} (decays x{} every {} epochs)", lr, decay_factor, decay_every);
    println!("  Epochs       : {}", epochs);
    println!("  Batch size   : {}", batch_size);
    println!("  RNG seed     : {}\n", seed);

    let mut net = build_network(seed, lr);
    let mut rng = rng::Rng::new(seed + 1); // separate rng for shuffling

    println!("Network built. Starting training...\n");

    let train_start = std::time::Instant::now();

    net.train(
        &mut train_data.images,
        &mut train_data.labels,
        &val_data.images,
        &val_data.labels,
        epochs,
        batch_size,
        &mut rng,
        decay_every,
        decay_factor,
    );

    let train_duration = train_start.elapsed();
    println!("\nTraining completed in {:.2?}", train_duration);

    let val_acc = net.evaluate(&val_data.images, &val_data.labels);
    println!("Final validation accuracy: {:.2}%", val_acc);

    let val_preds: Vec<usize> = val_data.images.iter()
        .map(|img| {
            let pred = net.forward_single(img);
            Network::argmax(&pred)
        })
        .collect();
    metrics::print_confusion_matrix(&val_preds, &val_data.labels, &CLASS_NAMES);

    match io_utils::save_weights(&net, "trained.weights") {
        Ok(_)  => println!("Weights saved to trained.weights"),
        Err(e) => println!("Warning: could not save weights: {}", e),
    }

    println!("\n================================================");
    println!(" FINAL TEST SET EVALUATION");
    println!("================================================");

    let test_acc = net.evaluate(&test_data.images, &test_data.labels);
    println!("Test accuracy: {:.2}%", test_acc);

    let test_preds: Vec<usize> = test_data.images.iter()
        .map(|img| {
            let pred = net.forward_single(img);
            Network::argmax(&pred)
        })
        .collect();
    metrics::print_confusion_matrix(&test_preds, &test_data.labels, &CLASS_NAMES);
    metrics::print_per_class_accuracy(&test_preds, &test_data.labels, &CLASS_NAMES);

    let log_line = format!(
        "Adam+Dropout,{},{},{},{:.2},{:.2},{:.1}\n",
        lr, epochs, batch_size, val_acc, test_acc,
        train_duration.as_secs_f64()
    );
    match std::fs::OpenOptions::new().append(true).create(true).open("results.log") {
        Ok(mut f) => {
            use std::io::Write;
            let _ = f.write_all(log_line.as_bytes());
            println!("\nResults logged to results.log");
        }
        Err(e) => println!("Warning: could not write results.log: {}", e),
    }

    println!("\nDone. Run `cargo run --release -- demo` any time to try the interactive demo.");
}