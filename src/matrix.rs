// =====================================================================
// matrix.rs
//
// Our hand-rolled replacement for numpy/ndarray.
//
// STORAGE: one flat Vec<f64>, index = row * cols + col.
//
// BATCHING (new): matrices now represent WHOLE MINI-BATCHES, not just
// single samples. A batch of 32 images is one (784 x 32) Matrix --
// each COLUMN is one sample. This means ONE matmul processes the
// entire batch at once instead of looping 32 times -- eliminates
// almost all per-sample overhead (clones, allocations, function calls).
//
// PARALLELISM: matmul_parallel() splits the OUTPUT ROWS of a matmul
// across std::thread::scope worker threads. This is safe because:
//   - threads only READ self/other (never mutate them)
//   - each thread writes to a DISJOINT slice of the result
//     (via split_at_mut -- the compiler proves no overlap)
// No Arc/Mutex needed -- thread::scope guarantees every spawned
// thread finishes before the scope block returns.
// =====================================================================

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    // ---------------------------------------------------------------
    // CONSTRUCTORS
    // ---------------------------------------------------------------

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(), rows * cols,
            "Matrix::from_vec: data length {} does not match rows*cols = {}",
            data.len(), rows * cols
        );
        Matrix { rows, cols, data }
    }

    // ---------------------------------------------------------------
    // ELEMENT ACCESS
    // ---------------------------------------------------------------

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    // ---------------------------------------------------------------
    // MATRIX MULTIPLICATION -- SEQUENTIAL
    //
    // Kept as the ground-truth implementation and used as a fallback
    // for small matrices where thread spawn overhead isn't worth it.
    // ---------------------------------------------------------------

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul shape mismatch: ({}x{}) * ({}x{})",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    // ---------------------------------------------------------------
    // MATRIX MULTIPLICATION -- PARALLEL
    //
    // Splits the OUTPUT ROWS of self.matmul(other) across worker
    // threads (one thread per CPU core). Each thread computes a
    // contiguous slice of output rows, writing into its own disjoint
    // chunk of the result buffer via split_at_mut.
    //
    // Falls back to sequential matmul for small matrices, since
    // thread spawn overhead would dominate the actual work.
    // ---------------------------------------------------------------

    pub fn matmul_parallel(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul_parallel shape mismatch: ({}x{}) * ({}x{})",
            self.rows, self.cols, other.rows, other.cols
        );

        let total_rows = self.rows;
        let out_cols = other.cols;

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Not worth threading a small matmul -- overhead would dominate.
        if num_threads <= 1 || total_rows < num_threads * 8 {
            return self.matmul(other);
        }

        let chunk = (total_rows + num_threads - 1) / num_threads;
        let mut result_data = vec![0.0; total_rows * out_cols];

        std::thread::scope(|scope| {
            let mut remaining: &mut [f64] = &mut result_data[..];
            let mut row_start = 0;

            while row_start < total_rows {
                let rows_here = chunk.min(total_rows - row_start);
                // split_at_mut proves the two halves are non-overlapping
                // memory -- this is what makes concurrent writes safe
                // without any lock.
                let (head, tail) = remaining.split_at_mut(rows_here * out_cols);
                remaining = tail;

                // self and other are already &Matrix (references are
                // Copy), so `move` just copies the reference itself
                // into the thread, not the underlying data.
                scope.spawn(move || {
                    for local_i in 0..rows_here {
                        let i = row_start + local_i;
                        for j in 0..out_cols {
                            let mut sum = 0.0;
                            for k in 0..self.cols {
                                sum += self.get(i, k) * other.get(k, j);
                            }
                            head[local_i * out_cols + j] = sum;
                        }
                    }
                });

                row_start += rows_here;
            }
            // thread::scope blocks here until every spawned thread
            // finishes -- no manual .join() needed.
        });

        Matrix::from_vec(total_rows, out_cols, result_data)
    }

    // ---------------------------------------------------------------
    // TRANSPOSE
    // ---------------------------------------------------------------

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result.set(c, r, self.get(r, c));
            }
        }
        result
    }

    // ---------------------------------------------------------------
    // ELEMENT-WISE OPERATIONS
    // ---------------------------------------------------------------

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "add: shape mismatch");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "hadamard: shape mismatch");
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    // ---------------------------------------------------------------
    // BATCH-AWARE OPERATIONS (new)
    // ---------------------------------------------------------------

    /// Adds a (rows x 1) bias vector to EVERY COLUMN of self.
    /// self is (rows x batch_size); bias is (rows x 1).
    /// Needed because after batching, layer output is
    /// (output_size x batch_size), but bias is still per-neuron only
    /// (output_size x 1) -- it must broadcast across every sample.
    pub fn add_bias_broadcast(&self, bias: &Matrix) -> Matrix {
        assert_eq!(self.rows, bias.rows, "add_bias_broadcast: row mismatch");
        assert_eq!(bias.cols, 1, "add_bias_broadcast: bias must be a column vector");

        let mut data = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            let b = bias.get(r, 0);
            for c in 0..self.cols {
                data[r * self.cols + c] = self.get(r, c) + b;
            }
        }
        Matrix::from_vec(self.rows, self.cols, data)
    }

    /// Sums each row across all columns, producing a (rows x 1) result.
    /// Used for bias gradients: since bias affects every sample
    /// identically, its gradient is the SUM of dz across the batch.
    pub fn sum_cols(&self) -> Matrix {
        let mut data = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c);
            }
            data[r] = sum;
        }
        Matrix::from_vec(self.rows, 1, data)
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![0.0; 6]);
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), 5.0);
        assert_eq!(m.get(1, 1), 0.0);
    }

    #[test]
    fn test_matmul_known_result() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_parallel_matches_sequential() {
        // Correctness check: parallel and sequential matmul must
        // produce IDENTICAL results. Use a large-ish matrix so the
        // parallel path actually triggers (not the small-matrix fallback).
        let rows = 200;
        let inner = 50;
        let cols = 10;

        let a_data: Vec<f64> = (0..rows * inner).map(|i| (i % 7) as f64 * 0.1).collect();
        let b_data: Vec<f64> = (0..inner * cols).map(|i| (i % 5) as f64 * 0.2).collect();
        let a = Matrix::from_vec(rows, inner, a_data);
        let b = Matrix::from_vec(inner, cols, b_data);

        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b);

        assert_eq!(seq.rows, par.rows);
        assert_eq!(seq.cols, par.cols);
        for i in 0..seq.data.len() {
            assert!(
                (seq.data[i] - par.data[i]).abs() < 1e-9,
                "mismatch at index {}: seq={} par={}", i, seq.data[i], par.data[i]
            );
        }
    }

    #[test]
    fn test_matmul_parallel_small_falls_back_correctly() {
        // Small matrices take the fallback path -- still must be correct.
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul_parallel(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 1);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(1, 2, vec![1.0, 2.0]);
        let b = Matrix::from_vec(1, 2, vec![10.0, 20.0]);
        assert_eq!(a.add(&b).data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_hadamard() {
        let a = Matrix::from_vec(1, 2, vec![2.0, 3.0]);
        let b = Matrix::from_vec(1, 2, vec![4.0, 5.0]);
        assert_eq!(a.hadamard(&b).data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        assert_eq!(a.scalar_mul(2.0).data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_matmul_shape_mismatch_panics() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3);
        a.matmul(&b);
    }

    #[test]
    fn test_add_bias_broadcast() {
        // self: (2 x 3) -- 2 neurons, batch of 3 samples.
        // bias:  (2 x 1) -- one bias per neuron.
        let m = Matrix::from_vec(2, 3, vec![
            1.0, 2.0, 3.0,   // neuron 0's outputs for 3 samples
            4.0, 5.0, 6.0,   // neuron 1's outputs for 3 samples
        ]);
        let bias = Matrix::from_vec(2, 1, vec![10.0, 100.0]);
        let result = m.add_bias_broadcast(&bias);

        // Every sample in neuron 0's row gets +10, neuron 1's row gets +100.
        assert_eq!(result.data, vec![
            11.0, 12.0, 13.0,
            104.0, 105.0, 106.0,
        ]);
    }

    #[test]
    fn test_sum_cols() {
        // (2 x 3): sum each row across its 3 columns.
        let m = Matrix::from_vec(2, 3, vec![
            1.0, 2.0, 3.0,   // sum = 6
            4.0, 5.0, 6.0,   // sum = 15
        ]);
        let result = m.sum_cols();
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 1);
        assert_eq!(result.data, vec![6.0, 15.0]);
    }
}