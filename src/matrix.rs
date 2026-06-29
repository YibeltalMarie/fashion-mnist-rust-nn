// TODO (Day 1): Flat Vec<f64>-backed Matrix struct.
// Required ops: new, zeros, matmul, transpose, add, sub,
// elementwise mul, scalar mul, add_row_broadcast (for bias).

// =====================================================================
// matrix.rs
//
// Our hand-rolled replacement for numpy/ndarray. Every number that
// flows through the network (inputs, weights, biases, gradients)
// lives inside a Matrix.
//
// STORAGE DESIGN:
// We store data as ONE flat Vec<f64> instead of a Vec<Vec<f64>> (a
// "list of lists"). A flat array keeps every element physically next
// to each other in memory, which is much faster for the CPU to read
// in tight loops like matmul. We simulate 2D indexing manually using
// the formula:
//
//      index = row * cols + col
//
// CONVENTION:
// For a Dense layer, weights are stored as (output_size x input_size),
// so the forward pass is simply:   output = W . input + bias
// (no transpose needed). The backward pass DOES need W transposed,
// since we're propagating the error signal in the opposite direction.
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

    /// Creates a new matrix filled entirely with 0.0.
    /// `Self` here just means "Matrix" — Rust lets you use `Self`
    /// inside an impl block instead of repeating the type name.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            // vec![value; count] is a macro that builds a Vec of
            // length `count`, every slot initialized to `value`.
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a matrix from an existing flat Vec<f64>.
    /// We check the length matches rows*cols -- if not, panic with
    /// a clear message rather than silently producing garbage.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix::from_vec: data length {} does not match rows*cols = {}",
            data.len(),
            rows * cols
        );
        Matrix { rows, cols, data }
    }

    // ---------------------------------------------------------------
    // ELEMENT ACCESS
    //
    // &self means these methods only need to READ the matrix --
    // they borrow it immutably (Lesson 2). You call them as
    // `my_matrix.get(r, c)`.
    // ---------------------------------------------------------------

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// &mut self means this method needs to WRITE into the matrix --
    /// a mutable borrow. The caller's variable must itself be
    /// declared `mut` for this to be callable.
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    // ---------------------------------------------------------------
    // MATRIX MULTIPLICATION
    //
    // For A (m x n) times B (n x p), result C is (m x p), where:
    //     C[i][j] = sum over k of ( A[i][k] * B[k][j] )
    //
    // We walk row i of A and column j of B together, multiplying
    // and accumulating. This is the single most-called function in
    // the whole project -- every forward/backward pass uses it.
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
    // TRANSPOSE
    //
    // Flips rows and columns: element (r, c) in the original becomes
    // element (c, r) in the result. Needed during backprop to push
    // gradients from one layer back to the previous layer.
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
    //
    // "Element-wise" means: same-shaped matrices, combine matching
    // positions one-to-one. Used for adding bias, and for multiplying
    // gradients by activation derivatives during backprop.
    // ---------------------------------------------------------------

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols),
            "add: shape mismatch");
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols),
            "hadamard: shape mismatch");
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }

    pub fn scalar_mul(&self, scalar: f64) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|x| x * scalar).collect();
        Matrix::from_vec(self.rows, self.cols, data)
    }
}

// =====================================================================
// TESTS
//
// #[cfg(test)] tells Rust "only compile this when running `cargo test`"
// -- it's excluded entirely from a normal `cargo build`/`cargo run`.
// Each #[test]-annotated function is a separate test case that
// `cargo test` discovers and runs automatically.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*; // brings everything from the outer module (Matrix) into scope

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
        assert_eq!(m.get(1, 1), 0.0); // untouched cell stays zero
    }

    #[test]
    fn test_matmul_known_result() {
        // A = [[1, 2],      B = [[5, 6],
        //      [3, 4]]           [7, 8]]
        //
        // A . B = [[1*5+2*7, 1*6+2*8],   = [[19, 22],
        //          [3*5+4*7, 3*6+4*8]]      [43, 50]]
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_transpose() {
        // [[1, 2, 3]]  ->  [[1],
        //                   [2],
        //                   [3]]
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
        let c = a.add(&b);
        assert_eq!(c.data, vec![11.0, 22.0]);
    }

    #[test]
    fn test_hadamard() {
        let a = Matrix::from_vec(1, 2, vec![2.0, 3.0]);
        let b = Matrix::from_vec(1, 2, vec![4.0, 5.0]);
        let c = a.hadamard(&b);
        assert_eq!(c.data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let c = a.scalar_mul(2.0);
        assert_eq!(c.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic] // we EXPECT this to panic, due to shape mismatch
    fn test_matmul_shape_mismatch_panics() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(2, 3); // wrong shape for matmul with a
        a.matmul(&b);
    }
}