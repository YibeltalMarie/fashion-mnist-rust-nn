// TODO (Day 1): Hand-rolled PRNG (xorshift64 or LCG), seedable.
// Required: next_f64() in [0,1), gaussian() via Box-Muller for He/Xavier init


// =====================================================================
// rng.rs
//
// Hand-rolled pseudo-random number generator (PRNG). Needed because
// we can't use the `rand` crate -- but neural network weights must
// start as random values, or every neuron would compute identical
// outputs forever (the "symmetry problem").
//
// ALGORITHM: xorshift64
// A simple, fast, well-known PRNG. Not cryptographically secure, but
// statistically good enough for weight initialization. Deterministic:
// same seed -> same sequence of numbers every time, which makes our
// training runs reproducible.
// =====================================================================

pub struct Rng {
    state: u64, // current internal state -- this IS the "randomness"
}

impl Rng {
    /// Creates a new Rng from a starting seed.
    /// State must never be 0 -- xorshift produces only 0 forever if it
    /// starts at 0, so we force a nonzero fallback.
    pub fn new(seed: u64) -> Self {
        let safe_seed = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Rng { state: safe_seed }
    }

    // -----------------------------------------------------------------
    // CORE ALGORITHM: xorshift64
    //
    // &mut self: every call changes `state`, so this must be a mutable
    // borrow (Lesson 2). Without &mut, we couldn't update self.state.
    //
    // ^=  means "XOR this value into self.state and store the result"
    //     (shorthand for: self.state = self.state ^ something)
    // <<  shifts bits left (e.g. state << 13 means "shift left 13 bits")
    // >>  shifts bits right
    // -----------------------------------------------------------------
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Random f64 in the range [0.0, 1.0).
    /// We divide the raw random u64 by the largest possible u64 value
    /// (u64::MAX, a built-in constant) to squeeze it into [0,1).
    ///
    /// "as f64" is a TYPE CAST -- Rust never converts types silently,
    /// you must explicitly say "treat this u64 as an f64 now."
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Random f64 in a custom range [min, max).
    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    // -----------------------------------------------------------------
    // BOX-MULLER TRANSFORM
    //
    // Converts two uniform [0,1) randoms into one Gaussian
    // (bell-curve) distributed random number. Needed for proper
    // He/Xavier weight initialization later in layer.rs.
    //
    // Formula: gaussian = sqrt(-2 * ln(u1)) * cos(2 * PI * u2)
    // -----------------------------------------------------------------
    pub fn next_gaussian(&mut self) -> f64 {
        // u1 must never be exactly 0.0 (ln(0) is undefined / -infinity),
        // so we nudge it up by a tiny amount if needed.
        let mut u1 = self.next_f64();
        if u1 <= 0.0 {
            u1 = 1e-10;
        }
        let u2 = self.next_f64();

        let two_pi = 2.0 * std::f64::consts::PI;
        (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
    }

    /// Gaussian scaled by a standard deviation -- this is the exact
    /// shape of call we'll use in layer.rs for He/Xavier init:
    /// e.g. rng.gaussian_scaled(sqrt(2.0 / n_inputs as f64))
    pub fn gaussian_scaled(&mut self, std_dev: f64) -> f64 {
        self.next_gaussian() * std_dev
    }
}

// =====================================================================
// TESTS
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_seed_same_sequence() {
        // Determinism check: two Rngs with the same seed must produce
        // the exact same sequence of "random" numbers.
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..5 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = Rng::new(1);
        let mut rng2 = Rng::new(2);
        // Extremely unlikely to collide on the very first draw if the
        // algorithm is working correctly.
        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_next_f64_in_range() {
        let mut rng = Rng::new(123);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "value {} out of [0,1) range", v);
        }
    }

    #[test]
    fn test_next_range_bounds() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.next_range(-5.0, 5.0);
            assert!(v >= -5.0 && v < 5.0, "value {} out of [-5,5) range", v);
        }
    }

    #[test]
    fn test_gaussian_roughly_centered_at_zero() {
        // Statistical sanity check -- not a strict proof, but if Box-Muller
        // is implemented correctly, the average of many draws should be
        // close to 0.0 (the mean of a standard Gaussian).
        let mut rng = Rng::new(99);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| rng.next_gaussian()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean {} too far from 0", mean);
    }

    #[test]
    fn test_zero_seed_does_not_break() {
        // We must never get stuck producing 0 forever.
        let mut rng = Rng::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }
}
