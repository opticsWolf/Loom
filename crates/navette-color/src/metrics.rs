// src/metrics.rs
//! Core metric helpers and broadcasting utilities.
//!
//! This module implements the runtime broadcast logic required by Delta‑E
//! functions (1 reference vs N samples, or N vs 1). The `map_pairs` function
//! is re‑exported by `func_15` for shape handling.

use rayon::prelude::*;

/// Apply a binary function to paired elements from two color batches,
/// broadcasting when one batch has length 1.
///
/// This function mimics NumPy’s broadcasting rules for two arrays of shape
/// `(n, 3)` where one dimension may be 1. If both lengths are equal, they are
/// paired element‑wise. If one length is 1, that single element is paired with
/// every element of the other batch. Any other combination panics.
///
/// # Panics
/// Panics if the two batches have different lengths and neither length is 1.
/// Below this many output elements the Rayon split/coordination overhead
/// outweighs the benefit, so `map_pairs` runs sequentially even when the
/// `parallel` feature is enabled. Tune to taste; ~50k is a safe default for
/// the cheap per-element kernels here.
pub const PAR_THRESHOLD: usize = 50_000;

pub fn map_pairs<F>(lab1: &[[f64; 3]], lab2: &[[f64; 3]], f: F) -> Vec<f64>
where
    F: Fn(&[f64; 3], &[f64; 3]) -> f64 + Sync + Send,
{
    let n1 = lab1.len();
    let n2 = lab2.len();

    // Validate broadcastability up front (single source of the panic message).
    let n = match (n1, n2) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        (a, b) => panic!("shapes {a} and {b} are not broadcastable"),
    };

    // Parallel execution (Rayon) for large batches only.
    {
        if n >= PAR_THRESHOLD {
            return if n1 == n2 {
                lab1.par_iter().zip(lab2.par_iter()).map(|(a, b)| f(a, b)).collect()
            } else if n1 == 1 {
                let a = &lab1[0];
                lab2.par_iter().map(|b| f(a, b)).collect()
            } else {
                let b = &lab2[0];
                lab1.par_iter().map(|a| f(a, b)).collect()
            };
        }
        // small batch: fall through to the sequential path below
    }

    // Sequential execution (auto-vectorizable by LLVM; also the small-batch
    // path when `parallel` is enabled).
    let _ = n;
    if n1 == n2 {
        lab1.iter().zip(lab2.iter()).map(|(a, b)| f(a, b)).collect()
    } else if n1 == 1 {
        let a = &lab1[0];
        lab2.iter().map(|b| f(a, b)).collect()
    } else {
        let b = &lab2[0];
        lab1.iter().map(|a| f(a, b)).collect()
    }
}

/// Internal helper to broadcast a single reference against many samples,
/// returning two vectors of equal length (both allocated as `Vec<[f64;3]>`).
#[doc(hidden)]
pub fn broadcast_pair(
    lab1: &[[f64; 3]],
    lab2: &[[f64; 3]],
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let n1 = lab1.len();
    let n2 = lab2.len();

    let n = match (n1, n2) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        (a, b) => panic!("shapes {a} and {b} are not broadcastable"),
    };

    let mut out1 = Vec::with_capacity(n);
    let mut out2 = Vec::with_capacity(n);

    for i in 0..n {
        out1.push(if n1 == 1 { lab1[0] } else { lab1[i] });
        out2.push(if n2 == 1 { lab2[0] } else { lab2[i] });
    }

    (out1, out2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_equal_length() {
        let a = [[1.0; 3]; 3];
        let b = [[2.0; 3]; 3];
        let res = map_pairs(&a, &b, |_, _| 42.0);
        assert_eq!(res, vec![42.0, 42.0, 42.0]);
    }

    #[test]
    fn broadcast_one_vs_n() {
        let a = [[1.0; 3]];
        let b = [[2.0; 3], [3.0; 3]];
        let res = map_pairs(&a, &b, |x, y| y[0] - x[0]);
        assert_eq!(res, vec![1.0, 2.0]);
    }

    #[test]
    fn broadcast_n_vs_one() {
        let a = [[1.0; 3], [2.0; 3]];
        let b = [[3.0; 3]];
        let res = map_pairs(&a, &b, |x, y| y[0] - x[0]);
        assert_eq!(res, vec![2.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "shapes 2 and 3 are not broadcastable")]
    fn panic_on_incompatible() {
        let a = [[0.0; 3]; 2];
        let b = [[0.0; 3]; 3];
        map_pairs(&a, &b, |_, _| 0.0);
    }

    #[test]
    fn broadcast_pair_helper() {
        let a = [[1.0; 3]];
        let b = [[2.0; 3], [3.0; 3]];
        let (a_bcast, b_bcast) = broadcast_pair(&a, &b);
        assert_eq!(a_bcast, vec![[1.0; 3], [1.0; 3]]);
        assert_eq!(b_bcast, vec![[2.0; 3], [3.0; 3]]);
    }
}