// optimizer.rs
//
// Landscape scanning, local minimization (Nelder-Mead) and field-profile
// extraction on top of the coherent-block solver.
use num_complex::Complex64;

// Reuse the shared low-level primitives from the crate
use crate::coherent_block::solve_coherent_block_fields_inner;

// -----------------------------------------------------------------------------
// Helper: compute the complex reflection coefficient for a given n_eff
// -----------------------------------------------------------------------------
#[inline]
pub fn reflection_coefficient_helper(
    n_stack: &[Complex64],
    inv_n: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    n_eff: Complex64,
    pol: i32,
) -> Complex64 {
    // inv_n is computed once per wavelength by the caller and shared read-only
    // across all grid points / simplex evaluations (and across rayon threads).
    let (r_front, _, _, _, _, _, _, _) = solve_coherent_block_fields_inner(
        0,
        n_stack.len() - 1,
        n_stack,
        inv_n,
        thicknesses,
        rough_vals,
        rough_types,
        lam,
        n_eff,
        pol,
    );
    r_front
}

// -----------------------------------------------------------------------------
// Characteristic function: |1 / r(n_eff)|²
// -----------------------------------------------------------------------------
#[inline]
pub fn char_func(
    n_stack: &[Complex64],
    inv_n: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    n_eff: Complex64,
    pol: i32,
) -> f64 {
    let r = reflection_coefficient_helper(
        n_stack, inv_n, thicknesses, rough_types, rough_vals, lam, n_eff, pol,
    );
    let abs_r = r.norm();
    if abs_r < 1e-15 {
        1e30
    } else {
        (1.0 / abs_r).powi(2)
    }
}

/// Real‑valued wrapper for minimisation (expects [Re, Im] slice)
#[allow(clippy::too_many_arguments)]
pub fn char_func_xy(
    xy: &[f64],
    n_stack: &[Complex64],
    inv_n: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    pol: i32,
) -> f64 {
    let n_eff = Complex64::new(xy[0], xy[1]);
    char_func(n_stack, inv_n, thicknesses, rough_types, rough_vals, lam, n_eff, pol)
}

// -----------------------------------------------------------------------------
// Coarse landscape scan (parallel, GIL‑free)
