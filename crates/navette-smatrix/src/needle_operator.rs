//! needle_operator.rs
//!
//! Analytic needle-operator sensitivities (Tikhonravov's method) for
//! multilayer stacks solved with scalar Redheffer S-matrices.
//!
//! Pure Rust core (no pyo3 / numpy); the Python entry point lives in
//! `needle_engine.rs`. Shared primitives (star products, roughness form
//! factor, fast complex kernels, spectral differentiation) come from
//! `optics_core` so this operator can never drift numerically from the main
//! solvers. Conventions match them verbatim: element ordering
//! (r_front, t_back, t_fwd, r_back), Im(beta) >= 0 branch, LOG_MIN
//! regularization, roughness types 0..=5.
//!
//! Theory
//! ------
//! Split the stack at a plane at depth z inside host layer j:
//!
//!     S_total(z) = U(z) ⊗ N ⊗ L(z)
//!
//! where U = everything above the plane (ambient side), L = everything below,
//! and N is the "needle": an infinitesimal slab of candidate material n'
//! embedded in the host medium n_j. To first order in its thickness δ the
//! needle's S-matrix entries are
//!
//!     rho(δ) = δ·rho_hat,   tau(δ) = 1 + δ·tau_hat
//!     rho_hat = -2i·beta'·r12        / (1 - r12²)
//!     tau_hat =  i·beta'·(1 + r12²)  / (1 - r12²)
//!
//! with r12 = (y_j − y')/(y_j + y'), beta' = k0·n'·cosθ', y the wave
//! admittance (s: n·cosθ, p: n/cosθ). The exact-to-first-order sensitivity of
//! the stack amplitude reflection coefficient is obtained by composing U, the
//! dual-number needle, and L through the SAME Redheffer star product used by
//! the forward solver, using complex dual numbers (value, d/dδ):
//!
//!     ∂r_k/∂δ (z) = slope of [ U ⊗ N_dual ⊗ L ]_r_front
//!
//! This captures every first-order multiple-reflection path automatically —
//! no hand-expanded algebra to get wrong. The Tikhonravov merit P-function is
//! then the residual-weighted accumulation over spectral points:
//!
//!     P(z) = Σ_k 2·w_k·(R_k − R_target,k)·Re{ conj(r_k) · ∂r_k/∂δ (z) }
//!
//! which equals ∂f₁/∂δ for f₁ = Σ_k w_k·(R_k − R_target,k)². Negative minima
//! of P(z) mark the most profitable needle insertion points.
//!
//! Scope: fully coherent stacks only (the coherent-block solver path).
//! Roughness factors on real interfaces are honoured; the two virtual
//! needle/host interfaces are abrupt (rtype 0), as physically appropriate.

use num_complex::{Complex64, ComplexFloat};
use std::f64::consts::PI;

pub use crate::optics_core::cexp_fast;
use crate::optics_core::{
    cplx, csqrt_fast, redheffer_product_complex_field_inner, redheffer_product_real_inner,
    w_function_inner, C_NM_PER_FS, DBL_EPS, EPS_COS, LOG_MIN,
};

/// S-matrix element in the crate-wide convention.
pub type S4 = (Complex64, Complex64, Complex64, Complex64);

/// cos θ inside a medium from the Snell invariant nsin_θi = n₀·sinθ₀.
#[inline(always)]
pub fn cos_from_nsin(nsin_fi: Complex64, n: Complex64) -> Complex64 {
    let r0 = nsin_fi / n;
    let v = cplx(1.0, 0.0) - r0 * r0;
    let c = csqrt_fast(v);
    if c.im < 0.0 {
        -c
    } else {
        c
    }}

/// Wave admittance: s-pol y = n·cosθ, p-pol y = n/cosθ (matches func_3).
#[inline(always)]
pub fn admittance(is_s: bool, n: Complex64, cos: Complex64) -> Complex64 {
    if is_s {
        n * cos
    } else {
        let c = if cos.norm() < EPS_COS {
            cplx(EPS_COS, 0.0)
        } else {
            cos
        };
        n / c
    }}

// ─── Plain Redheffer star product (delegates to optics_core) ────────────────

/// Complex field-amplitude star product: (A ⊗ B)(r_f, t_b, t_f, r_b).
/// Thin S4-tuple wrapper around the crate-shared implementation.
#[inline(always)]
pub fn star(a: S4, b: S4) -> S4 {
    redheffer_product_complex_field_inner(a.0, a.1, a.2, a.3, b.0, b.1, b.2, b.3)
}

/// Propagation element for phase `phi`.
#[inline(always)]
pub(crate) fn prop(phi: Complex64) -> S4 {
    (cplx(0.0, 0.0), phi, phi, cplx(0.0, 0.0))
}

/// Interface matrix between media i and i+1, including roughness factors.
#[inline(always)]
fn interface_matrix(
    i: usize,
    n: &[Complex64],
    cos: &[Complex64],
    rv: &[f64],
    rt: &[i32],
    two_pi_lam: f64,
    is_s: bool,
) -> S4 {
    let y_c = admittance(is_s, n[i], cos[i]);
    let y_next = admittance(is_s, n[i + 1], cos[i + 1]);

    let den = y_c + y_next;
    let den_safe = if den.norm() < LOG_MIN {
        cplx(LOG_MIN, LOG_MIN)
    } else {
        den
    };
    let inv_den = den_safe.recip();

    let r12 = (y_c - y_next) * inv_den;
    let r21 = -((y_c - y_next) * inv_den);
    let t12 = y_c * 2.0 * inv_den;
    let t21 = y_next * 2.0 * inv_den;

    let sigma = rv[i + 1];
    let rtype = rt[i + 1];
    let (r12m, r21m, t12m, t21m) = if rtype == 0 || sigma <= 0.0 {
        (r12, r21, t12, t21)
    } else if rtype == 5 {
        let kz1 = two_pi_lam * n[i] * cos[i];
        let kz2 = two_pi_lam * n[i + 1] * cos[i + 1];
        let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
        (r12 * f, r21 * f, t12 * f, t21 * f)
    } else {
        let kz1 = two_pi_lam * n[i] * cos[i];
        let kz2 = two_pi_lam * n[i + 1] * cos[i + 1];
        let al = w_function_inner(2.0 * kz1 * sigma, rtype);
        let be = w_function_inner(2.0 * kz2 * sigma, rtype);
        let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
        (r12 * al, r21 * be, t12 * ga, t21 * ga)
    };

    // Element convention matches func_3's composition operand:
    // (r_front, t_back, t_fwd, r_back).
    (r12m, t21m, t12m, r21m)
}

// ─── Stack decomposition into partial S-matrices ────────────────────────────

/// Partial S-matrices and per-layer propagation data for one
/// (wavelength, angle, polarization) point.
///
/// * `s_left[j]`  — block entrance down to the TOP plane of layer j
///   (after interface j-1|j, before propagation through layer j).
/// * `s_right[j]` — block exit up to the BOTTOM plane of layer j
///   (interface j|j+1 and everything below; layer j itself excluded).
/// * `beta_nm[j]` — k0·n·cosθ per nm of physical depth, Im >= 0 branch.
/// * `start`/`end` — the active block `[start, end)`; entries outside the
///   range are identity placeholders. Layers `start` and `end-1` act as the
///   block's boundary half-spaces; needle hosts must lie strictly inside.
///
/// Construction mirrors the sweep order of the validated solver loops
/// (`solve_coherent_block_fields_*` with `start_idx`/`end_idx`), so
/// `s_left[end].0` reproduces the block solver's front reflection amplitude
/// exactly (asserted by the tests below).
pub struct StackFields {
    pub s_left: Vec<S4>,
    pub s_right: Vec<S4>,
    pub n: Vec<Complex64>,
    pub cos: Vec<Complex64>,
    pub beta_nm: Vec<Complex64>,
    pub ds: Vec<f64>,
    pub start: usize,
    pub end: usize,
}

/// Build the partial-matrix decomposition over the FULL stack
/// (equivalent to `build_stack_fields_range(0, n_slice.len(), ...)`).
pub fn build_stack_fields(
    n_slice: &[Complex64],
    d_slice: &[f64],
    rv_slice: &[f64],
    rt_slice: &[i32],
    lam: f64,
    nsin_fi: Complex64,
    pol: i32,
) -> StackFields {
    let nl = n_slice.len();
    // Full stack: media 0..nl, substrate index nl-1 excluded as half-space.
    build_stack_fields_range(0, nl - 1, n_slice, d_slice, rv_slice, rt_slice, lam, nsin_fi, pol)
}

/// Build the partial-matrix decomposition confined to the sub-block
/// `[start_idx, end_idx)` of the full stack arrays (ABSOLUTE indexing kept
/// throughout, mirroring `solve_coherent_block_fields_inner`'s contract).
///
/// Medium `start_idx` plays the role of the block's ambient, medium `end_idx`
/// is excluded (plays the role of its substrate). Only interfaces strictly
/// inside the block are composed; layers outside are invisible to the
/// needle. All arrays stay full-length — nothing is re-indexed.
pub fn build_stack_fields_range(
    start_idx: usize,
    end_idx: usize,
    n_slice: &[Complex64],
    d_slice: &[f64],
    rv_slice: &[f64],
    rt_slice: &[i32],
    lam: f64,
    nsin_fi: Complex64,
    pol: i32,
) -> StackFields {
    assert!(start_idx < end_idx && end_idx < n_slice.len());
    let two_pi_lam = 2.0 * PI / lam;
    let is_s = pol == 0;
    let nl = n_slice.len();

    let cos: Vec<Complex64> = n_slice
        .iter()
        .map(|&n| cos_from_nsin(nsin_fi, n))
        .collect();

    let beta_nm: Vec<Complex64> = n_slice
        .iter()
        .zip(cos.iter())
        .map(|(&n, &c)| {
            let mut b = two_pi_lam * n * c;
            if b.im < 0.0 {
                b = cplx(b.re, -b.im);
            }            b
        })
        .collect();

    let id = || (cplx(0.0, 0.0), cplx(1.0, 0.0), cplx(1.0, 0.0), cplx(0.0, 0.0));
    let mut s_left: Vec<S4> = vec![id(); nl];
    let mut s_right: Vec<S4> = vec![id(); nl];

    // s_left: block entrance → top of layer j.
    for i in start_idx..end_idx {
        let mut sg = s_left[i];
        if i > start_idx && d_slice[i] > 1e-12 {
            sg = star(sg, prop(cexp_fast(cplx(0.0, 1.0) * beta_nm[i] * d_slice[i])));
        }        sg = star(sg, interface_matrix(i, n_slice, &cos, rv_slice, rt_slice, two_pi_lam, is_s));
        s_left[i + 1] = sg;
    }    // s_right: block exit → bottom of layer j.
    for i in (start_idx..end_idx).rev() {
        let mut sg = s_right[i + 1];
        if i + 1 < end_idx && d_slice[i + 1] > 1e-12 {
            sg = star(prop(cexp_fast(cplx(0.0, 1.0) * beta_nm[i + 1] * d_slice[i + 1])), sg);
        }        sg = star(interface_matrix(i, n_slice, &cos, rv_slice, rt_slice, two_pi_lam, is_s), sg);
        s_right[i] = sg;
    }    StackFields {
        s_left,
        s_right,
        n: n_slice.to_vec(),
        cos,
        beta_nm,
        ds: d_slice.to_vec(),
        start: start_idx,
        end: end_idx,
    }}

// ─── Complex dual numbers: value + d/dδ slope ──────────────────────────────

/// Complex dual number `v + d·δ`. Arithmetic propagates first-order
/// derivatives through the star product exactly.
#[derive(Clone, Copy, Debug)]
struct CDual {
    v: Complex64,
    d: Complex64,
}

impl CDual {
    #[inline(always)]
    fn cst(v: Complex64) -> Self {
        CDual { v, d: cplx(0.0, 0.0) }
    }    #[inline(always)]
    fn mul(self, o: CDual) -> CDual {
        CDual {
            v: self.v * o.v,
            d: self.d * o.v + self.v * o.d,
        }    }    #[inline(always)]
    fn sub(self, o: CDual) -> CDual {
        CDual {
            v: self.v - o.v,
            d: self.d - o.d,
        }    }    #[inline(always)]
    fn recip(self) -> CDual {
        let inv = self.v.recip();
        CDual {
            v: inv,
            d: -(inv * self.d * inv),
        }    }}

/// Dual-valued star product — mechanical translation of `star` above.
/// The regularization acts on the value component and rescales the slope by
/// the same complex factor, preserving consistency far from singularities
/// (where it never activates in normal operation anyway).
#[inline]
fn star_dual(a: (CDual, CDual, CDual, CDual), b: (CDual, CDual, CDual, CDual)) -> (CDual, CDual, CDual, CDual) {
    let one = CDual::cst(cplx(1.0, 0.0));
    let mut denom = one.sub(a.3.mul(b.0));
    if denom.v.norm() < LOG_MIN {
        let phase = denom.v / (denom.v.abs() + 1e-300);
        let repl = cplx(LOG_MIN, 0.0) * phase + 1e-300;
        let fac = repl / denom.v;
        denom = CDual { v: repl, d: denom.d * fac };
    }    let inv_denom = denom.recip();

    // s_rf = a.0 + a.1*b.0*a.2*inv ; etc., all dual ops.
    let term = a.1.mul(b.0).mul(a.2).mul(inv_denom);
    let s_rf = CDual { v: a.0.v + term.v, d: a.0.d + term.d };
    let s_tb = a.1.mul(b.1).mul(inv_denom);
    let s_tf = b.2.mul(a.2).mul(inv_denom);
    let term_b = b.2.mul(a.3).mul(b.1).mul(inv_denom);
    let s_rb = CDual { v: b.3.v + term_b.v, d: b.3.d + term_b.d };
    (s_rf, s_tb, s_tf, s_rb)
}

/// Needle slopes ρ̂ = −2iβ′r₁₂/(1−r₁₂²), τ̂ = 2iβ′/(1−r₁₂²) for a candidate
/// material `n_prime` embedded in host medium (`n_host`, `cos_host`).
#[inline]
pub fn needle_slopes(
    n_host: Complex64,
    cos_host: Complex64,
    n_prime: Complex64,
    nsin_fi: Complex64,
    is_s: bool,
    two_pi_lam: f64,
) -> (Complex64, Complex64) {
    let cos_p = cos_from_nsin(nsin_fi, n_prime);
    let y_h = admittance(is_s, n_host, cos_host);
    let y_n = admittance(is_s, n_prime, cos_p);
    let beta_p = two_pi_lam * n_prime * cos_p;

    let den = y_h + y_n;
    let r12 = (y_h - y_n) / den;
    let one_m_r12_sq = cplx(1.0, 0.0) - r12 * r12;
    // Both slopes verified against the exact thin-slab star product:
    //   rho_hat = −2i·β′·r12 / (1−r12²)
    //   tau_hat =  i·β′·(1+r12²) / (1−r12²)   (Fabry–Pérot-enhanced phase)
    let rho_hat = cplx(0.0, -2.0) * beta_p * r12 / one_m_r12_sq;
    let tau_hat = cplx(0.0, 1.0) * beta_p * (cplx(1.0, 0.0) + r12 * r12) / one_m_r12_sq;
    (rho_hat, tau_hat)
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Sensitivity of the stack's complex front-reflection amplitude to inserting
/// a unit-thickness needle of material `n_prime` at depth `xi` inside layer
/// `j` (measured from the layer's top plane), at one spectral point.
///
/// Exact to first order in the needle thickness. Cost: O(1) — two plain star
/// products to reference the cut planes (precomputable via `StackFields`)
/// plus two dual star products. Confined to the block in `fields`: the host
/// layer `j` must satisfy `fields.start < j < fields.end`; media outside the
/// block are invisible to the sensitivity.
/// Four-channel needle slopes: (∂r_front/∂δ, ∂t_back/∂δ, ∂t_fwd/∂δ,
/// ∂r_back/∂δ) of the BLOCK containing host layer `j`, all exact to first
/// order via one dual-number composition U ⊗ N_dual ⊗ L.
pub fn needle_slopes4_ddz(
    fields: &StackFields,
    nsin_fi: Complex64,
    j: usize,
    xi: f64,
    n_prime: Complex64,
    pol: i32,
    lam: f64,
) -> [Complex64; 4] {
    let two_pi_lam = 2.0 * PI / lam;
    debug_assert!(
        fields.start < j && j < fields.end,
        "needle host must be interior to the active block"
    );
    debug_assert!(xi >= -1e-9 && xi <= fields.ds[j] + 1e-9, "xi outside host layer");

    let bj = fields.beta_nm[j];
    // Upper part: ambient → plane z (propagate xi into layer j).
    let pp = cexp_fast(cplx(0.0, 1.0) * bj * xi);
    let u = star(fields.s_left[j], prop(pp));
    // Lower part: plane z → substrate (remaining host thickness, then below).
    let pr = cexp_fast(cplx(0.0, 1.0) * bj * (fields.ds[j] - xi));
    let l = star(prop(pr), fields.s_right[j]);

    let (rho_hat, tau_hat) = needle_slopes(
        fields.n[j],
        fields.cos[j],
        n_prime,
        nsin_fi,
        pol == 0,
        two_pi_lam,
    );

    // Needle S-matrix to first order: symmetric slab ⇒ equal front/back
    // reflection slopes; transmission phase slope identical both ways.
    let needle = (
        CDual { v: cplx(0.0, 0.0), d: rho_hat },
        CDual { v: cplx(1.0, 0.0), d: tau_hat },
        CDual { v: cplx(1.0, 0.0), d: tau_hat },
        CDual { v: cplx(0.0, 0.0), d: rho_hat },
    );
    let du = (
        CDual::cst(u.0),
        CDual::cst(u.1),
        CDual::cst(u.2),
        CDual::cst(u.3),
    );
    let dl = (
        CDual::cst(l.0),
        CDual::cst(l.1),
        CDual::cst(l.2),
        CDual::cst(l.3),
    );

    let m = star_dual(star_dual(du, needle), dl);
    [m.0.d, m.1.d, m.2.d, m.3.d]
}

/// Front-reflection needle sensitivity — thin wrapper over
/// [`needle_slopes4_ddz`] extracting channel 0.
pub fn needle_dr_ddz(
    fields: &StackFields,
    nsin_fi: Complex64,
    j: usize,
    xi: f64,
    n_prime: Complex64,
    pol: i32,
    lam: f64,
) -> Complex64 {
    needle_slopes4_ddz(fields, nsin_fi, j, xi, n_prime, pol, lam)[0]
}

/// Map absolute stack depth `z` (0 = top of the first needle-eligible layer
/// inside the block) to `(host_layer_index, depth_inside_layer)`.
/// Boundary-exact depths attach to the upper layer at ξ = d_j.
pub fn locate_depth(ds: &[f64], z: f64) -> (usize, f64) {
    locate_depth_in(ds, 1, ds.len() - 1, z)
}

/// Block-confined variant of [`locate_depth`]: hosts are the interior layers
/// `start_idx + 1 .. end_idx` of the block `[start_idx, end_idx)`.
pub fn locate_depth_in(ds: &[f64], start_idx: usize, end_idx: usize, z: f64) -> (usize, f64) {
    assert!(end_idx >= start_idx + 3, "block has no interior host layer");
    let mut cursor = 0.0;
    for j in start_idx + 1..end_idx {
        let bottom = cursor + ds[j];
        if z < bottom || j == end_idx - 1 {
            return (j, z - cursor);
        }        cursor = bottom;
    }    unreachable!("host loop always terminates via end-of-block arm")
}

// ─── Per-point kernels (grid drivers and parallel engines build on these) ──

/// Coherent P contribution of ONE spectral point, evaluated from PREBUILT
/// block fields (callers sweeping multiple observables build once, reuse
/// here). Half-gradient convention matching [`p_function`]: returns
/// 2·weight·(|r|² − target)·Re{conj(r)·∂r/∂δ} per z in `z_grid`.
pub fn p_coherent_from_fields(
    fields: &StackFields,
    nsin_fi: Complex64,
    lam: f64,
    pol: i32,
    needle_n: Complex64,
    target: f64,
    weight: f64,
    thicknesses: &[f64],
    start_idx: usize,
    end_idx: usize,
    z_grid: &[f64],
) -> Vec<f64> {
    let r_k = fields.s_left[end_idx].0;
    let resid = 2.0 * weight * (r_k.norm_sqr() - target);
    let rc = r_k.conj();
    let mut out = vec![0.0; z_grid.len()];
    for (zi, &z) in z_grid.iter().enumerate() {
        let (j, xi) = locate_depth_in(thicknesses, start_idx, end_idx, z);
        let dr = needle_dr_ddz(fields, nsin_fi, j, xi, needle_n, pol, lam);
        out[zi] = resid * (rc * dr).re;
    }
    out
}

/// Map every z in `z_grid` to its owning coherent block of an incoherent
/// stack: `(block_index, host_layer, xi)`. Rejects flagged spacer hosts;
/// masked-out hosts are skipped. Reference plane: top of film layer 1.
pub fn locate_hosts_multiblock(
    thicknesses: &[f64],
    incoherent_flags: &[i32],
    z_grid: &[f64],
    host_mask: Option<&[bool]>,
) -> Result<Vec<(usize, usize, f64)>, String> {
    let nl = thicknesses.len();
    let (blocks, _) = partition_blocks(incoherent_flags);
    let mut locs = Vec::with_capacity(z_grid.len());
    for (zi, &z) in z_grid.iter().enumerate() {
        let (j, xi) = locate_depth_in(thicknesses, 0, nl - 1, z);
        if incoherent_flags[j] == 1 {
            return Err(format!(
                "z_grid[{zi}]: host layer {j} is incoherent-flagged; \
                 needles inside spacers are not supported"
            ));
        }
        if let Some(mask) = host_mask {
            if !mask.get(j).copied().unwrap_or(false) {
                continue;
            }
        }
        let bi = blocks
            .iter()
            .position(|&(bs, be)| bs < j && j < be)
            .ok_or_else(|| format!("z_grid[{zi}]: host layer {j} lies in no coherent block"))?;
        locs.push((bi, j, xi));
    }
    Ok(locs)
}

/// Multiblock (Mode A/B) P contribution of ONE spectral point given the
/// precomputed host map from [`locate_hosts_multiblock`]. Same half-gradient
/// convention as [`p_coherent_from_fields`].
#[allow(clippy::too_many_arguments)]
pub fn p_multiblock_point(
    lam: f64,
    sin_theta: f64,
    n_slice: &[Complex64],
    thicknesses: &[f64],
    incoherent_flags: &[i32],
    rough_vals: &[f64],
    rough_types: &[i32],
    needle_n: Complex64,
    target: f64,
    weight: f64,
    locs: &[(usize, usize, f64)],
    pol: i32,
) -> Vec<f64> {
    let nsin_fi = n_slice[0] * cplx(sin_theta, 0.0);
    let (blocks, spacers) = partition_blocks(incoherent_flags);

    let fields: Vec<StackFields> = blocks
        .iter()
        .map(|&(bs, be)| {
            build_stack_fields_range(bs, be, n_slice, thicknesses, rough_vals, rough_types, lam, nsin_fi, pol)
        })
        .collect();

    let mut track = CascadeTrack::identity();
    for (bi, _) in blocks.iter().enumerate() {
        let inten = block_intensities(&fields[bi], pol);
        track = cascade_step(&track, inten, Some(()));
        if let Some(sp) = spacers[bi] {
            let tau = spacer_tau(n_slice[sp], thicknesses[sp], lam, nsin_fi);
            track = cascade_step(&track, [0.0, tau, tau, 0.0], None);
        }
    }
    let r_tot = track.v[0];
    let resid = 2.0 * weight * (r_tot - target);

    let mut out = vec![0.0; locs.len()];
    if resid == 0.0 {
        return out;
    }

    for (zi, &(bi, j, xi)) in locs.iter().enumerate() {
        let slopes = needle_slopes4_ddz(&fields[bi], nsin_fi, j, xi, needle_n, pol, lam);
        let amp = fields[bi].s_left[fields[bi].end];
        let fac = block_flux_factors(&fields[bi], pol);
        let g_int = [
            fac[0] * (amp.0.conj() * slopes[0]).re,
            fac[1] * (amp.1.conj() * slopes[1]).re,
            fac[2] * (amp.2.conj() * slopes[2]).re,
            fac[3] * (amp.3.conj() * slopes[3]).re,
        ];
        let wv = &track.g[bi * 4..bi * 4 + 4];
        let dot = wv[0][0] * g_int[0]
            + wv[1][0] * g_int[1]
            + wv[2][0] * g_int[2]
            + wv[3][0] * g_int[3];
        out[zi] = resid * dot;
    }
    out
}

/// Tikhonravov merit-function P-function, confined to the sub-block
/// `[start_idx, end_idx)`.
///
/// P(z) = Σ_k 2·w_k·(R_k − Rt_k)·Re{ conj(r_k) · ∂r_k/∂δ (z) }
///
/// accumulated over all (wavelength, angle) points for ONE polarization.
/// `R_k`/`r_k` are the BLOCK front reflection (media outside the block are
/// invisible). Arrays follow the crate-wide layout:
///   * `n_stack_cache`: flat f64, per wavelength 2 entries per layer (re, im),
///   * `target_r`/`weights`: flat, angle-major ordering k = a*num_wavs + w,
///   * `z_grid`: absolute depths from the top of layer `start_idx + 1`,
///   * `pol`: 0 = s, 1 = p.
///
/// Fully coherent stacks only. Returns P evaluated at each z in `z_grid`.
#[allow(clippy::too_many_arguments)]
pub fn p_function(
    wavls: &[f64],
    sin_theta_arr: &[f64],
    start_idx: usize,
    end_idx: usize,
    n_layers: usize,
    n_stack_cache: &[f64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    needle_n_per_wav: &[Complex64],
    target_r: &[f64],
    weights: &[f64],
    z_grid: &[f64],
    pol: i32,
) -> Result<Vec<f64>, String> {
    let num_wavs = wavls.len();
    let num_angles = sin_theta_arr.len();
    if num_wavs == 0 || num_angles == 0 || n_layers < 3 {
        return Err("empty spectral grid or degenerate stack".into());
    }
    if end_idx <= start_idx + 2 || end_idx >= n_layers {
        return Err("block must contain at least one interior host layer".into());
    }
    let total_points = num_wavs * num_angles;
    if target_r.len() != total_points || weights.len() != total_points
        || needle_n_per_wav.len() != num_wavs
        || n_stack_cache.len() != num_wavs * n_layers * 2
    {
        return Err("array length mismatch".into());
    }
    let mut acc = vec![0.0f64; z_grid.len()];

    for a in 0..num_angles {
        for w in 0..num_wavs {
            let k = a * num_wavs + w;
            let lam = wavls[w];

            let mut n_slice = Vec::with_capacity(n_layers);
            let base = w * n_layers * 2;
            for l in 0..n_layers {
                n_slice.push(cplx(n_stack_cache[base + l * 2], n_stack_cache[base + l * 2 + 1]));
            }            let nsin_fi = n_slice[0] * cplx(sin_theta_arr[a], 0.0);

            let fields = build_stack_fields_range(
                start_idx, end_idx,
                &n_slice, thicknesses, rough_vals, rough_types, lam, nsin_fi, pol,
            );
            let contrib = p_coherent_from_fields(
                &fields, nsin_fi, lam, pol, needle_n_per_wav[w],
                target_r[k], weights[k], thicknesses, start_idx, end_idx, z_grid,
            );
            for (zi, cv) in contrib.iter().enumerate() {
                acc[zi] += cv;
            }
        }
    }
    Ok(acc)
}

// ─── Incoherent (Mode A/B) block cascade adjoint ──────────────────────────
//
// A stack with incoherent-flagged spacer layers is solved by the engine as a
// chain of COHERENT blocks joined by the REAL (intensity) Redheffer star
// product, with each spacer contributing an exp(-2·Im β·d) attenuation
// element — see `solve_point` in func_4. Phases only live inside blocks, so
// the merit gradient decomposes exactly:
//
//   dMerit/dδ = Σ_k 2 w_k (R_tot − Rt_k) · Σ_b Σ_c W[b][c] · dI_{b,c}/dδ
//
// where I_b = (R_f, T_b, T_f, R_b) are block b's four intensity outputs,
// W[b] = ∂R_tot/∂I_b are adjoint weights obtained by differentiating the
// real-star cascade (forward-mode, hand-derived Jacobians below), and
// dI/dδ = 2 Re{conj(a)·da/dδ} per channel from the analytic needle operator.
// The admittance normalization factors Re y_end / Re y_start depend only on
// the block's boundary half-spaces, never on the needle host ⇒ δ-independent.

/// Real-intensity star product — thin [f64; 4] wrapper around the
/// crate-shared `redheffer_product_real_inner` (identical regularization).
#[inline]
pub fn star_real(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let (rf, tb, tf, rb) =
        redheffer_product_real_inner(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
    [rf, tb, tf, rb]
}

/// One real-star step with forward-mode gradient tracking.
///
/// Element order throughout: [r_front, t_back, t_fwd, r_back]. The right
/// operand's parameters (a block's four intensity outputs) are appended as
/// gradient slots when `new_params` is Some(base); constant elements (tau
/// spacers) pass None and only back-propagate the accumulated gradients
/// through the left Jacobian. When the regularized denominator zeroes V,
/// value AND gradients follow the same convention as `star_real` (all V
/// terms vanish).
#[inline]
pub fn cascade_step(a: &CascadeTrack, b: [f64; 4], new_params: Option<()>) -> CascadeTrack {
    let denom = 1.0 - a.v[3] * b[0];
    let v = if denom.abs() < DBL_EPS { 0.0 } else { 1.0 / denom };
    let v2 = v * v;

    let out = [
        a.v[0] + a.v[1] * b[0] * a.v[2] * v,
        a.v[1] * b[1] * v,
        b[2] * a.v[2] * v,
        b[3] + b[2] * a.v[3] * b[1] * v,
    ];

    // ∂out/∂A (rows: out element e, cols: A element x):
    // out.g[p][e] += JA[e][x] · a.g[p][x].
    let ja: [[f64; 4]; 4] = [
        [1.0, b[0] * a.v[2] * v, a.v[1] * b[0] * v, a.v[1] * b[0] * a.v[2] * b[0] * v2],
        [0.0, b[1] * v, 0.0, a.v[1] * b[1] * b[0] * v2],
        [0.0, 0.0, b[2] * v, b[2] * a.v[2] * b[0] * v2],
        [0.0, 0.0, 0.0, b[2] * b[1] * (v + a.v[3] * b[0] * v2)],
    ];
    // ∂out/∂B (rows: out element e, cols: B element c). Note V depends on
    // B.rf, so every V-carrying output picks up a ∂/∂B.rf term.
    let jb: [[f64; 4]; 4] = [
        [a.v[1] * a.v[2] * v2, 0.0, 0.0, 0.0],
        [a.v[1] * b[1] * a.v[3] * v2, a.v[1] * v, 0.0, 0.0],
        [b[2] * a.v[2] * a.v[3] * v2, 0.0, a.v[2] * v, 0.0],
        [b[2] * a.v[3] * b[1] * a.v[3] * v2, b[2] * a.v[3] * v, a.v[3] * b[1] * v, 1.0],
    ];

    let n_old = a.g.len();
    let mut g: Vec<[f64; 4]> = Vec::with_capacity(n_old + 4);
    for gp in &a.g {
        let mut row = [0.0; 4];
        for (e, rowe) in row.iter_mut().enumerate() {
            *rowe = ja[e][0] * gp[0] + ja[e][1] * gp[1] + ja[e][2] * gp[2] + ja[e][3] * gp[3];
        }
        g.push(row);
    }
    if new_params.is_some() {
        // Gradient rows are indexed by PARAMETER: g[p][e] = ∂out_e/∂param_p.
        // That is COLUMN c of the Jacobian (∂out_e/∂B_c), not its rows.
        for c in 0..4 {
            g.push([jb[0][c], jb[1][c], jb[2][c], jb[3][c]]);
        }
    }

    CascadeTrack { v: out, g }
}

/// Real-intensity accumulator plus per-parameter gradient rows.
/// `g[p][e]` = ∂v[e]/∂param_p, where params are appended 4-per-block in
/// sweep order: (R_f, T_b, T_f, R_b) of each block's intensity tuple.
pub struct CascadeTrack {
    pub v: [f64; 4],
    pub g: Vec<[f64; 4]>,
}

impl CascadeTrack {
    pub fn identity() -> Self {
        CascadeTrack { v: [0.0, 1.0, 1.0, 0.0], g: Vec::new() }
    }
}

/// Flux-normalized intensity outputs of one coherent block, byte-mirroring
/// the finalize step of `solve_pol_specialized` in func_3: T channels carry
/// the Re-admittance ratio of the block's boundary half-spaces (clamped at
/// 1e-15); p-pol admittance carries the EPS_COS guard. Boundary media are
/// needle-invariant, so these factors are constants under δ.
pub fn block_intensities(fields: &StackFields, pol: i32) -> [f64; 4] {
    let m = fields.s_left[fields.end]; // composed block matrix (rf, tb, tf, rb)
    let f = block_flux_factors(fields, pol);
    [
        m.0.norm_sqr() * f[0],
        m.1.norm_sqr() * f[1],
        m.2.norm_sqr() * f[2],
        m.3.norm_sqr() * f[3],
    ]
}

/// Flux-normalization multipliers per intensity channel: [1, T_back factor,
/// T_fwd factor, 1]. The Re-admittance ratios of the block's boundary
/// half-spaces (clamped exactly like func_3); boundary media are
/// needle-invariant, so these are constants under δ.
pub fn block_flux_factors(fields: &StackFields, pol: i32) -> [f64; 4] {
    const EPS_COS: f64 = 1e-12;
    let is_s = pol == 0;
    let adm = |n: Complex64, cos: Complex64| -> Complex64 {
        if is_s {
            n * cos
        } else {
            let c = if cos.norm() < EPS_COS {
                cplx(EPS_COS, 0.0)
            } else {
                cos
            };
            n / c
        }
    };
    let y_first = adm(fields.n[fields.start], fields.cos[fields.start]);
    let y_last = adm(fields.n[fields.end], fields.cos[fields.end]);

    let mut ry0 = y_first.re;
    let mut ry1 = y_last.re;
    if ry0 < 1e-15 {
        ry0 = 0.0;
    }
    if ry1 < 1e-15 {
        ry1 = 0.0;
    }
    let f_fwd = if ry0 > 1e-15 { ry1 / ry0 } else { 0.0 };
    let f_back = if ry1 > 1e-15 { ry0 / ry1 } else { 0.0 };
    [1.0, f_back, f_fwd, 1.0]
}

/// Partition media `[0, nl-1)` into coherent blocks separated by
/// incoherent-flagged spacer layers, mirroring func_4's sweep. Returns
/// `(blocks, spacers)`: `blocks[i] = (start_idx, end_idx)` and `spacers[i]`
/// is the flagged medium index following block i (None when absent or at
/// the substrate boundary). A flagged medium doubles as the entrance
/// half-space of the next block — engine convention preserved.
pub fn partition_blocks(incoherent_flags: &[i32]) -> (Vec<(usize, usize)>, Vec<Option<usize>>) {
    let idx_n = incoherent_flags.len() - 1;
    let mut blocks = Vec::new();
    let mut spacers = Vec::new();
    let mut cur = 0usize;
    while cur < idx_n {
        let mut ni = cur + 1;
        while ni < idx_n && incoherent_flags[ni] == 0 {
            ni += 1;
        }
        blocks.push((cur, ni));
        let spacer = if ni < idx_n && incoherent_flags[ni] == 1 {
            Some(ni)
        } else {
            None
        };
        spacers.push(spacer);
        cur = ni;
    }
    (blocks, spacers)
}

/// Spacer attenuation element exp(-2·max(0, Im β)·d), byte-mirroring the
/// tau computation in func_4's solve loops.
pub fn spacer_tau(n_inc: Complex64, d_inc: f64, lam: f64, nsin_fi: Complex64) -> f64 {
    let rinc = nsin_fi / n_inc;
    let val = cplx(1.0, 0.0) - rinc * rinc;
    let mut cos_inc = val.sqrt();
    if cos_inc.im < 0.0 {
        cos_inc = -cos_inc;
    }
    let beta_imag = (2.0 * PI * d_inc / lam) * (n_inc * cos_inc).im;
    let beta_imag = if beta_imag < 0.0 { 0.0 } else { beta_imag };
    (-2.0 * beta_imag).exp()
}

/// Merit P-function for stacks containing INCOHERENT layers (engine Modes
/// A/B). Identical contract to [`p_function`] except:
///   * `incoherent_flags` splits the stack into coherent blocks joined by
///     the real intensity cascade (func_2/func_4 conventions),
///   * `host_mask`: optional per-layer eligibility (None = all interior
///     coherent layers eligible). Masked z-points simply receive no
///     contribution; needles may not target flagged spacer layers.
///
/// Gradient path: per spectral point, block matrices come from the same
/// range-built partial S-matrices as the coherent solver, intensities are
/// flux-normalized exactly like func_3's finalize, and adjoint weights
/// W[b][c] = ∂R_tot/∂I_{b,c} propagate backwards through hand-derived real-
/// star Jacobians (validated against finite differences in the tests).
pub fn p_function_multiblock(
    wavls: &[f64],
    sin_theta_arr: &[f64],
    n_stack_cache: &[f64],
    thicknesses: &[f64],
    incoherent_flags: &[i32],
    rough_types: &[i32],
    rough_vals: &[f64],
    needle_n_per_wav: &[Complex64],
    target_r: &[f64],
    weights: &[f64],
    z_grid: &[f64],
    host_mask: Option<&[bool]>,
    pol: i32,
) -> Result<Vec<f64>, String> {
    let num_wavs = wavls.len();
    let num_angles = sin_theta_arr.len();
    let nl = thicknesses.len();
    if num_wavs == 0 || num_angles == 0 || nl < 3 {
        return Err("empty spectral grid or degenerate stack".into());
    }
    if incoherent_flags.len() != nl || rough_types.len() != nl || rough_vals.len() != nl {
        return Err("per-layer arrays must all have n_layers entries".into());
    }


    // Geometry is wavelength-independent: map every z once via the shared
    // host-locator (reference plane: top of film layer 1).
    let locs = locate_hosts_multiblock(thicknesses, incoherent_flags, z_grid, host_mask)?;

    let mut p_out = vec![0.0; z_grid.len()];
    for a in 0..num_angles {
        for w in 0..num_wavs {
            let k = a * num_wavs + w;
            let base_n = w * nl * 2;
            let ns: Vec<Complex64> = (0..nl)
                .map(|l| cplx(n_stack_cache[base_n + l * 2], n_stack_cache[base_n + l * 2 + 1]))
                .collect();
            let contrib = p_multiblock_point(
                wavls[w], sin_theta_arr[a], &ns, thicknesses, incoherent_flags,
                rough_vals, rough_types, needle_n_per_wav[w],
                target_r[k], weights[k], &locs, pol,
            );
            for (zi, cv) in contrib.iter().enumerate() {
                p_out[zi] += cv;
            }
        }
    }
    Ok(p_out)
}


// ─── Dispersion (GD/GDD) sensitivity via spectral differentiation ─────────
//
// The phase of a coherent stack, phi = arg(a), has an EXACT analytic needle
// sensitivity: since dr/dδ is a smooth (C^inf) function of wavelength,
//
//     Q(λ, z) = ∂phi/∂δ = Im{ conj(a)·∂a/∂δ } / |a|²
//
// can be evaluated pointwise and then differentiated w.r.t. angular frequency
// ω = 2πc/λ with the SAME non-uniform central-difference operator the engine
// uses for its GD/GDD/TOD/FOD post-pass. Differentiating the smooth
// sensitivity function converges far better than finite-differencing noisy
// solver GDD values in δ. Convention: order 0 -> Q, 1 -> ∂GD/∂δ,
// 2 -> ∂GDD/∂δ, 3 -> ∂TOD/∂δ, 4 -> ∂FOD/∂δ (GD in fs when λ is in nm).
// Phase is only defined coherently, so — like the engine's dispersion pass —
// this operates on ONE coherent range [start_idx, end_idx], not across
// incoherent joins.

/// Speed of light in nm/fs: with λ in nm, ω = 2π·c/λ is in rad/fs, so GD is
/// in fs and GDD in fs² (mirrors func_4's C_NM_PER_FS).
pub use crate::optics_core::grad_nonuniform;

/// One spectral differentiation sweep over rows laid out as `[k][zi]`
/// (k = a*num_wavs + w, angle-major). Exposed so callers can differentiate
/// their own sensitivity rows and tests can inject synthetic data.
pub fn spectral_gradient_step(
    q: &[Vec<f64>],
    omega: &[f64],
    num_wavs: usize,
    num_angles: usize,
    nz: usize,
) -> Vec<Vec<f64>> {
    let total = num_wavs * num_angles;
    let mut next = vec![vec![0.0; nz]; total];
    for a in 0..num_angles {
        for zi in 0..nz {
            let row: Vec<f64> = (0..num_wavs).map(|w| q[a * num_wavs + w][zi]).collect();
            let d = grad_nonuniform(&row, omega);
            for (w, dv) in d.into_iter().enumerate() {
                next[a * num_wavs + w][zi] = dv;
            }
        }
    }
    next
}

/// Sensitivity spectrum of the phase-dispersion channels to needle insertion.
///
/// For every spectral point k (angle-major: k = a*num_wavs + w) and every z
/// in `z_grid` (absolute depths from the top of layer `start_idx + 1`),
/// returns ∂(dⁿφ_channel/dωⁿ)/∂δ. Layout: `out[order][k*nz + zi]` with
/// `order = 0..=deriv_order`, i.e. a triple-nested Vec.
///
/// * `channel`: element of the composed block matrix whose phase is tracked:
///   0 = r_front, 1 = t_back, 2 = t_fwd, 3 = r_back.
/// * hosts are located once (wavelength-independent geometry); each must be
///   strictly interior to `[start_idx, end_idx]`.
/// * where |a|² < 1e-20 the phase is degenerate and Q is set to 0.
pub fn phase_dispersion_sensitivity(
    wavls: &[f64],
    sin_theta_arr: &[f64],
    start_idx: usize,
    end_idx: usize,
    n_stack_cache: &[f64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    needle_n_per_wav: &[Complex64],
    z_grid: &[f64],
    pol: i32,
    channel: usize,
    deriv_order: usize,
) -> Result<Vec<Vec<Vec<f64>>>, String> {
    let num_wavs = wavls.len();
    let num_angles = sin_theta_arr.len();
    let nl = thicknesses.len();
    let nz = z_grid.len();
    if num_wavs == 0 || num_angles == 0 || nl < 3 || nz == 0 {
        return Err("empty spectral grid or degenerate stack".into());
    }
    if end_idx <= start_idx + 2 || end_idx >= nl {
        return Err("range must contain at least one interior host layer".into());
    }
    if channel > 3 {
        return Err("channel must be one of 0..=3".into());
    }
    if deriv_order > 4 {
        return Err("deriv_order up to 4 (FOD) supported".into());
    }
    if needle_n_per_wav.len() != num_wavs {
        return Err("needle_n_per_wav must have one entry per wavelength".into());
    }
    if n_stack_cache.len() != num_wavs * nl * 2 {
        return Err("n_stack_cache layout mismatch".into());
    }

    // Geometry is wavelength-independent: map every z to its host plane once.
    let locs: Vec<(usize, f64)> = z_grid
        .iter()
        .map(|&z| locate_depth_in(thicknesses, start_idx, end_idx, z))
        .collect();

    // Q(λ, z): phase-sensitivity spectrum, [k][zi].
    let mut q = vec![vec![0.0_f64; nz]; num_wavs * num_angles];
    for a in 0..num_angles {
        for w in 0..num_wavs {
            let lam = wavls[w];
            let base = w * nl * 2;
            let mut ns = Vec::with_capacity(nl);
            for l in 0..nl {
                ns.push(cplx(
                    n_stack_cache[base + l * 2],
                    n_stack_cache[base + l * 2 + 1],
                ));
            }
            let sin_v = sin_theta_arr[a];
            let nsin_fi = ns[0] * cplx(sin_v, 0.0);
            let fields = build_stack_fields_range(
                start_idx, end_idx, &ns, thicknesses, rough_vals, rough_types, lam, nsin_fi, pol,
            );
            let amp = match channel {
                0 => fields.s_left[end_idx].0,
                1 => fields.s_left[end_idx].1,
                2 => fields.s_left[end_idx].2,
                _ => fields.s_left[end_idx].3,
            };
            let r2 = amp.norm_sqr();
            if r2 <= 1e-20 {
                continue; // degenerate phase; Q stays 0
            }
            let np = needle_n_per_wav[w];
            let k = a * num_wavs + w;
            for (zi, &(j, xi)) in locs.iter().enumerate() {
                let dr = needle_dr_ddz(&fields, nsin_fi, j, xi, np, pol, lam);
                q[k][zi] = (amp.conj() * dr).im / r2;
            }
        }
    }

    // Spectral differentiation chain in ω (per angle, per z row).
    let omega: Vec<f64> = wavls.iter().map(|&l| 2.0 * PI * C_NM_PER_FS / l).collect();
    let mut out = vec![q];
    for _ in 0..deriv_order {
        let prev = out.last().unwrap();
        out.push(spectral_gradient_step(prev, &omega, num_wavs, num_angles, nz));
    }
    Ok(out)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn n_(re: f64, im: f64) -> Complex64 {
        cplx(re, im)
    }    /// Stack builder: (material, thickness nm) list, ambient first, substrate
    /// last. Roughness arrays default to zero (abrupt interfaces).
    fn make_stack(layers: &[(Complex64, f64)]) -> (Vec<Complex64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        let n: Vec<Complex64> = layers.iter().map(|&(m, _)| m).collect();
        let d: Vec<f64> = layers.iter().map(|&(_, t)| t).collect();
        let len = layers.len();
        (n, d, vec![0.0; len], vec![0; len])
    }    fn solve_r(
        n: &[Complex64], d: &[f64], rv: &[f64], rt: &[i32],
        lam: f64, sin_t: f64, pol: i32,
    ) -> Complex64 {
        let nsin = n[0] * cplx(sin_t, 0.0);
        let f = build_stack_fields(n, d, rv, rt, lam, nsin, pol);
        f.s_left[f.s_left.len() - 1].0
    }    const LAM: f64 = 550.0;

    fn stack_a() -> (Vec<Complex64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        make_stack(&[
            (n_(1.0, 0.0), 0.0),   // air
            (n_(2.35, 0.0), 40.0), // TiO2-like
            (n_(1.45, 0.0), 80.0), // SiO2-like
            (n_(2.35, 0.0), 30.0),
            (n_(1.52, 0.0), 0.0), // glass
        ])
    }    fn stack_absorbing() -> (Vec<Complex64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        make_stack(&[
            (n_(1.0, 0.0), 0.0),
            (n_(2.35, 0.0), 40.0),
            (n_(1.8, 0.4), 50.0), // absorbing
            (n_(1.45, 0.0), 70.0),
            (n_(1.52, 0.0), 0.0),
        ])
    }    /// Insert a physical needle slab of thickness delta at depth xi inside
    /// layer j; returns modified arrays. Internal interfaces are abrupt; all
    /// original interface roughness entries shift past the insertion point.
    fn insert_needle(
        n: &[Complex64], d: &[f64], rv: &[f64], rt: &[i32],
        j: usize, xi: f64, n_prime: Complex64, delta: f64,
    ) -> (Vec<Complex64>, Vec<f64>, Vec<f64>, Vec<i32>) {
        let nl = n.len();
        // Host layer j splits into top part + needle + bottom part: the host
        // material therefore appears TWICE; net +2 array entries.
        let mut nn = Vec::with_capacity(nl + 2);
        nn.extend_from_slice(&n[..=j]);
        nn.push(n_prime);
        nn.push(n[j]);
        nn.extend_from_slice(&n[j + 1..]);

        let mut dd = Vec::with_capacity(nl + 1);
        dd.extend_from_slice(&d[..j]);
        dd.push(xi);
        dd.push(delta);
        dd.push(d[j] - xi);
        dd.extend_from_slice(&d[j + 1..]);

        let mut rr = Vec::with_capacity(nl + 1);
        rr.extend_from_slice(&rv[..=j]);
        rr.push(0.0);
        rr.push(0.0);
        rr.extend_from_slice(&rv[j + 1..]);

        let mut tt = Vec::with_capacity(nl + 1);
        tt.extend_from_slice(&rt[..=j]);
        tt.push(0);
        tt.push(0);
        tt.extend_from_slice(&rt[j + 1..]);

        (nn, dd, rr, tt)
    }    /// Finite-difference oracle vs analytic sensitivity.
    fn check_fd_case(
        name: &str,
        (n, d, rv, rt): &(Vec<Complex64>, Vec<f64>, Vec<f64>, Vec<i32>),
        j: usize, xi: f64,
        n_prime: Complex64, sin_t: f64, pol: i32,
    ) {
        let nsin = n[0] * cplx(sin_t, 0.0);
        let fields = build_stack_fields(n, d, rv, rt, LAM, nsin, pol);
        let r0 = solve_r(n, d, rv, rt, LAM, sin_t, pol);

        let delta = 5e-4_f64;
        let (nn, dd, rr, tt) = insert_needle(n, d, rv, rt, j, xi, n_prime, delta);
        let r_new = solve_r(&nn, &dd, &rr, &tt, LAM, sin_t, pol);

        let fd = (r_new - r0) / delta;
        let an = needle_dr_ddz(&fields, nsin, j, xi, n_prime, pol, LAM);

        let scale = fd.norm().max(an.norm()).max(1e-12);
        let err = (fd - an).norm() / scale;
        assert!(
            err < 2e-3,
            "{name}: fd={fd:.6e} analytic={an:.6e} rel_err={err:.2e}"
        );
    }    #[test]
    fn subblock_range_matches_sliced_arrays() {
        // Block [1, 4) on the full arrays must reproduce the amplitude AND
        // needle sensitivity of an independent solve on the sliced arrays.
        let (n, d, rv, rt) = stack_a();
        let (start, end) = (1usize, 4usize);
        let sin_t = 0.3_f64;
        let nsin = n[0] * cplx(sin_t, 0.0);
        let n_prime = n_(1.9, 0.0);

        let f_range = build_stack_fields_range(start, end, &n, &d, &rv, &rt, LAM, nsin, 0);
        let r_block = f_range.s_left[end].0;

        // Sliced solve: cut all arrays at the block boundaries; interface
        // roughness entries stay aligned (entry m describes iface(m, m+1)).
        let ns = &n[start..=end];
        let ds_ = &d[start..=end];
        let rs = &rv[start..=end];
        let ts = &rt[start..=end];
        let f_sliced = build_stack_fields(ns, ds_, rs, ts, LAM, nsin, 0);
        let r_sliced = f_sliced.s_left.last().unwrap().0;
        assert!((r_block - r_sliced).norm() < 1e-12);

        // Sensitivity at the same physical plane: absolute j=2 ↔ relative j=1.
        let dr_range = needle_dr_ddz(&f_range, nsin, 2, 33.0, n_prime, 0, LAM);
        let dr_sliced = needle_dr_ddz(&f_sliced, nsin, 1, 33.0, n_prime, 0, LAM);
        assert!((dr_range - dr_sliced).norm() / dr_range.norm() < 1e-12);
    }    #[test]
    fn fd_oracle_confined_to_subblock() {
        // Needles inside block [1, 4): finite-difference reference computed by
        // solving ONLY the sliced block (layers outside are excluded from the
        // physics), analytic value via the range-restricted fields.
        let st = stack_a();
        let (start, end) = (1usize, 4usize);
        let np = n_(2.6, 0.0);
        // Hosts must be strictly interior to the block: j in {2, 3}.
        for &(j, xi, pol, sin_t) in &[
            (2usize, 20.0f64, 0i32, 0.0f64),
            (3, 15.0, 1, 0.4),
            (2, 1e-7, 0, 0.25),
            (2, 80.0 - 1e-7, 1, 0.35),
        ] {
            let (ref n, ref d, ref rv, ref rt) = st;
            let nsin = n[0] * cplx(sin_t, 0.0);
            let f_range = build_stack_fields_range(start, end, n, d, rv, rt, LAM, nsin, pol);
            let r0 = f_range.s_left[end].0;

            let delta = 5e-4_f64;
            let (nn, dd, rr, tt) = insert_needle(n, d, rv, rt, j, xi, np, delta);
            // Insertion shifts layers ≥ j by +2, so the block's exit medium
            // (originally at absolute index `end`) now sits at `end + 2`.
            let off = if j < end { 2 } else { 0 };
            let ns = &nn[start..=end + off];
            let ds_ = &dd[start..=end + off];
            let rs = &rr[start..=end + off];
            let ts = &tt[start..=end + off];
            let f_new = build_stack_fields(ns, ds_, rs, ts, LAM, nsin, pol);
            let r_new = f_new.s_left.last().unwrap().0;

            let fd = (r_new - r0) / delta;
            let an = needle_dr_ddz(&f_range, nsin, j, xi, np, pol, LAM);
            let scale = fd.norm().max(an.norm()).max(1e-12);
            let err = (fd - an).norm() / scale;
            assert!(err < 2e-3, "j={j} xi={xi} pol={pol}: err={err:.2e}");
        }
    }
    #[test]
    fn p_function_block_confined_matches_manual_sum() {
        // Same as the full-stack consistency test but restricted to a block:
        // manual hand-sum must use the BLOCK reflection amplitude.
        let (n, d, rv, rt) = stack_a();
        let (start, end) = (1usize, 4usize);
        let wavls = [550.0f64];
        let angles = [0.0f64];
        let nl = n.len();
        let mut cache = Vec::new();
        for &m in &n {
            cache.push(m.re);
            cache.push(m.im);
        }
        let needle_per_wav = [n_(1.9, 0.0)];
        let target = [0.05f64];
        let weights = [1.0];
        let z = [33.0f64];

        let p = p_function(
            &wavls, &angles, start, end, nl, &cache, &d, &rt, &rv,
            &needle_per_wav, &target, &weights, &z, 0,
        )
        .unwrap();

        // Manual reference: block amplitude + block-local sensitivity.
        let nsin = n[0] * cplx(angles[0], 0.0);
        let f = build_stack_fields_range(start, end, &n, &d, &rv, &rt, wavls[0], nsin, 0);
        let r = f.s_left[end].0;
        let loc = locate_depth_in(&d, start, end, z[0]);
        let dr = needle_dr_ddz(&f, nsin, loc.0, loc.1, needle_per_wav[0], 0, wavls[0]);
        let expect = 2.0 * weights[0] * (r.norm_sqr() - target[0]) * (r.conj() * dr).re;
        assert!((p[0] - expect).abs() < 1e-12);
    }
    #[test]
    fn cascade_gradient_matches_finite_difference() {
        // Pure-machinery check: hand-derived real-star Jacobians vs central
        // FD of the cascade value w.r.t. one middle block's intensity inputs.
        let elems: [[f64; 4]; 3] = [
            [0.21, 0.87, 0.92, 0.13],
            [0.55, 0.60, 0.70, 0.08],
            [0.33, 0.95, 0.80, 0.02],
        ];
        let run = |mid: [f64; 4]| -> f64 {
            let mut t = CascadeTrack::identity();
            t = cascade_step(&t, elems[0], Some(()));
            t = cascade_step(&t, mid, Some(()));
            t = cascade_step(&t, elems[2], Some(()));
            t.v[0]
        };
        for c in 0..4 {
            for h in [1e-6f64, -1e-6] {
                let mut mid = elems[1];
                mid[c] += h;
                let fd = (run(mid) - run(elems[1])) / h;
                let an = {
                    let mut t = CascadeTrack::identity();
                    t = cascade_step(&t, elems[0], Some(()));
                    t = cascade_step(&t, elems[1], Some(()));
                    t = cascade_step(&t, elems[2], None); // propagate to output
                    t.g[4 + c][0] // middle block params start at slot 4
                };
                // One-sided FD at h=1e-6 carries O(h) truncation (~1e-6);
                // agreement to 1e-5 still validates the Jacobians deeply.
                assert!(
                    (fd - an).abs() < 1e-5,
                    "c={c}: fd={fd:.9e} an={an:.9e}"
                );
            }
        }
    }

    #[test]
    fn multiblock_reduces_to_single_block_p() {
        // All-coherent flags: the cascade collapses to a single block with
        // W = (1,0,0,0), so p_function_multiblock must equal p_function.
        let (n, d, rv, rt) = stack_a();
        let nl = n.len();
        let wavls = [500.0f64, 550.0, 600.0];
        let angles = [0.0f64, 0.35];
        let mut cache = Vec::new();
        for &lam in &wavls {
            let disp = |l: f64, b: f64, s: f64| b + s * (550.0 / l - 1.0);
            let ls = [
                n_(1.0, 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(disp(lam, 1.45, 0.05), 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(1.52, 0.0),
            ];
            for m in ls {
                cache.push(m.re);
                cache.push(m.im);
            }
        }
        let needle_per_wav = [n_(1.9, 0.0); 3];
        let target = [0.10f64; 6];
        let weights = [1.0f64; 6];
        let z = [20.0f64, 60.0, 100.0];
        let flags = [0i32; 5];

        let p_ref = p_function(
            &wavls, &angles, 0, nl - 1, nl, &cache, &d, &rt, &rv,
            &needle_per_wav, &target, &weights, &z, 0,
        )
        .unwrap();
        let p_mb = p_function_multiblock(
            &wavls, &angles, &cache, &d, &flags, &rt, &rv,
            &needle_per_wav, &target, &weights, &z, None, 0,
        )
        .unwrap();
        for (a, b) in p_ref.iter().zip(&p_mb) {
            assert!((a - b).abs() < 1e-10 * a.abs().max(1e-12), "{a} vs {b}");
        }
    }

    /// Full Mode-A reference: R_tot of the whole incoherent stack via the
    /// same machinery the engine uses — block matrices from range-built
    /// fields, flux-normalized intensities, plain real-star cascade.
    fn solve_r_mode_a(
        n: &[Complex64], d: &[f64], rv: &[f64], rt: &[i32], flags: &[i32],
        lam: f64, sin_t: f64, pol: i32,
    ) -> f64 {
        let nsin = n[0] * cplx(sin_t, 0.0);
        let (blocks, spacers) = partition_blocks(flags);
        let mut ig = [0.0f64, 1.0, 1.0, 0.0];
        for (bi, &(bs, be)) in blocks.iter().enumerate() {
            let f = build_stack_fields_range(bs, be, n, d, rv, rt, lam, nsin, pol);
            ig = star_real(ig, block_intensities(&f, pol));
            if let Some(sp) = spacers[bi] {
                let tau = spacer_tau(n[sp], d[sp], lam, nsin);
                ig = star_real(ig, [0.0, tau, tau, 0.0]);
            }
        }
        ig[0]
    }

    #[test]
    fn fd_oracle_mode_a_two_blocks() {
        // air | T(40) | S(60, INCOH) | T(50) | S(30) | glass
        // Needles in either coherent block; merit targets are ZERO so
        // P(z) = 2·R_tot·dR_tot/dδ exactly (w=1). Each case runs a
        // single-(λ,θ) solve so P maps 1:1 onto the FD combination.
        let layers = [
            (n_(1.0, 0.0), 0.0),   // air
            (n_(2.35, 0.0), 40.0), // block 0 film
            (n_(1.45, 0.0), 60.0), // incoherent spacer (thick slab)
            (n_(2.35, 0.0), 50.0), // block 1 film
            (n_(1.45, 0.0), 30.0), // block 1 film
            (n_(1.52, 0.0), 0.0),  // glass
        ];
        let n: Vec<Complex64> = layers.iter().map(|&(m, _)| m).collect();
        let d: Vec<f64> = layers.iter().map(|&(_, t)| t).collect();
        let rv = vec![0.0; 6];
        let rt = vec![0; 6];
        let flags = [0, 0, 1, 0, 0, 0];

        let lam = 550.0f64;
        let cache = {
            let mut c = Vec::new();
            for m in &n {
                c.push(m.re);
                c.push(m.im);
            }
            c
        };
        let np = [n_(2.6, 0.0)];
        let target = [0.0f64];
        let weights = [1.0f64];

        // Depths from top of layer 1: L1 [0,40) T, L2 [40,100) SPACER,
        // L3 [100,150) T, L4 [150,180) S.
        // (z, host j, xi, sin_theta)
        for &(z, j, xi, sin_t) in &[
            (15.0f64, 1usize, 15.0f64, 0.0f64),  // block 0
            (120.0, 3, 20.0, 0.0),               // block 1, first film
            (15.0, 1, 15.0, 0.4),                // oblique
            (160.0, 4, 10.0, 0.4),               // block 1, second film
            (135.0, 3, 35.0, 0.25),              // deeper plane in block 1
        ] {
            let p = p_function_multiblock(
                &[lam], &[sin_t], &cache, &d, &flags, &rt, &rv,
                &np, &target, &weights, &[z], None, 0,
            )
            .unwrap()[0];

            let delta = 5e-4_f64;
            let r0 = solve_r_mode_a(&n, &d, &rv, &rt, &flags, lam, sin_t, 0);
            let (nn, dd, rr, tt) = insert_needle(&n, &d, &rv, &rt, j, xi, np[0], delta);
            // Flags shift identically with the arrays past the insertion point.
            let mut fl2 = flags.to_vec();
            let tail = flags[j + 1..].to_vec();
            fl2.truncate(j + 1);
            fl2.push(0);
            fl2.push(0);
            fl2.extend_from_slice(&tail);
            let r1 = solve_r_mode_a(&nn, &dd, &rr, &tt, &fl2, lam, sin_t, 0);
            let dr = (r1 - r0) / delta;
            // Half-gradient convention: P = R_tot · dR_tot/dδ (no factor 2).
            let expect = r0 * dr;

            let err = (p - expect).abs() / expect.abs().max(1e-12);
            assert!(
                err < 2e-3,
                "z={z} j={j} xi={xi} st={sin_t}: p={p:.6e} expect={expect:.6e} err={err:.2e}"
            );
        }
    }


    #[test]
    fn phase_sensitivity_matches_fd_of_phase() {
        // Order 0 end-to-end: Q(λ,z) vs central-free FD of the solved phase
        // with a physically inserted needle. Wrap-safe phase difference.
        let (n, d, rv, rt) = stack_a();
        let lam = 550.0f64;
        for &(sin_t, pol) in &[(0.0f64, 0i32), (0.35, 1)] {
            let nsin = n[0] * cplx(sin_t, 0.0);
            let f = build_stack_fields(&n, &d, &rv, &rt, lam, nsin, pol);
            let phi0 = f.s_left.last().unwrap().0.arg();
            // z depths from top of layer 1; layer tops: L1=0, L2=40, L3=120.
            for &(z, j, xi) in &[(15.0f64, 1usize, 15.0f64), (80.0, 2, 40.0), (130.0, 3, 10.0)] {
                let np = [n_(2.6, 0.0)];
                let out = phase_dispersion_sensitivity(
                    &[lam], &[sin_t], 0, nl_last_idx(&n), &cache_of(&n),
                    &d, &rt, &rv, &np, &[z], pol, 0, 0,
                )
                .unwrap();
                let q_an = out[0][0][0];

                let delta = 5e-4_f64;
                let (nn, dd, rr, tt) = insert_needle(&n, &d, &rv, &rt, j, xi, np[0], delta);
                let fmod = build_stack_fields(&nn, &dd, &rr, &tt, lam, nsin, pol);
                let phi1 = fmod.s_left.last().unwrap().0.arg();
                let mut dphi = phi1 - phi0;
                dphi = ((dphi + PI).rem_euclid(2.0 * PI)) - PI; // wrap-safe
                let q_fd = dphi / delta;

                let err = (q_an - q_fd).abs() / q_fd.abs().max(1e-12);
                assert!(
                    err < 2e-2,
                    "st={sin_t} pol={pol} z={z}: q_an={q_an:.4e} q_fd={q_fd:.4e} err={err:.2e}"
                );
            }
        }
    }

    fn nl_last_idx(n: &[Complex64]) -> usize {
        n.len() - 1
    }
    fn cache_of(n: &[Complex64]) -> Vec<f64> {
        let mut c = Vec::new();
        for m in n {
            c.push(m.re);
            c.push(m.im);
        }
        c
    }

    #[test]
    fn spectral_chain_exact_on_polynomials() {
        // Machinery check: the non-uniform 3-point stencil is exact for
        // polynomials of degree <= 2 at interior points, even on irregular
        // grids. Q(ω) = 1 + 2ω − 3ω² ⇒ dQ/dω = 2 − 6ω, d²Q/dω² = −6.
        let wavls: Vec<f64> = (0..25)
            .map(|i| 480.0 + 7.0 * i as f64 + 0.37 * ((i * 13) % 7) as f64)
            .collect();
        let omega: Vec<f64> = wavls.iter().map(|&l| 2.0 * PI * C_NM_PER_FS / l).collect();
        let num_wavs = wavls.len();
        let nz = 2usize;
        let q: Vec<Vec<f64>> = (0..num_wavs)
            .map(|w| {
                let om = omega[w];
                vec![1.0 + 2.0 * om - 3.0 * om * om; nz]
            })
            .collect();

        let d1 = spectral_gradient_step(&q, &omega, num_wavs, 1, nz);
        let d2 = spectral_gradient_step(&d1, &omega, num_wavs, 1, nz);

        // Exactness zone shrinks by one index per derivative order: the
        // first-order one-sided endpoint stencils contaminate their inward
        // neighbour on the next sweep (same behaviour as func_4's chain).
        for w in 1..num_wavs - 1 {
            let expect1 = 2.0 - 6.0 * omega[w];
            assert!((d1[w][0] - expect1).abs() < 1e-9, "w={w}: {} vs {expect1}", d1[w][0]);
        }
        for w in 2..num_wavs - 2 {
            assert!((d2[w][0] + 6.0).abs() < 1e-9, "w={w}: {}", d2[w][0]);
        }
    }

    #[test]
    fn gd_gdd_sensitivity_end_to_end_fd() {
        // Full physics chain: ∂GD/∂δ and ∂GDD/∂δ from this module vs
        // finite-differencing the SOLVED dispersion curves themselves
        // (identical unwrap+gradient operators applied to perturbed phases).
        let (n, d, rv, rt) = stack_a();
        let nl = n.len();
        let sin_t = 0.0f64;
        let pol = 0i32;

        // Dense grid for two clean derivative orders.
        let wavls: Vec<f64> = (0..20).map(|i| 505.0 + 5.0 * i as f64).collect();
        let num_wavs = wavls.len();
        // Cache layout: num_wavs * nl * 2 (this stack is dispersionless, but
        // the layout still repeats per wavelength).
        let mut cache = Vec::new();
        for _ in 0..num_wavs {
            for m in &n {
                cache.push(m.re);
                cache.push(m.im);
            }
        }
        let np = [n_(2.6, 0.0); 20];
        let z = [60.0f64]; // middle of film layer 2

        let out = phase_dispersion_sensitivity(
            &wavls, &[sin_t], 0, nl - 1, &cache, &d, &rt, &rv,
            &np, &z, pol, 0, 2,
        )
        .unwrap();

        // Solver-side GD/GDD without and with a physical needle.
        let delta = 5e-4_f64;
        let (nn, dd, rr, tt) = insert_needle(&n, &d, &rv, &rt, 2, 20.0, np[0], delta);
        let nsin = n[0] * cplx(sin_t, 0.0);
        let omega: Vec<f64> = wavls.iter().map(|&l| 2.0 * PI * C_NM_PER_FS / l).collect();

        let mut phi_base = Vec::with_capacity(num_wavs);
        let mut phi_mod = Vec::with_capacity(num_wavs);
        for &lam in wavls.iter() {
            let fb = build_stack_fields(&n, &d, &rv, &rt, lam, nsin, pol);
            let fm = build_stack_fields(&nn, &dd, &rr, &tt, lam, nsin, pol);
            phi_base.push(fb.s_left.last().unwrap().0.arg());
            phi_mod.push(fm.s_left.last().unwrap().0.arg());
        }
        // GD rows: ONE gradient sweep on unwrapped phase; GDD rows: two.
        let gd_base = grad_nonuniform(&unwrap_local(&phi_base), &omega);
        let gd_mod = grad_nonuniform(&unwrap_local(&phi_mod), &omega);
        let gdd_base = grad_nonuniform(&gd_base, &omega);
        let gdd_mod = grad_nonuniform(&gd_mod, &omega);

        for w in 2..num_wavs - 2 {
            // GD sensitivity
            let fd_gd = (gd_mod[w] - gd_base[w]) / delta;
            let an_gd = out[1][w][0];
            if fd_gd.abs() > 1e-6 {
                let err = (an_gd - fd_gd).abs() / fd_gd.abs();
                assert!(err < 5e-2, "GD w={w}: an={an_gd:.4e} fd={fd_gd:.4e} err={err:.2e}");
            }
        }
        // GDD rows.
        for wi in 2..num_wavs - 2 {
            let fd = (gdd_mod[wi] - gdd_base[wi]) / delta;
            let an = out[2][wi][0];
            let scale = fd.abs().max(an.abs());
            if scale > 1e-6 {
                let err = (an - fd).abs() / scale;
                assert!(err < 8e-2, "GDD w={wi}: an={an:.4e} fd={fd:.4e} err={err:.2e}");
            }
        }
    }

    fn unwrap_local(y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; y.len()];
        if y.is_empty() {
            return out;
        }
        out[0] = y[0];
        for i in 1..y.len() {
            let dd = ((y[i] - y[i - 1] + PI).rem_euclid(2.0 * PI)) - PI;
            out[i] = out[i - 1] + dd;
        }
        out
    }

    #[test]
    fn kernels_reproduce_grid_drivers() {
        // Per-point kernels must sum EXACTLY to the validated grid functions.
        let (n, d, rv, rt) = stack_a();
        let nl = n.len();
        let wavls = [500.0f64, 550.0, 600.0];
        let angles = [0.0f64, 0.35];
        let num_wavs = wavls.len();
        let total = num_wavs * angles.len();
        let mut cache = Vec::new();
        for &lam in &wavls {
            let disp = |l: f64, b: f64, sp: f64| b + sp * (550.0 / l - 1.0);
            let ls = [
                n_(1.0, 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(disp(lam, 1.45, 0.05), 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(1.52, 0.0),
            ];
            for m in ls {
                cache.push(m.re);
                cache.push(m.im);
            }
        }
        let npw = [n_(2.6, 0.0); 3];
        let target = vec![0.10f64; total];
        let weights = vec![1.5f64; total];
        let z = [15.0f64, 60.0, 130.0];

        // Coherent kernel vs p_function.
        let pref = p_function(
            &wavls, &angles, 0, nl - 1, nl, &cache, &d, &rt, &rv,
            &npw, &target, &weights, &z, 0,
        )
        .unwrap();
        let mut acc = vec![0.0; z.len()];
        for a in 0..angles.len() {
            for w in 0..num_wavs {
                let base = w * nl * 2;
                let ns: Vec<Complex64> =
                    (0..nl).map(|l| cplx(cache[base + l * 2], cache[base + l * 2 + 1])).collect();
                let c = p_coherent_from_fields(
                    &build_stack_fields_range(0, nl - 1, &ns, &d, &rv, &rt, wavls[w], ns[0] * cplx(angles[a], 0.0), 0),
                    ns[0] * cplx(angles[a], 0.0), wavls[w], 0, npw[w],
                    target[a * num_wavs + w], weights[a * num_wavs + w],
                    &d, 0, nl - 1, &z,
                );
                for (zi, cv) in c.iter().enumerate() {
                    acc[zi] += cv;
                }
            }
        }
        for (a, b) in pref.iter().zip(&acc) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }

        // Multiblock kernel vs p_function_multiblock.
        let flags = [0i32, 0, 1, 0, 0, 0];
        // 6-layer stack with spacer at index 2
        let layers6 = [
            (n_(1.0, 0.0), 0.0), (n_(2.35, 0.0), 40.0), (n_(1.45, 0.0), 60.0),
            (n_(2.35, 0.0), 50.0), (n_(1.45, 0.0), 30.0), (n_(1.52, 0.0), 0.0),
        ];
        let n6: Vec<Complex64> = layers6.iter().map(|&(m, _)| m).collect();
        let d6: Vec<f64> = layers6.iter().map(|&(_, t)| t).collect();
        let rv6 = vec![0.0; 6];
        let rt6 = vec![0; 6];
        let mut cache6 = Vec::new();
        for _ in 0..num_wavs {
            for m in &n6 {
                cache6.push(m.re);
                cache6.push(m.im);
            }
        }
        let npw6 = [n_(2.6, 0.0); 3];
        let t6 = vec![0.05f64; total];
        let w6 = vec![1.0f64; total];
        let z6 = [15.0f64, 120.0];

        let pmb = p_function_multiblock(
            &wavls, &angles, &cache6, &d6, &flags, &rt6, &rv6,
            &npw6, &t6, &w6, &z6, None, 0,
        )
        .unwrap();
        let locs = locate_hosts_multiblock(&d6, &flags, &z6, None).unwrap();
        let mut acc6 = vec![0.0; z6.len()];
        for a in 0..angles.len() {
            for w in 0..num_wavs {
                let base = w * 6 * 2;
                let ns: Vec<Complex64> =
                    (0..6).map(|l| cplx(cache6[base + l * 2], cache6[base + l * 2 + 1])).collect();
                let c = p_multiblock_point(
                    wavls[w], angles[a], &ns, &d6, &flags, &rv6, &rt6,
                    npw6[w], t6[a * num_wavs + w], w6[a * num_wavs + w], &locs, 0,
                );
                for (zi, cv) in c.iter().enumerate() {
                    acc6[zi] += cv;
                }
            }
        }
        for (a, b) in pmb.iter().zip(&acc6) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
    }

    #[test]
    fn anchor_partial_composition_matches_full_stack() {
        // r(U at bottom of last film layer ⊗ L) must reproduce the full-stack
        // amplitude — exercises both partial-matrix loops end to end.
        for &pol in &[0i32, 1] {
            for &sin_t in &[0.0f64, 0.5] {
                let (n, d, rv, rt) = stack_a();
                let nsin = n[0] * cplx(sin_t, 0.0);
                let f = build_stack_fields(&n, &d, &rv, &rt, LAM, nsin, pol);

                let last_film = f.ds.len() - 2;
                let pp = cexp_fast(cplx(0.0, 1.0) * f.beta_nm[last_film] * f.ds[last_film]);
                let u = star(f.s_left[last_film], prop(pp)); // == s_left[last]
                let l = f.s_right[last_film]; // nothing below but the substrate interface? no:
                // s_right[last_film] = iface(last_film, substrate) ⊗ identity
                let composed = star(u, l).0;
                let direct = f.s_left.last().unwrap().0;
                assert!(
                    (composed - direct).norm() < 1e-12,
                    "pol={pol} sin_t={sin_t}: {composed} vs {direct}"
                );
            }        }    }    #[test]
    fn energy_conservation_lossless_normal_incidence() {
        let (n, d, rv, rt) = stack_a();
        let nsin = n[0] * cplx(0.0, 0.0);
        let f = build_stack_fields(&n, &d, &rv, &rt, LAM, nsin, 0);
        let s = f.s_left.last().unwrap();
        let r_amp = s.0;
        let t_amp = s.2;
        let y_amb = admittance(true, f.n[0], f.cos[0]).re;
        let y_sub = admittance(true, f.n[f.n.len() - 1], f.cos[f.n.len() - 1]).re;
        let rr = r_amp.norm_sqr();
        let tt = t_amp.norm_sqr() * (y_sub / y_amb);
        assert!((rr + tt - 1.0).abs() < 1e-10, "R+T={}", rr + tt);
    }    #[test]
    fn thin_slab_linearization_matches_exact_slab() {
        // Bare slab in uniform host: exact star-product reflection vs the
        // derived slopes — pins the rho_hat / tau_hat algebra.
        let host = n_(2.35, 0.0);
        let n_prime = n_(1.45, 0.0);
        let sin_t = 0.4;
        let nsin = host * cplx(sin_t, 0.0);
        let is_s = true;

        let cos_h = cos_from_nsin(nsin, host);
        let (rho_hat, tau_hat) = needle_slopes(host, cos_h, n_prime, nsin, is_s, 2.0 * PI / LAM);

        let cos_p = cos_from_nsin(nsin, n_prime);
        let y_h = admittance(is_s, host, cos_h);
        let y_p = admittance(is_s, n_prime, cos_p);
        let bp = 2.0 * PI / LAM * n_prime * cos_p;
        let r12 = (y_h - y_p) / (y_h + y_p);
        let phi_d = cexp_fast(cplx(0.0, 1.0) * bp * 1e-6);

        let i1 = (r12, y_p * 2.0 / (y_h + y_p), y_h * 2.0 / (y_h + y_p), -r12);
        let i2 = (-r12, y_h * 2.0 / (y_h + y_p), y_p * 2.0 / (y_h + y_p), r12);
        let slab = star(star(i1, prop(phi_d)), i2);

        assert!(
            ((slab.0 / 1e-6) - rho_hat).norm() / rho_hat.norm() < 1e-4,
            "rho_hat mismatch: {} vs {}",
            slab.0 / 1e-6,
            rho_hat
        );
        assert!(
            (((slab.2 - cplx(1.0, 0.0)) / 1e-6) - tau_hat).norm() / tau_hat.norm() < 1e-4,
            "tau_hat mismatch"
        );
    }    #[test]
    fn fd_oracle_lossless_stack() {
        let st = stack_a();
        let np_hi = n_(2.6, 0.0);
        let np_lo = n_(1.6, 0.0);
        // middle of each film layer
        check_fd_case("j1 mid hi", &st, 1, 20.0, np_hi, 0.0, 0);
        check_fd_case("j2 mid lo", &st, 2, 40.0, np_lo, 0.0, 0);
        check_fd_case("j3 mid hi", &st, 3, 15.0, np_hi, 0.0, 0);
        // near-boundary placements
        check_fd_case("j2 top", &st, 2, 1e-7, np_hi, 0.0, 0);
        check_fd_case("j2 bottom", &st, 2, 80.0 - 1e-7, np_lo, 0.0, 0);
        // oblique incidence, both polarizations
        check_fd_case("oblique s", &st, 2, 33.0, np_hi, 0.5, 0);
        check_fd_case("oblique p", &st, 2, 47.0, np_lo, 0.5, 1);
    }    #[test]
    fn fd_oracle_absorbing_stack() {
        let st = stack_absorbing();
        check_fd_case("absorbing mid s", &st, 2, 25.0, n_(2.35, 0.0), 0.0, 0);
        check_fd_case("absorbing mid p", &st, 2, 25.0, n_(1.6, 0.0), 0.3, 1);
        check_fd_case("absorbing needle abs", &st, 1, 10.0, n_(1.9, 0.3), 0.0, 0);
    }    #[test]
    fn p_function_reduces_to_weighted_sum_of_sensitivities() {
        // Consistency: P at a single z equals the hand-computed weighted sum
        // over two spectral points.
        let (n, d, rv, rt) = stack_a();
        let wavls = [500.0, 600.0];
        let angles = [0.0f64];
        let nl = n.len();
        let mut cache = Vec::new();
        for &lam in &wavls {
            let disp = |l: f64, base: f64, spread: f64| base + spread * (550.0 / l - 1.0);
            let layers = [
                n_(1.0, 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(disp(lam, 1.45, 0.05), 0.0),
                n_(disp(lam, 2.35, 0.15), 0.0),
                n_(1.52, 0.0),
            ];
            for m in layers {
                cache.push(m.re);
                cache.push(m.im);
            }        }        let needle_per_wav = [n_(1.9, 0.0), n_(1.9, 0.0)];
        let target = [0.05f64, 0.02];
        let weights = [1.0, 2.0];
        let z = [60.0f64];

        let p = p_function(
            &wavls, &angles, 0, nl - 1, nl, &cache, &d, &rt, &rv,
            &needle_per_wav, &target, &weights, &z, 0,
        )
        .unwrap();

        // Hand sum:
        let mut expect = 0.0;
        for (wi, &lam) in wavls.iter().enumerate() {
            let mut ns = Vec::new();
            for l in 0..nl {
                ns.push(cplx(cache[wi * nl * 2 + l * 2], cache[wi * nl * 2 + l * 2 + 1]));
            }            let nsin = ns[0] * cplx(angles[0], 0.0);
            let f = build_stack_fields(&ns, &d, &rv, &rt, lam, nsin, 0);
            let r = f.s_left.last().unwrap().0;
            let dr = {
                let loc = locate_depth_in(&d, 0, nl - 1, z[0]);
                needle_dr_ddz(&f, nsin, loc.0, loc.1, needle_per_wav[wi], 0, lam)
            };
            expect += 2.0 * weights[wi] * (r.norm_sqr() - target[wi]) * (r.conj() * dr).re;
        }        assert!((p[0] - expect).abs() < 1e-12, "p={} expect={}", p[0], expect);
    }}
