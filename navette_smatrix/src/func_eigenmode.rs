// src/eigenmode_solver.rs
use num_complex::{Complex64, ComplexFloat};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

// Reuse existing low‑level functions from the crate
use crate::func_0::w_function_inner;
use crate::func_1::redheffer_product_complex_field_inner;
use crate::func_3::solve_coherent_block_fields_inner;

// -----------------------------------------------------------------------------
// Helper: compute the complex reflection coefficient for a given n_eff
// -----------------------------------------------------------------------------
#[inline]
fn reflection_coefficient_helper(
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    n_eff: Complex64,
    pol: i32,
) -> Complex64 {
    let inv_n: Vec<Complex64> = n_stack.iter().map(|n| n.recip()).collect();
    let (r_front, _, _, _, _, _, _, _) = solve_coherent_block_fields_inner(
        0,
        n_stack.len() - 1,
        n_stack,
        &inv_n,
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
fn char_func(
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    n_eff: Complex64,
    pol: i32,
) -> f64 {
    let r = reflection_coefficient_helper(n_stack, thicknesses, rough_types, rough_vals, lam, n_eff, pol);
    let abs_r = r.norm();
    if abs_r < 1e-15 {
        1e30
    } else {
        (1.0 / abs_r).powi(2)
    }
}

/// Real‑valued wrapper for minimisation (expects [Re, Im] slice)
fn char_func_xy(
    xy: &[f64],
    n_stack: &[Complex64],
    thicknesses: &[f64],
    rough_types: &[i32],
    rough_vals: &[f64],
    lam: f64,
    pol: i32,
) -> f64 {
    let n_eff = Complex64::new(xy[0], xy[1]);
    char_func(n_stack, thicknesses, rough_types, rough_vals, lam, n_eff, pol)
}

// -----------------------------------------------------------------------------
// Coarse landscape scan (parallel, GIL‑free)
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "scan_landscape")]
pub fn scan_landscape(
    py: Python<'_>,
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    pol: i32,
    real_min: f64,
    real_max: f64,
    imag_min: f64,
    imag_max: f64,
    points_real: usize,
    points_imag: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Py<PyArray2<f64>>)> {
    let n_slice = n_stack.as_slice()?;
    let d_slice = thicknesses.as_slice()?;
    let rt_slice = rough_types.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;

    let real_vals: Vec<f64> = (0..points_real)
        .map(|i| real_min + (i as f64) * (real_max - real_min) / ((points_real - 1) as f64))
        .collect();
    let imag_vals: Vec<f64> = (0..points_imag)
        .map(|i| imag_min + (i as f64) * (imag_max - imag_min) / ((points_imag - 1) as f64))
        .collect();

    let landscape: Vec<f64> = py.detach(|| {
        (0..points_imag * points_real)
            .into_par_iter()
            .map(|idx| {
                let i = idx / points_real;
                let j = idx % points_real;
                let nr = real_vals[j];
                let ni = imag_vals[i];
                let n_eff = Complex64::new(nr, ni);
                char_func(n_slice, d_slice, rt_slice, rv_slice, lam, n_eff, pol)
            })
            .collect()
    });

    let land_arr = PyArray1::from_vec(py, landscape).reshape([points_imag, points_real]).unwrap();
    Ok((real_vals, imag_vals, land_arr.into()))
}

// -----------------------------------------------------------------------------
// Find local minima on the coarse grid
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "find_local_minima")]
pub fn find_local_minima(
    landscape: PyReadonlyArray2<f64>,
    real_vals: Vec<f64>,
    imag_vals: Vec<f64>,
    median_factor: f64,
) -> Vec<(f64, f64)> {
    let land = landscape.as_array();
    let (n_imag, n_real) = (land.shape()[0], land.shape()[1]);
    let median = land.iter().copied().fold(0.0, |a, b| a + b) / (n_imag * n_real) as f64;
    let threshold = median * median_factor;

    let mut candidates = Vec::new();
    for i in 0..n_imag {
        for j in 0..n_real {
            let val = land[[i, j]];
            if val >= threshold {
                continue;
            }
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(n_imag - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(n_real - 1);
            let mut is_min = true;
            'neighbors: for ii in i0..=i1 {
                for jj in j0..=j1 {
                    if ii == i && jj == j {
                        continue;
                    }
                    if land[[ii, jj]] <= val {
                        is_min = false;
                        break 'neighbors;
                    }
                }
            }
            if is_min {
                candidates.push((real_vals[j], imag_vals[i]));
            }
        }
    }
    candidates
}

// -----------------------------------------------------------------------------
// Nelder‑Mead minimiser (2D, adaptive)
// -----------------------------------------------------------------------------
#[pyfunction]
#[pyo3(name = "nelder_mead")]
pub fn nelder_mead(
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    pol: i32,
    x0: (f64, f64),
    step: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, f64) {
    let n_slice = n_stack.as_slice().unwrap();
    let d_slice = thicknesses.as_slice().unwrap();
    let rt_slice = rough_types.as_slice().unwrap();
    let rv_slice = rough_vals.as_slice().unwrap();

    let mut simplex = vec![
        [x0.0, x0.1],
        [x0.0 + step, x0.1],
        [x0.0, x0.1 + step * 0.1],
    ];
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|x| char_func_xy(x, n_slice, d_slice, rt_slice, rv_slice, lam, pol))
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut iter = 0;

    loop {
        let mut indices: Vec<usize> = (0..3).collect();
        indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
        let (best, good, worst) = (indices[0], indices[1], indices[2]);

        let centroid = [
            (simplex[best][0] + simplex[good][0]) / 2.0,
            (simplex[best][1] + simplex[good][1]) / 2.0,
        ];
        let reflected = [
            centroid[0] + alpha * (centroid[0] - simplex[worst][0]),
            centroid[1] + alpha * (centroid[1] - simplex[worst][1]),
        ];
        let f_ref = char_func_xy(&reflected, n_slice, d_slice, rt_slice, rv_slice, lam, pol);

        if f_ref < values[best] {
            let expanded = [
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            ];
            let f_exp = char_func_xy(&expanded, n_slice, d_slice, rt_slice, rv_slice, lam, pol);
            if f_exp < f_ref {
                simplex[worst] = expanded;
                values[worst] = f_exp;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_ref;
            }
        } else if f_ref < values[good] {
            simplex[worst] = reflected;
            values[worst] = f_ref;
        } else {
            let contracted = [
                centroid[0] + rho * (simplex[worst][0] - centroid[0]),
                centroid[1] + rho * (simplex[worst][1] - centroid[1]),
            ];
            let f_con = char_func_xy(&contracted, n_slice, d_slice, rt_slice, rv_slice, lam, pol);
            if f_con < values[worst] {
                simplex[worst] = contracted;
                values[worst] = f_con;
            } else {
                for i in 0..3 {
                    if i != best {
                        simplex[i][0] = simplex[best][0] + sigma * (simplex[i][0] - simplex[best][0]);
                        simplex[i][1] = simplex[best][1] + sigma * (simplex[i][1] - simplex[best][1]);
                        values[i] = char_func_xy(&simplex[i], n_slice, d_slice, rt_slice, rv_slice, lam, pol);
                    }
                }
            }
        }

        iter += 1;
        let size = ((simplex[0][0] - simplex[1][0]).powi(2) + (simplex[0][1] - simplex[1][1]).powi(2)).sqrt()
                + ((simplex[1][0] - simplex[2][0]).powi(2) + (simplex[1][1] - simplex[2][1]).powi(2)).sqrt()
                + ((simplex[2][0] - simplex[0][0]).powi(2) + (simplex[2][1] - simplex[0][1]).powi(2)).sqrt();
        if size < tol || iter >= max_iter {
            break;
        }
    }

    let best_idx = (0..3).min_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap()).unwrap();
    (simplex[best_idx][0], simplex[best_idx][1], values[best_idx])
}

// -----------------------------------------------------------------------------
// Field profile: |E(z)| through the stack for a given eigenmode
// -----------------------------------------------------------------------------
/// Data needed to compute fields inside a layer
struct LayerData {
    n: Complex64,
    cos: Complex64,
    thickness: f64,
    // inv_n no longer needed (was unused)
}

/// Compute the field profile inside the entire stack.
/// Returns (z_positions, |E|_values, layer_bounds_start, layer_bounds_end, layer_indices).
#[pyfunction]
#[pyo3(name = "field_profile")]
pub fn field_profile(
    n_stack: PyReadonlyArray1<Complex64>,
    thicknesses: PyReadonlyArray1<f64>,
    rough_types: PyReadonlyArray1<i32>,
    rough_vals: PyReadonlyArray1<f64>,
    lam: f64,
    n_eff: Complex64,
    pol: i32,
    points_per_layer: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Complex64>)> {
    let n_slice = n_stack.as_slice()?;
    let d_slice = thicknesses.as_slice()?;
    let rt_slice = rough_types.as_slice()?;
    let rv_slice = rough_vals.as_slice()?;

    let n_layers = n_slice.len();
    if n_layers < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("At least two layers required."));
    }

    let two_pi_lam = 2.0 * PI / lam;

    // Precompute layer data: n, cosθ, thickness
    let mut layers: Vec<LayerData> = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let n = n_slice[i];
        let r0 = n_eff * n.recip();
        let v = Complex64::new(1.0, 0.0) - r0 * r0;
        let mut cos = v.sqrt();
        if cos.im < 0.0 {
            cos = -cos;
        }
        layers.push(LayerData {
            n,
            cos,
            thickness: d_slice[i],
        });
    }

    // Helper for Fresnel + roughness at an interface (i -> i+1)
    let interface_props = |i: usize| -> (Complex64, Complex64, Complex64, Complex64) {
        let n_curr = layers[i].n;
        let cos_curr = layers[i].cos;
        let y_curr = if pol == 0 {
            n_curr * cos_curr
        } else {
            let c = if cos_curr.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_curr };
            n_curr / c
        };
        let n_next = layers[i+1].n;
        let cos_next = layers[i+1].cos;
        let y_next = if pol == 0 {
            n_next * cos_next
        } else {
            let c = if cos_next.norm() < 1e-12 { Complex64::new(1e-12, 0.0) } else { cos_next };
            n_next / c
        };

        let den = y_curr + y_next;
        let den_safe = if den.norm() < 1e-100 { Complex64::new(1e-100, 1e-100) } else { den };
        let inv_den = den_safe.recip();
        let r12 = (y_curr - y_next) * inv_den;
        let t12 = y_curr * 2.0 * inv_den;
        let t21 = y_next * 2.0 * inv_den;
        let r21 = -r12;

        let sigma = rv_slice[i+1];
        let rtype = rt_slice[i+1];
        if rtype != 0 && sigma > 0.0 {
            let kz1 = two_pi_lam * n_curr * cos_curr;
            let kz2 = two_pi_lam * n_next * cos_next;
            if rtype == 5 {
                let f = (-2.0 * kz1 * kz2 * sigma * sigma).exp();
                (r12 * f, r21 * f, t12 * f, t21 * f)
            } else {
                let al = w_function_inner(2.0 * kz1 * sigma, rtype);
                let be = w_function_inner(2.0 * kz2 * sigma, rtype);
                let ga = w_function_inner((kz1 - kz2) * sigma, rtype);
                (r12 * al, r21 * be, t12 * ga, t21 * ga)
            }
        } else {
            (r12, r21, t12, t21)
        }
    };

    // Propagation phase through a layer (i)
    let prop_phase = |i: usize| -> Complex64 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            return Complex64::new(1.0, 0.0);
        }
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }
        (Complex64::new(0.0, 1.0) * beta).exp()
    };

    // ---------- Build left and right S‑matrices ----------
    // S_left[i] = S‑matrix from ambient up to the left side of layer i (i from 1 to n_layers-1)
    let mut s_left: Vec<(Complex64, Complex64, Complex64, Complex64)> = Vec::with_capacity(n_layers);
    s_left.push((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))); // identity before ambient

    for i in 0..n_layers-1 {
        let mut sg = s_left.last().unwrap().clone();
        if i > 0 && layers[i].thickness > 1e-12 {
            let phi = prop_phase(i);
            sg = redheffer_product_complex_field_inner(
                sg.0, sg.1, sg.2, sg.3,
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(sg.0, sg.1, sg.2, sg.3, iface.0, iface.1, iface.2, iface.3);
        s_left.push(sg);
    }

    // S_right[i] = S‑matrix from substrate up to the right side of layer i (i from n_layers-2 down to 0)
    let mut s_right: Vec<Option<(Complex64, Complex64, Complex64, Complex64)>> = vec![None; n_layers];
    s_right[n_layers-1] = Some((Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)));

    for i in (0..n_layers-1).rev() {
        let mut sg = s_right[i+1].unwrap();
        if i+1 < n_layers-1 && layers[i+1].thickness > 1e-12 {
            let phi = prop_phase(i+1);
            sg = redheffer_product_complex_field_inner(
                Complex64::new(0.0, 0.0), phi, phi, Complex64::new(0.0, 0.0),
                sg.0, sg.1, sg.2, sg.3,
            );
        }
        let iface = interface_props(i);
        sg = redheffer_product_complex_field_inner(iface.0, iface.1, iface.2, iface.3, sg.0, sg.1, sg.2, sg.3);
        s_right[i] = Some(sg);
    }

    // ---------- Compute field inside each layer ----------
    let mut z_pos = Vec::new();
    let mut e_mag = Vec::new();
    let mut layer_start = Vec::new();
    let mut layer_end = Vec::new();
    let mut layer_n = Vec::new();

    let mut z_cursor = 0.0;

    for i in 1..n_layers-1 {
        let d = layers[i].thickness;
        if d <= 1e-12 {
            continue;
        }
        let sl = &s_left[i];
        let sr = s_right[i].as_ref().unwrap();
        let denom = Complex64::new(1.0, 0.0) - sl.3 * sr.0;
        let denom_safe = if denom.norm() < 1e-100 {
            Complex64::new(1e-100, 1e-100)
        } else {
            denom
        };
        let inv_denom = denom_safe.recip();
        let e_plus = sl.2 * inv_denom;
        let e_minus = sr.0 * e_plus;
        let mut beta = two_pi_lam * d * layers[i].n * layers[i].cos;
        if beta.im < 0.0 {
            beta = Complex64::new(beta.re, -beta.im);
        }

        let step = d / (points_per_layer as f64);
        for k in 0..=points_per_layer {
            let zz = k as f64 * step;
            let xi = zz / d;
            let e_z = e_plus * (Complex64::new(0.0, 1.0) * beta * xi).exp()
                    + e_minus * (-Complex64::new(0.0, 1.0) * beta * xi).exp();
            z_pos.push(z_cursor + zz);
            e_mag.push(e_z.norm());
        }
        layer_start.push(z_cursor);
        layer_end.push(z_cursor + d);
        layer_n.push(layers[i].n);
        z_cursor += d;
    }

    // Normalise E‑field to max = 1
    let max_e = e_mag.iter().copied().fold(0.0, f64::max);
    if max_e > 0.0 {
        for val in &mut e_mag {
            *val /= max_e;
        }
    }

    Ok((z_pos, e_mag, layer_start, layer_end, layer_n))
}