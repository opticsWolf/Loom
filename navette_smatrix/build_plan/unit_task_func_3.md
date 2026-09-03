# rust-codegen-worker Task — Unit func_3

## Unit Info
- ID: func_3
- Name: solve_coherent_block_fields
- Kind: function (calls w_function + redheffer_product_complex_field internally)
- Jit Type: @njit(cache=True, inline='always')

## Source Code
```python
@njit(cache=True, inline='always')
def solve_coherent_block_fields(
    start_idx, end_idx, n_stack, d_stack, rough_vals, rough_types, lam, NSinFi, pol
):
    """Returns: (sg_rf, sg_tb, sg_tf, sg_rb, R_front, T_back, T_fwd, R_back)"""
    # Initialize identity S-matrix: r=0+0j, t=1+0j
    sg_rf = 0.0 + 0j; sg_tb = 1.0 + 0j; sg_tf = 1.0 + 0j; sg_rb = 0.0 + 0j
    two_pi_lam = 2.0 * np.pi / lam

    # First layer: compute cos(theta) and admittance with branch-cut safety
    N_curr = n_stack[start_idx]
    val_curr = 1.0 - (NSinFi / N_curr)**2
    cos_curr = np.sqrt(val_curr)
    if cos_curr.imag < 0.0:
        cos_curr = -cos_curr

    if pol == POL_S:
        Y_curr = N_curr * cos_curr
    else:
        if abs(cos_curr) < 1e-12:
            cos_curr = complex(1e-12, 0.0)
        Y_curr = N_curr / cos_curr

    Y_first = Y_curr
    LOG_MIN = 1e-100

    for idx in range(start_idx, end_idx):
        N_next = n_stack[idx + 1]
        sigma = rough_vals[idx + 1]
        rtype = rough_types[idx + 1]

        val_next = 1.0 - (NSinFi / N_next)**2
        cos_next = np.sqrt(val_next)
        if cos_next.imag < 0.0:
            cos_next = -cos_next

        if pol == POL_S:
            Y_next = N_next * cos_next
        else:
            if abs(cos_next) < 1e-12:
                cos_next = complex(1e-12, 0.0)
            Y_next = N_next / cos_next

        den = Y_curr + Y_next
        if abs(den) < LOG_MIN:
            den = complex(LOG_MIN, LOG_MIN)
        inv_den = 1.0 / den
        r12 = (Y_curr - Y_next) * inv_den; r21 = -r12
        t12 = 2.0 * Y_curr * inv_den; t21 = 2.0 * Y_next * inv_den

        if rtype == 5:
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next
            nc_factor = np.exp(-2.0 * kz1 * kz2 * sigma * sigma)
            r12 *= nc_factor; r21 *= nc_factor; t12 *= nc_factor; t21 *= nc_factor
        elif rtype != 0:
            kz1 = two_pi_lam * N_curr * cos_curr
            kz2 = two_pi_lam * N_next * cos_next
            al = w_function(2.0 * kz1 * sigma, rtype)
            be = w_function(2.0 * kz2 * sigma, rtype)
            ga = w_function((kz1 - kz2) * sigma, rtype)
            r12 *= al; r21 *= be; t12 *= ga; t21 *= ga

        # Accumulate via Redheffer star product (calls redheffer_product_complex_field)
        sg_rf, sg_tb, sg_tf, sg_rb = redheffer_product_complex_field(
            sg_rf, sg_tb, sg_tf, sg_rb, r12, t21, t12, r21)

        if idx + 1 < end_idx:
            d = d_stack[idx + 1]
            if d > 1e-12:
                beta = two_pi_lam * d * N_next * cos_next
                if beta.imag < 0.0:
                    beta = complex(beta.real, -beta.imag)
                phi = np.exp(complex(0, 1) * beta)
                sg_rb *= (phi * phi); sg_tb *= phi; sg_tf *= phi

        N_curr = N_next; cos_curr = cos_next; Y_curr = Y_next

    # Convert to intensities
    R_front = abs(sg_rf)**2; R_back = abs(sg_rb)**2
    real_Y_first = Y_first.real; real_Y_last = Y_curr.real
    if real_Y_first < 1e-15: real_Y_first = 0.0
    if real_Y_last < 1e-15: real_Y_last = 0.0
    factor_fwd = (real_Y_last / real_Y_first) if real_Y_first > 1e-15 else 0.0
    factor_back = (real_Y_first / real_Y_last) if real_Y_last > 1e-15 else 0.0
    T_fwd = abs(sg_tf)**2 * factor_fwd; T_back = abs(sg_tb)**2 * factor_back

    return sg_rf, sg_tb, sg_tf, sg_rb, R_front, T_back, T_fwd, R_back
```

## Signature: (i32, i32, complex128[:], f64[:], f64[:], int32[:], f64, complex128, i32) -> (complex×4, f64×4) — 8-element tuple
- Calls `w_function` and `redheffer_product_complex_field` internally

Write output to C:/Users/Frank/rust-rewrite/ (src/func_3.rs, benchmarks/test_solve_coherent_block_fields.py)
