# rust-codegen-worker Task — Unit func_1

## Unit Info
- ID: func_1
- Name: redheffer_product_complex_field
- Kind: function (standalone, no dependencies)
- Jit Type: @njit(cache=True, inline='always')

## Source Code
```python
@njit(cache=True, inline='always')
def redheffer_product_complex_field(
    r_A_front, t_A_back, t_A_fwd, r_A_back,
    r_B_front, t_B_back, t_B_fwd, r_B_back
):
    '''Complex Redheffer Star Product for FIELD amplitudes.'''
    denom = 1.0 - r_A_back * r_B_front
    LOG_MIN = 1e-100
    if abs(denom) < LOG_MIN:
        phase = denom / (abs(denom) + 1e-300)
        denom = LOG_MIN * phase + 1e-300
    inv_denom = 1.0 / denom

    s_r_front = r_A_front + t_A_back * r_B_front * t_A_fwd * inv_denom
    s_t_back  = t_A_back * t_B_back * inv_denom
    s_t_fwd   = t_B_fwd * t_A_fwd * inv_denom
    s_r_back  = r_B_back + t_B_fwd * r_A_back * t_B_back * inv_denom

    return s_r_front, s_t_back, s_t_fwd, s_r_back
```

## Signature: (complex128 x 8 args) -> (complex×4 tuple)
- ALL inputs are complex128 — use num_complex::Complex64 for everything
- Use Complex64::sin(), .cos(), .exp() etc. for complex math
- Do NOT extract q.real

Write output to C:/Users/Frank/rust-rewrite/ (src/func_1.rs, benchmarks/test_redheffer_product_complex_field.py)
