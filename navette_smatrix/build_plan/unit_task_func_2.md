# rust-codegen-worker Task — Unit func_2

## Unit Info
- ID: func_2
- Name: redheffer_product_real
- Kind: function (standalone, no dependencies)
- Jit Type: @njit(cache=True, inline='always')

## Source Code
```python
@njit(cache=True, inline='always')
def redheffer_product_real(ra_Rf, ra_Tb, ra_Tf, ra_Rb,
                           rb_Rf, rb_Tb, rb_Tf, rb_Rb):
    '''Real-valued Intensity Redheffer Star Product.'''
    denom = 1.0 - ra_Rb * rb_Rf
    DBL_EPSILON = 2.22e-16
    if abs(denom) < DBL_EPSILON:
        inv_denom = 0.0
    else:
        inv_denom = 1.0 / denom

    Rf  = ra_Rf + ra_Tb * rb_Rf * ra_Tf * inv_denom
    Tb  = ra_Tb * rb_Tb * inv_denom
    Tf  = rb_Tf * ra_Tf * inv_denom
    Rb  = rb_Rb + rb_Tf * ra_Rb * rb_Tb * inv_denom
    return Rf, Tb, Tf, Rb
```

## Signature: (float64 x 8 args) -> (f64×4 tuple)
- ALL inputs are float64 — use Rust f64 for everything
- No complex math needed

Write output to C:/Users/Frank/rust-rewrite/ (src/func_2.rs, benchmarks/test_redheffer_product_real.py)
