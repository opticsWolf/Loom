# loom_color

Rust rewrite (parity port) of the **Loom Unified Color Engine**
(`loom_colorengine.py`, "Gold Master", LGPL-3.0, opticsWolf).

## Module map (renumbered `func_01`–`func_15`)
| Module   | Conversion / metric                         |
|----------|---------------------------------------------|
| func_01  | XYZ ↔ xyY                                    |
| func_02  | Lab ↔ LCh                                    |
| func_03  | XYZ ↔ CIELUV                                 |
| func_04  | XYZ ↔ Oklab (direct-XYZ matrices)            |
| func_05  | sRGB ↔ Oklab (legacy sRGB-baked matrices)    |
| func_06  | CIE 1964 U*V*W*                              |
| func_07  | CIE 1960 UCS & chromaticity                  |
| func_08  | Bradford chromatic adaptation                |
| func_09  | Delta E 76                                   |
| func_10  | Delta E 94 (`De94Params`)                    |
| func_11  | Delta E CMC(l:c)                             |
| func_12  | DIN99                                        |
| func_13  | spectral pipeline (SPD → sRGB)               |
| func_14  | photometry engine                            |
| func_15  | shape handling & broadcasting (`map_pairs`)  |

The foundational sRGB↔XYZ and XYZ↔Lab conversions plus transfer functions live
in `common.rs`; generated matrices in `matrices.rs`.

## Build & test
```bash
cargo build
cargo test     # 55 unit + 15 parity + 15 doc tests, all pass
```
Parity vs the Python reference: worst-case deviation anywhere is **2.5e-14**
(func_03 CIELUV), otherwise mostly exactly 0 — machine precision.

### Optional `parallel` feature (rayon)
`rayon-core` 1.13 needs `rustc >= 1.80`. Built/tested on rustc/cargo **1.75**
(edition 2021) with the default sequential build. To use `--features parallel`
on 1.75, pin an older rayon (`cargo update -p rayon-core --precise <older>`),
otherwise use rustc >= 1.80.

## Reproducing the golden vectors
```bash
cd refgen
python gen_matrices.py   # -> ../src/matrices.rs
python gen_golden.py     # -> ../tests/golden.rs
```
`numba.py` is a no-op shim so the engine runs in pure NumPy. Drop a copy of
`loom_colorengine.py` into `refgen/` first.

## Parity convention notes
- Matrices stored natural; `mat3_mul_vec(M,v)[j] = Σ M[j][i]·v[i]`.
- `calc_transform_matrix` (Bradford) returns a **row-vector** matrix:
  `adapted = white @ M`, i.e. `out[j] = Σ_i white[i]·M[i][j]`.
- Oklab uses sign-preserving `signed_pow` (not `cbrt`); `sign(0)=0` per NumPy.
- `LAB_EPSILON = (6/29)³` (cubic/linear switch), `LAB_DELTA = 6/29`.
- Delta-E 94 / CMC weights use the reference sample `lab1`; `dH²` clamped ≥ 0.

## License
LGPL-3.0-or-later, matching the upstream engine.
