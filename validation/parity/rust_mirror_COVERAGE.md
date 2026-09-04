# Rust → Python test mirror: coverage map

Every `#[test]` in the workspace, mapped to its Python mirror or to the
reason it cannot be reached from Python. Mirror tests live next to the
existing parity scripts and follow the same runnable-script convention
(`python validation/parity/<area>/test_*.py`).

| mirror file | Rust source | coverage |
|---|---|---|
| `parity/color/test_color_mirror.py` | `navette-color`: `parity.rs` (15), `func_01`–`func_14` unit tests, `metrics.rs`, `common.rs`/`func_08.rs` checks | 70/70 + DE2000 bonus |
| `parity/color/golden_mirror.py` | generated from `navette-color/src/golden.rs` by `gen_golden_mirror.py` | — |
| `parity/materials/test_materials_mirror.py` | `navette-materials/tests/parity.rs` (22), `table.rs` (2), `evaluate()` dispatch pins | 24/24 + 17 dispatch |
| `parity/synthesis/test_merit_mirror.py` | `smatrix/src/synthesis/merit.rs` | 30/30 |
| `parity/synthesis/test_needlefold_mirror.py` | `needle_pass.rs` fold tests | 14/20 |
| `parity/smatrix/test_physics_mirror.py` | `evaluator.rs` physics, `optics_core.rs`, `needle_operator.rs` subset, interpolate smoke | 5 + 1 + 11 |

## Fully mirrored (1:1 — same inputs, expectations, tolerances)

- `merit.rs` — all 30, incl. `merit_two_keys_no_double_count`,
  `absorption_derived_from_companions`, PD `passes_scale`/`zero_d`,
  weight/count/integral, back `RBs`/`ABs`, all validation rejections.
  (`constraint_kind_from_str` → accept e/a/b/r/c + reject `"x"` at `add_target`.)
- `needle_pass.rs` fold half — all 14 `build_needle_targets` tests, incl.
  `phi_gain_shift_matches_fd`, `fold_applies_weight_and_count`, both
  integral folds, all 10 `targets_builder_*`.
- `navette-materials` — all 22 goldens (same `.npy`, same rtol/atol) + 2 table tests.
- `navette-color` — all 15 goldens (same vectors, same `1e-12 + 1e-9·|b|`
  tolerance) + all `func_0x` unit tests + broadcast semantics.
  Bonus: the `DE2000` golden is asserted here although NO Rust test uses it.
- `evaluator.rs` physics (5): lossless conservation, quarter-wave AR,
  optimizer recovery (bad start → QW, PD thickness; scipy-driven —
  the Rust LM itself is not exposed), propagation sign (`+π/2`),
  oblique PD reference. PD hand merit is in the merit mirror.
- `optics_core.rs` (1): hand values + `kz·D` identity via PD-merit zeros.
- `needle_operator.rs` exposed subset (11): coherent T/A/phase FD
  (pre-existing `test_needle_t_a_phi.py`), back TB/RB/AB FD (membrane,
  fb = 1), `P_MB`/`P_MB_T`/`P_MB_A` cascade FD, `P_MB*` single-block
  reduction (R + back TB/RB/AB), `DPHI` FD, `DGDD` end-to-end FD,
  lossless conservation.

## Gaps — Rust internals with no Python exposure

These can only be covered by exposing more API (deliberately not done here).
The physics each one guards is covered *end-to-end* by the mirrors above.

- Optimizer pipeline (55): `cleanup` (6), `config` (2), `inflate` (8),
  `pipeline` (7), `stagnation` (11), `structure` (11), `thick_opt` (10).
- `run_needle_pass` machinery (6): `scan_sites_*`, `profile_*`,
  `dual_pol_*`, `best_*`, `interp_clamped_edges`, `invalid_inputs_rejected`.
- `needle_operator` oracles (12): `subblock_range_*`,
  `fd_oracle_confined_to_subblock`, `p_function_block_confined_*`,
  `cascade_gradient_*`, `fd_oracle_mode_a_two_blocks`,
  `pmb_back_channels_match_cascade_finite_difference` (cascade-Tb oracle
  is internal; reduction half is mirrored as `pmb_back_reduces`),
  `spectral_chain_exact_on_polynomials`, `kernels_reproduce_grid_drivers`,
  `anchor_partial_composition_matches_full_stack`,
  `thin_slab_linearization_matches_exact_slab`, `fd_oracle_lossless_stack`,
  `fd_or
...[truncated 911 chars]