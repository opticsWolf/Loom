# Exposure audit (R5) — every feature reachable from both sides

**Rule:** every feature-level `pub fn` in the `navette` crate is callable
from Rust AND exposed via PyO3. Internal kernels (sub-kernel math,
cross-module plumbing) stay Rust-only but are catalogued, not hidden.

**Enforcement:** `tools/check_exposure.py` (run in CI) fails on any pub
fn that is neither referenced in `navette-py` nor listed in its audited
`ALLOWLIST`. Status: **203 pub fns, 90 allowlisted internals, 0 gaps.**

## Feature surface (bound 1:1 or via a composite entry point)

- **Solver:** `Solver::{new, from_raw, solve, needle_gradient,
  index_column, landscape, local_minima, refine_mode, find_eigenmodes,
  field_profile}`, `solve_arrays`, view-mask fns, `energy_conservation`.
- **Synthesis:** `build_design` (+`DesignRequest`), `assemble_stack`,
  `run_design`, `compile_merit_spec` (+`TargetSet`), curve setters,
  `reference_rotation`/`rotate_rows`, `build_needle_targets`.
- **Providers:** `DictProvider` (+insert/refresh), `SpecProvider`
  (+upsert/invalidate), `WeaverProvider` (+strict/target/invalidate),
  `grids_equal`, `assert_provider_grid`.
- **Config:** envelope gate, all six section loaders, whole-doc and
  file loaders, authoring `validate()` inventory.
- **Model:** `Layer`/`Group`/`Structure`/`Architect` mutators,
  `check_schema_version`, providers snapshot path.

## Deliberately internal (allowlisted with entry points)

`needle_operator::*` internals (via `needle_gradient`), optics
fast-math (via solve paths), `solve_point*`/`resolve_plan` (via
`Solver::solve`), optimizer `char_func*` (via scan/refine), pipeline
stages `run_needle_pass/cycles`, `cleanup_*`, `inflate_*`,
`levenberg_marquardt` (via `run_design`), color batch helpers (via the
per-model bindings), materials unit/grid/kk helpers (via model
kernels), `next_table_name`/`shared_group` (via expansion paths).

## Won't-ports (Python-only by nature)

- `apply_to_all_layers(func)` (callable-crossing), cosmetic `__repr__`s.
- YAML text parsing (format handling; validates natively after).
- Contrast-key normalization, dict-order orchestration, result reshapes
  (presentation over native results).
- Duck-backend adapter in `WeaverMaterialProvider` (foreign objects).
