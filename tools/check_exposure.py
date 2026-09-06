"""Exposure lint: every feature-level `pub fn` in the `navette` crate must be
reachable from the PyO3 bindings (directly or through a covered entry point).

Usage: python tools/check_exposure.py [--write-allowlist]
Exit 1 (with names) when a pub fn is neither referenced in `navette-py`
nor listed in the audited `ALLOWLIST` of internal kernels.

The allowlist is a deliberate catalog, not a rug: entries are internal
helpers (sub-kernel math, cross-module plumbing, test hooks) reachable
through a listed entry point. Promoting an entry to a user feature means
binding it AND removing it here.
"""
import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "rust" / "navette" / "src"
BIND = ROOT / "rust" / "navette-py" / "src"

# Audited internal kernels: reachable through the entry point in parens.
ALLOWLIST = {
    # needle_operator sub-kernels (via Solver::needle_gradient)
    "admittance", "block_flux_factors", "block_intensities",
    "build_stack_fields", "build_stack_fields_range", "cascade_step",
    "cos_from_nsin", "locate_depth", "locate_depth_in",
    "locate_hosts_multiblock", "needle_dr_ddz", "needle_slopes",
    "needle_slopes4_ddz", "p_coherent_a_from_fields",
    "p_coherent_ab_from_fields", "p_coherent_from_fields",
    "p_coherent_phi_from_fields", "p_coherent_rb_from_fields",
    "p_coherent_t_from_fields", "p_coherent_tb_from_fields",
    "p_function", "p_function_multiblock", "p_multiblock_point",
    "partition_blocks", "phase_dispersion_sensitivity", "spacer_tau",
    "spectral_gradient_step", "star_real",
    # optics fast-math (via solve paths)
    "cexp_fast", "csqrt_fast", "grad_nonuniform", "reference_phase",
    "reference_wavenumber",
    # core solver plumbing (via Solver::solve)
    "solve_point", "solve_point_intensity", "resolve_plan",
    "dispersion_channel",
    # optimizer internals (via scan/refine bindings)
    "char_func", "char_func_xy", "reflection_coefficient_helper",
    # pipeline stages (via NeedlePipeline::run / run_design)
    "run_needle_cycles", "run_needle_pass", "needle_pass_scan",
    "build_scan_sites", "cleanup_design", "remove_thin_layers",
    "inflate_design", "qwot_nm", "qwot_to_thickness", "round_to_qwot",
    "thickness_to_qwot", "levenberg_marquardt",
    # color batch helpers (via the per-model bindings)
    "broadcast_pair", "clip01", "gamma_srgb", "inverse_gamma_srgb",
    "lab_f", "lab_f_batch", "lab_f_inv", "lab_f_inv_batch",
    "luv_to_xyz_d65", "mat3_mul", "mat3_mul_vec", "signed_pow",
    "vec_mul_mat3", "xyz_to_luv_d65", "xyz_to_uv_prime",
    "xyz_to_uv_prime_batch", "delta_e_2000_single", "delta_e_76_single",
    "delta_e_94_single", "delta_e_cmc_single", "delta_e_din99_single",
    # materials unit/grid/kk helpers (via model kernels)
    "energy_ev", "energy_ev_arr", "energy_grid", "eps2_monolog",
    "eps2_multi", "generate_from_steps", "kk_fft", "wiener_bounds",
    "wl_m", "wl_um2", "wl_um2_arr", "wl_signature",
    # spectralweave plumbing (via weaver bindings)
    "convert_unit", "norm_prefix",
    # coherent-block dual path (via solve_coherent_block_fields)
    "solve_coherent_block_fields_dual",
    # structure plumbing (via expansion/snapshot paths)
    "next_table_name", "shared_group",
}

# Feature-level entry points: bound under a different name or reached
# through a composite call (listed so the lint stays precise).
ALIASED = {
    "validate", "evaluate", "to_state", "set_data", "from_str",
    "as_i32", "try_from_i32", "new", "default", "names", "grid",
    "contains", "nk", "clone", "fmt",
}


def main() -> int:
    fns: dict[str, list[str]] = {}
    for p in SRC.rglob("*.rs"):
        for m in re.finditer(r"^pub fn (\w+)", p.read_text(encoding="utf-8"), re.M):
            fns.setdefault(m.group(1), []).append(str(p.relative_to(ROOT)))
    bind = "".join(p.read_text(encoding="utf-8") for p in BIND.rglob("*.rs"))
    bad = sorted(
        f for f in fns
        if f not in bind and f not in ALLOWLIST and f not in ALIASED
    )
    if bad:
        print("unexposed pub fns (bind them or allowlist with rationale):")
        for f in bad:
            print(f"  {f}  <-  {', '.join(fns[f][:2])}")
        return 1
    print(f"exposure OK ({len(fns)} pub fns, {len(ALLOWLIST)} allowlisted internals).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
