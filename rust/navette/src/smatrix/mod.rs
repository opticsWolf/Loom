// Navette S-matrix engine — pure-Rust core (no Python, no I/O).
//
// Module map:
//   optics_core      — shared primitives (stars, roughness, complex kernels)
//   solver           — configured solver: validation + parallel solve + derives
//   coherent_block   — s/p/dual coherent-block solvers
//   core_engine      — request-driven unified engine (resolve_plan/solve_point)
//   needle_operator  — analytic needle-operator sensitivities (pure Rust)
//   needle_engine    — needle request bits + dispersion-order helper
//                      (the rayon/Python API lives in the navette-py crate)
//   optimizer        — landscape/minimize helpers (char_func et al.)
//   synthesis        — automated design synthesis (pure Rust core)

pub mod optics_core;
pub mod solver;
pub mod coherent_block;
pub mod core_engine;
pub mod needle_operator;
pub mod needle_engine;
pub mod optimizer;
pub mod synthesis;
