//! Navette -- Rust Rewrite of Numba-optimized thin-film optical solver
//!
//! synthesis::structure — DesignStack / LayerSpec.
//!
//! Port of the Loom_Structure / Layer subset used by needle synthesis
//! (loom_structure.py) plus the mutation primitives used by
//! needle_synthesis.py (`_insert_needle`, `merge_adjacent_layers`) and
//! needle_pipeline.py (`clamp_all_layers`).
//!
//! Stack model (matches Python exactly):
//!     layer_list = [ambient, film_0, …, film_{N-1}, substrate]
//! Ambient and substrate are fixed: never optimized, never removed,
//! never hosts. Only `films` is mutable.
//!
//! Solver-array layout (inherited from navette/smatrix.py):
//!   n_stack_cache — flat f64, wav-major with re/im interleaved per layer:
//!                   base = w * n_layers * 2 → [Re0, Im0, Re1, Im1, …]
//!   thicknesses   — [n_layers] nm
//!   incoherent_flags / rough_types / rough_vals — [n_layers]

use std::sync::Arc;

use num_complex::Complex64;

use crate::optics_core::cplx;

// ---------------------------------------------------------------------------
// LayerSpec
// ---------------------------------------------------------------------------

/// One layer of the design stack.
///
/// `nk` holds the complex refractive index evaluated on the *fixed*
/// simulation wavelength grid (one entry per wavelength), mirroring
/// `ArrayMaterialProvider` semantics from needle_synthesis.py where nk
/// arrays are pre-interpolated onto the grid.
#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub material: Arc<str>,
    /// Complex nk per simulation wavelength, len == num_wavs.
    pub nk: Arc<[Complex64]>,
    /// Physical thickness in nm.
    pub d_nm: f64,
    /// `true` → coherent propagation (incoherent_flag = 0).
    pub coherent: bool,
    /// Roughness model id passed through to the solver untouched.
    pub rough_type: i32,
    /// Roughness amplitude (Å) passed through to the solver untouched.
    pub rough_val: f64,
    /// Included in thickness optimization.
    pub optimize: bool,
    /// Admissible host for needle insertion.
    pub needle: bool,
}

impl LayerSpec {
    /// Constant-index helper for tests and synthetic designs.
    pub fn constant(
        material: &str,
        n_re: f64,
        n_im: f64,
        d_nm: f64,
        num_wavs: usize,
    ) -> Self {
        LayerSpec {
            material: Arc::from(material),
            nk: vec![cplx(n_re, n_im); num_wavs].into(),
            d_nm,
            coherent: true,
            rough_type: 0,
            rough_val: 0.0,
            optimize: true,
            needle: true,
        }
    }

    /// Verbatim port of Python `Layer.clone()` semantics: an independent
    /// copy (Arc clones share the immutable nk buffer — fine, it is never
    /// mutated after construction).
    pub fn cloned(&self) -> Self {
        self.clone()
    }
}

// ---------------------------------------------------------------------------
// DesignStack
// ---------------------------------------------------------------------------

/// Thin-film stack: fixed ambient + films + fixed substrate.
#[derive(Clone, Debug)]
pub struct DesignStack {
    ambient: LayerSpec,
    substrate: LayerSpec,
    films: Vec<LayerSpec>,
    num_wavs: usize,
}

impl DesignStack {
    /// Build a stack from boundary layers plus film layers.
    ///
    /// All layers must share the same nk length (the simulation grid size).
    pub fn with_films(
        ambient: LayerSpec,
        substrate: LayerSpec,
        films: Vec<LayerSpec>,
    ) -> Result<Self, String> {
        let num_wavs = ambient.nk.len();
        if substrate.nk.len() != num_wavs {
            return Err(format!(
                "substrate nk length {} != ambient nk length {}",
                substrate.nk.len(),
                num_wavs
            ));
        }
        for (i, f) in films.iter().enumerate() {
            if f.nk.len() != num_wavs {
                return Err(format!(
                    "film {} ('{}') nk length {} != ambient nk length {}",
                    i,
                    f.material,
                    f.nk.len(),
                    num_wavs
                ));
            }
        }
        Ok(DesignStack { ambient, substrate, films, num_wavs })
    }

    // -- properties ---------------------------------------------------------

    pub fn num_wavs(&self) -> usize {
        self.num_wavs
    }

    pub fn ambient(&self) -> &LayerSpec {
        &self.ambient
    }

    pub fn substrate(&self) -> &LayerSpec {
        &self.substrate
    }

    /// The film layers (excluding ambient and substrate).
    pub fn films(&self) -> &[LayerSpec] {
        &self.films
    }

    pub fn film(&self, idx: usize) -> Option<&LayerSpec> {
        self.films.get(idx)
    }

    /// Total number of solver-visible layers including boundaries.
    pub fn total_layer_count(&self) -> usize {
        self.films.len() + 2
    }

    /// Total physical thickness of the film stack (nm).
    pub fn total_thickness_nm(&self) -> f64 {
        self.films.iter().map(|f| f.d_nm).sum()
    }

    // -- mutation primitives -------------------------------------------------

    /// Split the film at `film_idx` and insert a seed layer.
    ///
    /// Verbatim port of Python `_insert_needle`: replaces films[film_idx]
    /// with `[top_portion, seed, bottom_portion]` where both portions are
    /// clones of the original host (keeping its flags) re-thickened to
    /// `depth_into_layer_nm` and `d_total − depth_into_layer_nm`.
    ///
    /// Note: like the Python code this permits degenerate zero-thickness
    /// portions when `depth_into_layer_nm` is 0 or == host thickness;
    /// cleanup removes them later.
    pub fn insert_needle_seed(
        &mut self,
        film_idx: usize,
        depth_into_layer_nm: f64,
        seed: LayerSpec,
    ) -> Result<(), String> {
        let original = self.films.get(film_idx).ok_or_else(|| {
            format!("film_idx {} out of range ({} films)", film_idx, self.films.len())
        })?;
        let d_total = original.d_nm;
        let d_bot = d_total - depth_into_layer_nm;

        let mut top = original.cloned();
        top.d_nm = depth_into_layer_nm;
        let mut bot = original.cloned();
        bot.d_nm = d_bot;

        // Replace in place without touching unrelated entries.
        self.films[film_idx] = top;
        self.films.insert(film_idx + 1, seed);
        self.films.insert(film_idx + 2, bot);
        Ok(())
    }

    /// Merge consecutive film layers of identical material.
    ///
    /// Verbatim port of `merge_adjacent_layers`: combined thickness is the
    /// sum; the *first* layer's properties (optimize/needle/roughness/…)
    /// are preserved. Returns the number of merges performed.
    pub fn merge_adjacent(&mut self) -> usize {
        if self.films.is_empty() {
            return 0;
        }
        let films = std::mem::take(&mut self.films);
        let mut merged: Vec<LayerSpec> = Vec::with_capacity(films.len());
        let mut merge_count = 0usize;

        let mut i = 0usize;
        while i < films.len() {
            let current = &films[i];
            let mut combined_d = current.d_nm;
            let mut j = i + 1;
            while j < films.len() && films[j].material == current.material {
                combined_d += films[j].d_nm;
                j += 1;
                merge_count += 1;
            }
            let mut result = current.cloned();
            result.d_nm = combined_d;
            merged.push(result);
            i = j;
        }

        self.films = merged;
        merge_count
    }

    /// Remove film at `film_idx`. Returns the removed layer.
    pub fn remove_film(&mut self, film_idx: usize) -> Result<LayerSpec, String> {
        if film_idx >= self.films.len() {
            return Err(format!(
                "film_idx {} out of range ({} films)",
                film_idx,
                self.films.len()
            ));
        }
        Ok(self.films.remove(film_idx))
    }

    /// Enforce [min_nm, max_nm] on every film layer.
    ///
    /// Verbatim port of `ClampedNeedleSynthesizer.clamp_all_layers`:
    /// layers below `min_nm` are *removed* (not clamped up — the optimizer
    /// tried to eliminate them); layers above `max_nm` are hard-capped.
    /// Returns `(n_removed, n_capped)`.
    pub fn clamp_all(&mut self, min_nm: f64, max_nm: f64) -> (usize, usize) {
        debug_assert!(min_nm >= 0.0 && max_nm > min_nm);
        let old = std::mem::take(&mut self.films);
        let mut surviving = Vec::with_capacity(old.len());
        let mut n_removed = 0usize;
        let mut n_capped = 0usize;

        for mut layer in old {
            if layer.d_nm < min_nm {
                n_removed += 1;
                continue;
            }
            if layer.d_nm > max_nm {
                layer.d_nm = max_nm;
                n_capped += 1;
            }
            surviving.push(layer);
        }
        self.films = surviving;
        (n_removed, n_capped)
    }

    /// Set the thickness of film `film_idx` (used by the thickness optimizer).
    pub fn set_thickness(&mut self, film_idx: usize, d_nm: f64) -> Result<(), String> {
        let f = self
            .films
            .get_mut(film_idx)
            .ok_or_else(|| format!("film_idx {} out of range", film_idx))?;
        f.d_nm = d_nm;
        Ok(())
    }

    // -- solver-array materialization ----------------------------------------

    /// Materialize flat solver arrays in the layouts consumed by
    /// `core_engine` / `needle_engine`.
    pub fn solver_arrays(&self) -> SolverArrays {
        let n_layers = self.total_layer_count();
        let nw = self.num_wavs;

        let mut n_stack_cache = vec![0.0f64; nw * n_layers * 2];
        let mut thicknesses = vec![0.0f64; n_layers];
        let mut incoherent_flags = vec![0i32; n_layers];
        let mut rough_types = vec![0i32; n_layers];
        let mut rough_vals = vec![0.0f64; n_layers];

        for (slot, layer) in std::iter::once(&self.ambient)
            .chain(self.films.iter())
            .chain(std::iter::once(&self.substrate))
            .enumerate()
        {
            thicknesses[slot] = layer.d_nm;
            incoherent_flags[slot] = i32::from(!layer.coherent);
            rough_types[slot] = layer.rough_type;
            rough_vals[slot] = layer.rough_val;
            for w in 0..nw {
                let base = w * n_layers * 2 + slot * 2;
                n_stack_cache[base] = layer.nk[w].re;
                n_stack_cache[base + 1] = layer.nk[w].im;
            }
        }

        SolverArrays {
            n_stack_cache,
            thicknesses,
            incoherent_flags,
            rough_types,
            rough_vals,
            n_layers: n_layers as i32,
        }
    }
}

// ---------------------------------------------------------------------------
// SolverArrays
// ---------------------------------------------------------------------------

/// Flat solver input arrays (mirrors loom_structure.SolverArrays).
#[derive(Clone, Debug)]
pub struct SolverArrays {
    /// Wav-major, re/im interleaved: base = w * n_layers * 2.
    pub n_stack_cache: Vec<f64>,
    /// Per-layer thickness (nm), including ambient + substrate.
    pub thicknesses: Vec<f64>,
    /// 1 = incoherent spacer, 0 = coherent.
    pub incoherent_flags: Vec<i32>,
    pub rough_types: Vec<i32>,
    pub rough_vals: Vec<f64>,
    pub n_layers: i32,
}

// ---------------------------------------------------------------------------
// Tests — mirror Python behaviors verbatim
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NW: usize = 8; // simulation wavelengths

    fn air(d: f64) -> LayerSpec {
        let mut l = LayerSpec::constant("air", 1.0, 0.0, d, NW);
        l.optimize = false;
        l.needle = false;
        l
    }

    fn sub(d: f64) -> LayerSpec {
        let mut l = LayerSpec::constant("sub", 1.52, 0.0, d, NW);
        l.optimize = false;
        l.needle = false;
        l
    }

    fn stack(films: Vec<LayerSpec>) -> DesignStack {
        DesignStack::with_films(air(0.0), sub(0.0), films).unwrap()
    }

    fn h(d: f64) -> LayerSpec {
        LayerSpec::constant("H", 2.35, 0.0, d, NW)
    }

    fn l(d: f64) -> LayerSpec {
        LayerSpec::constant("L", 1.46, 0.0, d, NW)
    }

    #[test]
    fn basic_properties() {
        let s = stack(vec![h(100.0), l(50.0)]);
        assert_eq!(s.film_count_public(), 2);
        assert_eq!(s.total_layer_count(), 4);
        assert!((s.total_thickness_nm() - 150.0).abs() < 1e-12);
        assert_eq!(s.num_wavs(), NW);
    }

    #[test]
    fn mismatched_nk_lengths_rejected() {
        let mut bad = h(10.0);
        bad.nk = vec![cplx(2.35, 0.0); NW - 1].into();
        assert!(DesignStack::with_films(air(0.0), sub(0.0), vec![bad]).is_err());
    }

    #[test]
    fn insert_needle_seed_splits_host() {
        // Python: [top=depth, seed, bot=d_total-depth]; host flags preserved.
        let mut s = stack(vec![h(100.0)]);
        let seed = LayerSpec::constant("L", 1.46, 0.0, 5.0, NW);
        s.insert_needle_seed(0, 60.0, seed).unwrap();

        let f = s.films();
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].material.as_ref(), "H");
        assert!((f[0].d_nm - 60.0).abs() < 1e-12);
        assert_eq!(f[1].material.as_ref(), "L");
        assert!((f[1].d_nm - 5.0).abs() < 1e-12);
        assert_eq!(f[2].material.as_ref(), "H");
        assert!((f[2].d_nm - 40.0).abs() < 1e-12);
        // total preserved
        assert!((s.total_thickness_nm() - 105.0).abs() < 1e-12);
        // host flags kept on fragments
        assert!(f[0].optimize && f[0].needle);
        assert!(f[2].optimize && f[2].needle);
    }

    #[test]
    fn insert_out_of_range_errors() {
        let mut s = stack(vec![h(100.0)]);
        assert!(s.insert_needle_seed(3, 5.0, l(1.0)).is_err());
    }

    #[test]
    fn merge_adjacent_sums_and_keeps_first_props() {
        // H(50) L(30) H(20) H(40) L(60) → merges: one pair of H's.
        // First H keeps its own props; give the first fragment different
        // flags to verify "keeps FIRST layer's properties".
        let mut first_h = h(20.0);
        first_h.optimize = false;
        let second_h = h(40.0);

        let mut s = stack(vec![h(50.0), l(30.0), first_h, second_h, l(60.0)]);
        let n = s.merge_adjacent();
        assert_eq!(n, 1);

        let f = s.films();
        assert_eq!(f.len(), 4);
        assert_eq!(f[2].material.as_ref(), "H");
        assert!((f[2].d_nm - 60.0).abs() < 1e-12);
        // first-of-pair properties win
        assert!(!f[2].optimize);
        assert!(f[2].needle);
        assert!((s.total_thickness_nm() - 200.0).abs() < 1e-12);
    }

    #[test]
    fn merge_full_run_collapses() {
        let mut s = stack(vec![h(1.0), h(2.0), h(3.0)]);
        assert_eq!(s.merge_adjacent(), 2);
        assert_eq!(s.film_count_public(), 1);
        assert!((s.films()[0].d_nm - 6.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_removes_thin_caps_thick() {
        let mut s = stack(vec![h(1.0), l(5000.0), h(50.0), l(2.0 - 1e-9)]);
        let (removed, capped) = s.clamp_all(2.0, 1000.0);
        assert_eq!((removed, capped), (2, 1));
        let f = s.films();
        assert_eq!(f.len(), 2);
        assert!((f[0].d_nm - 1000.0).abs() < 1e-12);
        assert!((f[1].d_nm - 50.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_boundary_values_survive() {
        // Exactly-at-boundary layers survive untouched (strict < and >).
        let mut s = stack(vec![h(2.0), l(1000.0)]);
        let (removed, capped) = s.clamp_all(2.0, 1000.0);
        assert_eq!((removed, capped), (0, 0));
        assert_eq!(s.film_count_public(), 2);
    }

    #[test]
    fn remove_film_returns_layer() {
        let mut s = stack(vec![h(10.0), l(20.0)]);
        let removed = s.remove_film(0).unwrap();
        assert_eq!(removed.material.as_ref(), "H");
        assert_eq!(s.film_count_public(), 1);
        assert!(s.remove_film(5).is_err());
    }

    #[test]
    fn solver_arrays_layout_matches_smatrix_convention() {
        // wav-major re/im interleave: base = w * n_layers * 2 + slot * 2
        let s = stack(vec![h(10.0)]);
        let sa = s.solver_arrays();
        assert_eq!(sa.n_layers, 3);
        assert_eq!(sa.n_stack_cache.len(), NW * 3 * 2);
        assert_eq!(sa.thicknesses.len(), 3);

        // ambient slot 0, film slot 1, substrate slot 2
        assert!((sa.thicknesses[0]).abs() < 1e-15);
        assert!((sa.thicknesses[1] - 10.0).abs() < 1e-12);
        assert!((sa.thicknesses[2]).abs() < 1e-15);

        for w in 0..NW {
            let base = w * 3 * 2;
            assert!((sa.n_stack_cache[base] - 1.0).abs() < 1e-12); // air Re
            assert!(sa.n_stack_cache[base + 1].abs() < 1e-12); // air Im
            assert!((sa.n_stack_cache[base + 2] - 2.35).abs() < 1e-12); // H Re
            assert!((sa.n_stack_cache[base + 4] - 1.52).abs() < 1e-12); // sub Re
        }
        assert!(sa.incoherent_flags.iter().all(|&v| v == 0));
    }

    #[test]
    fn incoherent_flag_and_roughness_passthrough() {
        let mut sp = LayerSpec::constant("spacer", 1.45, 0.0, 1_000_000.0, NW);
        sp.coherent = false;
        sp.rough_type = 2;
        sp.rough_val = 7.5;
        let s = stack(vec![sp]);
        let sa = s.solver_arrays();
        assert_eq!(sa.incoherent_flags, vec![0, 1, 0]);
        assert_eq!(sa.rough_types[1], 2);
        assert!((sa.rough_vals[1] - 7.5).abs() < 1e-12);
    }

    // small helper so tests read like the Python property name
    impl DesignStack {
        fn film_count_public(&self) -> usize {
            self.films().len()
        }
    }
}
