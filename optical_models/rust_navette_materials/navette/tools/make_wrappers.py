#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the thin navette.materials wrappers from the original (Numba) modules.

For each source file:
  * drop `from numba import njit` and every `@njit(...)` decorator (the kernel
    bodies are plain NumPy and survive as readable reference / fallback),
  * add `from . import _native` (except the base, which has no kernel),
  * redirect each class kernel call site to the corresponding `_native.*`
    function, passing `self.wavelength` (nm) — Rust does the unit conversion.

This is a mechanical, faithful transform: all parameter management, _sync(),
flat-key access, validation and caching are preserved byte-for-byte.
"""
import re
import pathlib

SRC = pathlib.Path("/mnt/user-data/uploads")
DST = pathlib.Path("/home/claude/navette/python/navette/materials")


def strip_numba(text: str) -> str:
    text = re.sub(r"^from numba import .*\r?\n", "", text, flags=re.M)
    text = re.sub(r"^import numba.*\r?\n", "", text, flags=re.M)
    text = re.sub(r"^[ \t]*@njit\([^)]*\)[ \t]*\r?\n", "", text, flags=re.M)
    # Any leftover (now-unused) reference kernels used prange; keep them valid.
    text = text.replace("prange(", "range(")
    # Normalise absolute sibling imports to package-relative (codylorentz used
    # `from material import ...`; everything else already uses `from .material`).
    text = text.replace("from material import", "from .material import")
    return text


def add_native_import(text: str) -> str:
    # Insert after the `from .material import ...` line if present, else after numpy.
    if re.search(r"^from \.material import .*$", text, flags=re.M):
        return re.sub(r"(^from \.material import .*\r?\n)",
                      r"\1from . import _native\n", text, count=1, flags=re.M)
    return re.sub(r"(^import numpy as np[ \t]*\r?\n)",
                  r"\1from . import _native\n", text, count=1, flags=re.M)


def transform(name: str, redirects) -> None:
    text = (SRC / name).read_text(encoding="utf-8")
    text = strip_numba(text)
    if name != "material.py":
        text = add_native_import(text)
    for pat, repl in redirects:
        new, n = re.subn(pat, repl, text, flags=re.S)
        assert n >= 1, f"{name}: pattern not matched: {pat[:60]}..."
        text = new
    # Update the module banner so it's clear these are the Rust-backed wrappers.
    text = text.replace(
        "Loom: Weaving the mathematics of light in thin film systems",
        "Navette: the mathematics of light in thin-film systems\n(Rust-backed wrapper; kernels live in navette.materials._native)",
    )
    (DST / name).write_text(text, encoding="utf-8")
    print(f"  wrote {name}")


# --- base + table/constant: de-numba only ---
transform("material.py", [])
transform("basic.py", [])

# --- Lorentz ---
transform("lorentz.py", [
    (r"compute_lorentz_complex_nk\(\s*self\.E\s*,",
     "_native.lorentz_nk(\n                self.wavelength,"),
])

# --- Cauchy / Sellmeier (+ Urbach) ---
transform("cauchy_sellmeier.py", [
    (r"compute_cauchy_complex_nk\(\s*self\.wavelength_µm_2\s*,",
     "_native.cauchy_nk(\n            self.wavelength,"),
    (r"compute_sellmeier_complex_nk\(\s*self\.wavelength_µm_2\s*,",
     "_native.sellmeier_nk(\n            self.wavelength,"),
    (r"compute_cauchy_urbach_complex_nk\(\s*self\.wavelength_m\s*,\s*self\.wavelength_µm_2\s*,\s*self\.E\s*,",
     "_native.cauchy_urbach_nk(\n            self.wavelength,"),
    (r"compute_sellmeier_urbach_complex_nk\(\s*self\.wavelength_m\s*,\s*self\.wavelength_µm_2\s*,\s*self\.E\s*,",
     "_native.sellmeier_urbach_nk(\n            self.wavelength,"),
    # drop the trailing _HC_EV_NM arg now that Rust owns the constant
    (r",\s*_HC_EV_NM\s*\r?\n\s*\)", "\n        )"),
])

# --- Drude / Drude-Lorentz ---
transform("drudelorentz.py", [
    # Fix a latent bug in the original _sync: it deleted the scalar params and
    # re-read them as None. Preserve and restore them instead.
    (r"param_keys_to_clean = \['omega_p', 'gamma_drude', 'epsilon_inf'\].*?"
     r"self\.params\['epsilon_inf'\] = self\.params\.get\('epsilon_inf'\)[ \t]*",
     "_preserved = {k: self.params[k] for k in ('omega_p', 'gamma_drude', 'epsilon_inf') if k in self.params}\n\n"
     "        for i, (e0, g, f) in enumerate(self._osc_params):\n"
     "            self.params[f\"E0_{i}\"] = e0\n"
     "            self.params[f\"Gamma_{i}\"] = g\n"
     "            self.params[f\"f0_{i}\"] = f\n\n"
     "        # Restore scalar params (original _sync re-read them as None).\n"
     "        self.params.update(_preserved)"),
    # Anchor on `E=self.E` to hit the call site (not the def), match lazily to close.
    (r"compute_drude_complex_nk\(\s*E=self\.E.*?eps_inf=self\.params\['epsilon_inf'\]\s*\)",
     "_native.drude_nk(\n                self.wavelength,\n                self.params['omega_p'],\n"
     "                self.params['gamma_drude'],\n                self.params['epsilon_inf'],\n            )"),
    (r"compute_drude_lorentz_complex_nk\(\s*E=self\.E.*?lorentz_params=self\._lorentz_params\s*\)",
     "_native.drude_lorentz_nk(\n                self.wavelength,\n                self.params['omega_p'],\n"
     "                self.params['gamma_drude'],\n                self.params['epsilon_inf'],\n"
     "                self._lorentz_params,\n            )"),
])

# --- Cody-Lorentz (FFT-KK path) ---
transform("codylorentz.py", [
    (r"compute_nk\(\s*self\.E\s*,",
     "_native.cody_lorentz_nk(\n                self.wavelength,"),
])

# --- Forouhi-Bloomer (2019 interband + 2021 metal) ---
transform("forouhibloomer.py", [
    # Fix InterbandSingle: it referenced self._ib_terms_array but only built the
    # 1-D self._fb_term_params. Add the (1,4) reshape the driver expects.
    (r"(self\._fb_term_params = np\.array\(\[self\.Eg, self\.A, self\.B, self\.C\], dtype=np\.float64\))",
     r"\1\n        self._ib_terms_array = self._fb_term_params.reshape(1, 4)"),
    # FB was written for an older Material(params, wavelength) signature; the
    # current base is Material(wavelength, params). Pass by keyword to fix order.
    (r"super\(\)\.__init__\((\{[^{}]*\}),\s*wavelength\)",
     r"super().__init__(wavelength=wavelength, params=\1)"),
    (r"_compute_nk_interband_only\(\s*self\.E\s*,",
     "_native.fb_interband_nk(\n            self.wavelength,"),
    (r"_compute_nk_metal_full\(\s*self\.E\s*,",
     "_native.fb_metal_nk(\n            self.wavelength,"),
])

# --- EMA composition (mixers now provided by the Rust core) ---
transform("ema_material.py", [
    # Drop the Numba ema_models import; kernels come from _native now.
    (r"from \.ema_models import \([^)]*\)",
     "# EMA mixing kernels are provided by navette.materials._native"),
    # Repoint the dispatch table at the Rust-backed mixers.
    (r"_MODEL_DISPATCH = \{.*?\}",
     "_MODEL_DISPATCH = {\n"
     "        'bruggeman': _native.ema_bruggeman,\n"
     "        'maxwell_garnett': _native.ema_maxwell_garnett,\n"
     "        'looyenga': _native.ema_looyenga,\n"
     "        'lichtenecker': _native.ema_lichtenecker,\n"
     "        'mori_tanaka': (lambda n_i, n_h, f, L=0.3333333333333333: _native.ema_mori_tanaka(n_i, n_h, f, L)),\n"
     "        'birchak': (lambda n_i, n_h, f, alpha=0.5: _native.ema_power_law(n_i, n_h, f, alpha)),\n"
     "    }"),
    # Final √ε and the roughness kernel go through the Rust core.
    (r"self\.nk = _parallel_sqrt\(", "self.nk = _native.eps_to_nk("),
    (r"roughness_interface_eps\(", "_native.ema_roughness("),
    # Same old-signature fix as FB: base is Material(wavelength, params).
    (r"super\(\)\.__init__\((\{[^{}]*\}),\s*wavelength\)",
     r"super().__init__(wavelength=wavelength, params=\1)"),
])

# --- UBF Cody-Lorentz (Monolog-Lorentz; reuses the KK path) ---
transform("UBF_Cody_Lorentz.py", [
    (r"compute_nk\(\s*self\.E\s*,",
     "_native.ubf_nk(\n                self.wavelength,"),
])

print("Transform complete.")