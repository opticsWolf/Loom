"""
Migration for LoomScatterMatrix after dropping func_4 / func_5.

Both `compute_ellipsometry` and `compute_RT` now route through the single
`core_engine` (func_6) with a request mask. Paste these into the class,
renaming the `self._*` attributes to whatever your __init__ actually stores.

func_6 returns a dict keyed by observable name. The keys it can emit (relevant
ones here):
    "Rs" "Rp" "Ts" "Tp" "R_avg" "T_avg" "A_s" "A_p" "A_avg"
    "Psi_R" "Delta_R" "DOP_R" "Psi_T" "Delta_T" "DOP_T"
    "Diattenuation_R/T" "S0_R".."S3_T" "phi_rs/rp/ts/tp"
    "rs_c" "rp_c" "ts_c" "tp_c" "cross_R" "cross_T"
If downstream code expects the old key spellings, alias them in the dicts below.

NOTE: func_6 has no "conservation" observable (func_4's debug diagnostic). It is
derived here in Python from the intensities, which is exact and free.
"""

import numpy as np
from request_flags import Request  # CoherenceMode also available there

try:
    from smatrix import core_engine
except ImportError:  # adjust to your build/layout
    from loom_matrix import core_engine


# ---- paste the following into LoomScatterMatrix ----------------------------

def _run(self, requested):
    """Single entry point to the Rust engine. Returns the raw {name: ndarray}
    dict (arrays shaped [num_angles, num_wavs])."""
    return core_engine(
        self._wavls,            # float64 ndarray
        self._sin_theta,        # float64 ndarray
        int(self._n_layers),
        self._n_stack_cache,    # float64 ndarray (re/im interleaved)
        self._thicknesses,      # float64 ndarray
        self._incoherent_flags, # int32 ndarray
        self._rough_types,      # int32 ndarray
        self._rough_vals,       # float64 ndarray
        int(self._coherence_mode),  # 0 front_block, 1 coherency_matrix, 2 fully_coherent
        int(requested),
    )


def compute_ellipsometry(self):
    req = (Request.PSI_R | Request.DELTA_R | Request.DOP_R
           | Request.RS | Request.RP | Request.R_AVG
           | Request.PSI_T | Request.DELTA_T | Request.DOP_T
           | Request.TS | Request.TP | Request.T_AVG)
    out = self._run(req)
    res = {k: self._squeeze(v) for k, v in out.items()}
    if self.debug:
        Rs, Rp, Ts, Tp = out["Rs"], out["Rp"], out["Ts"], out["Tp"]
        cons = np.maximum(np.abs(1.0 - Rs - Ts), np.abs(1.0 - Rp - Tp))
        res["conservation"] = self._squeeze(cons)
    return res


def compute_RT(self, mode='u'):
    """mode: 's', 'p', or 'u' (unpolarized = both)."""
    req = Request.R_AVG | Request.T_AVG
    if mode in ('s', 'u'):
        req |= Request.RS | Request.TS
    if mode in ('p', 'u'):
        req |= Request.RP | Request.TP
    out = self._run(req)
    return {k: self._squeeze(v) for k, v in out.items()}
