"""Request flags for the unified ``core_engine`` (func_6).

OR these together and pass the integer as the ``requested`` argument. The engine
returns a dict containing exactly the requested keys (plus, for each ``DISP_*``
flag, the four dispersion arrays GD/GDD/TOD/FOD for that channel).

    from request_flags import Request, CoherenceMode, expected_keys
    from smatrix import core_engine

    req = Request.RS | Request.PSI_R          # or a bundle: Request.PHOTOMETRY
    out = core_engine(wavls, sin_theta, n_layers, n_cache, thick,
                      inc, rtypes, rvals,
                      int(CoherenceMode.COHERENCY_MATRIX), int(req))
    assert set(out) == set(expected_keys(req))

This module is the single source of truth for the request interface. The bit
positions MUST stay in sync with the ``REQ_*`` constants in ``func_6.rs``; the
``ScatterMatrix`` wrapper should import ``Request``/``CoherenceMode`` from here
rather than redefining them.
"""

from enum import IntEnum, IntFlag
from typing import List, Union

__all__ = ["Request", "CoherenceMode", "ALL_OBSERVABLES", "expected_keys"]


class Request(IntFlag):
    # ── Intensities (Intensities level) ──
    RS = 1 << 0
    RP = 1 << 1
    TS = 1 << 2
    TP = 1 << 3
    R_AVG = 1 << 4
    T_AVG = 1 << 5
    A_S = 1 << 6
    A_P = 1 << 7
    A_AVG = 1 << 8
    PSI_R = 1 << 9
    PSI_T = 1 << 10
    # ── Coherency channel (Cross level) ──
    DELTA_R = 1 << 11
    DELTA_T = 1 << 12
    DOP_R = 1 << 13
    DOP_T = 1 << 14
    DIATT_R = 1 << 15
    DIATT_T = 1 << 16
    S0_R = 1 << 17
    S1_R = 1 << 18
    S2_R = 1 << 19
    S3_R = 1 << 20
    S0_T = 1 << 21
    S1_T = 1 << 22
    S2_T = 1 << 23
    S3_T = 1 << 24
    # ── Complex amplitudes / phases (ComplexAmps level) ──
    PHI_RS = 1 << 25
    PHI_RP = 1 << 26
    PHI_TS = 1 << 27
    PHI_TP = 1 << 28
    RS_C = 1 << 29
    RP_C = 1 << 30
    TS_C = 1 << 31
    TP_C = 1 << 32
    # ── Coherency channel, raw + retardance (Cross level) ──
    CROSS_R = 1 << 33
    CROSS_T = 1 << 34
    RETARD_R = 1 << 35
    RETARD_T = 1 << 36
    # ── Dispersion (ComplexAmps level); each emits GD/GDD/TOD/FOD ──
    DISP_R_S = 1 << 37   # -> GD_R_s, GDD_R_s, TOD_R_s, FOD_R_s
    DISP_R_P = 1 << 38   # -> GD_R_p, GDD_R_p, TOD_R_p, FOD_R_p
    DISP_T_S = 1 << 39   # -> GD_T_s, GDD_T_s, TOD_T_s, FOD_T_s
    DISP_T_P = 1 << 40   # -> GD_T_p, GDD_T_p, TOD_T_p, FOD_T_p

    # ── Convenience bundles ──
    PHOTOMETRY = RS | RP | TS | TP | R_AVG | T_AVG
    ELLIPSOMETRY = PSI_R | DELTA_R | DOP_R | PSI_T | DELTA_T | DOP_T
    ABSORPTION = A_S | A_P | A_AVG
    STOKES_R = S0_R | S1_R | S2_R | S3_R
    STOKES_T = S0_T | S1_T | S2_T | S3_T
    STOKES = STOKES_R | STOKES_T
    DIATTENUATION = DIATT_R | DIATT_T
    RETARDANCE = RETARD_R | RETARD_T
    CROSS = CROSS_R | CROSS_T
    PHASES = PHI_RS | PHI_RP | PHI_TS | PHI_TP
    COMPLEX_AMPS = RS_C | RP_C | TS_C | TP_C
    DISPERSION = DISP_R_S | DISP_R_P | DISP_T_S | DISP_T_P


#: Every defined observable bit (0..40) OR-ed together. Useful for tests or for
#: "give me everything" requests. Composed from the atomic flags so it never
#: includes an undefined bit.
ALL_OBSERVABLES = Request((1 << 41) - 1)


class CoherenceMode(IntEnum):
    """How incoherent boundaries and the p-s coherency channel are handled.

    These are mutually exclusive integer modes, not flags -- do not OR them.
    """

    FRONT_BLOCK = 0       # incoherent gaps applied at flagged boundaries
    COHERENCY_MATRIX = 1  # tracks the complex p-s coherency channel (Mode B)
    FULLY_COHERENT = 2    # whole stack treated as one coherent block


# ─── Request bit -> output dict key(s) the engine emits ──────────────────────
# Mirrors the emit_* calls in func_6.rs. Single source for "which keys come
# back for a given mask".
_SCALAR_KEYS = {
    Request.RS: "Rs", Request.RP: "Rp", Request.TS: "Ts", Request.TP: "Tp",
    Request.R_AVG: "R_avg", Request.T_AVG: "T_avg",
    Request.A_S: "A_s", Request.A_P: "A_p", Request.A_AVG: "A_avg",
    Request.PSI_R: "Psi_R", Request.PSI_T: "Psi_T",
    Request.DELTA_R: "Delta_R", Request.DELTA_T: "Delta_T",
    Request.DOP_R: "DOP_R", Request.DOP_T: "DOP_T",
    Request.DIATT_R: "Diattenuation_R", Request.DIATT_T: "Diattenuation_T",
    Request.S0_R: "S0_R", Request.S1_R: "S1_R", Request.S2_R: "S2_R", Request.S3_R: "S3_R",
    Request.S0_T: "S0_T", Request.S1_T: "S1_T", Request.S2_T: "S2_T", Request.S3_T: "S3_T",
    Request.RETARD_R: "Retardance_R", Request.RETARD_T: "Retardance_T",
    Request.PHI_RS: "phi_rs", Request.PHI_RP: "phi_rp",
    Request.PHI_TS: "phi_ts", Request.PHI_TP: "phi_tp",
    Request.RS_C: "rs_c", Request.RP_C: "rp_c", Request.TS_C: "ts_c", Request.TP_C: "tp_c",
    Request.CROSS_R: "cross_R", Request.CROSS_T: "cross_T",
}
_DISP_KEYS = {
    Request.DISP_R_S: ("GD_R_s", "GDD_R_s", "TOD_R_s", "FOD_R_s"),
    Request.DISP_R_P: ("GD_R_p", "GDD_R_p", "TOD_R_p", "FOD_R_p"),
    Request.DISP_T_S: ("GD_T_s", "GDD_T_s", "TOD_T_s", "FOD_T_s"),
    Request.DISP_T_P: ("GD_T_p", "GDD_T_p", "TOD_T_p", "FOD_T_p"),
}


def expected_keys(request: Union[int, "Request"]) -> List[str]:
    """Return the output dict keys ``core_engine`` will emit for ``request``.

    Works for any mask, including the convenience bundles (they are just unions
    of the atomic flags). Order is stable: scalar observables in bit order,
    then each requested dispersion channel's four arrays.
    """
    req = Request(int(request))
    keys: List[str] = [name for bit, name in _SCALAR_KEYS.items() if req & bit]
    for bit, names in _DISP_KEYS.items():
        if req & bit:
            keys.extend(names)
    return keys
