import numpy as np, loom_colorengine as ce
import colour  # reference library; the sRGB matrices are taken from here verbatim

def m(name, M):
    M = np.asarray(M, dtype=np.float64)
    rows = ",\n    ".join("[" + ", ".join(f"{x:.17e}" for x in M[i]) + "]" for i in range(3))
    return f"pub const {name}: [[f64; 3]; 3] = [\n    {rows},\n];\n"

o = ["// AUTO-GENERATED matrices (natural row-major form) + numpy inverses.",
     "// out[j] = sum_i M[j][i] * v[i]  (i.e. standard M * column-vector).", ""]

# sRGB: copied VERBATIM from colour-science's stored matrices.
# colour-science (0.4.x) hardcodes the IEC 61966-2-1 spec table at ~4 decimals
# and stores a SEPARATELY-rounded inverse, so matrix_XYZ_to_RGB is NOT the exact
# inverse of matrix_RGB_to_XYZ (round-trip ~3.5e-5). Reproducing exact parity with
# the reference requires copying BOTH matrices as-is rather than inverting one.
_srgb = colour.RGB_COLOURSPACES['sRGB']
o.append(m("M_SRGB_TO_XYZ", _srgb.matrix_RGB_to_XYZ))
o.append(m("M_XYZ_TO_SRGB", _srgb.matrix_XYZ_to_RGB))

# Oklab standard XYZ pipeline (inverses computed; these are not spec-rounded pairs)
o.append(m("M1_XYZ_TO_LMS_OKLAB", ce._M1_XYZ_TO_LMS_OKLAB))
o.append(m("M1_LMS_TO_XYZ_OKLAB", np.linalg.inv(ce._M1_XYZ_TO_LMS_OKLAB)))
o.append(m("M2_LMS_TO_LAB_OKLAB", ce._M2_LMS_TO_LAB_OKLAB))
o.append(m("M2_LAB_TO_LMS_OKLAB", np.linalg.inv(ce._M2_LMS_TO_LAB_OKLAB)))
# Oklab legacy sRGB pipeline
o.append(m("M1_OKLAB_SRGB", ce._M1_OKLAB_SRGB))
o.append(m("M1_OKLAB_SRGB_INV", np.linalg.inv(ce._M1_OKLAB_SRGB)))
o.append(m("M2_OKLAB_SRGB", ce._M2_OKLAB_SRGB))
o.append(m("M2_OKLAB_SRGB_INV", np.linalg.inv(ce._M2_OKLAB_SRGB)))
# Bradford
o.append(m("M_BRADFORD", ce._M_BRADFORD))
o.append(m("M_BRADFORD_INV", np.linalg.inv(ce._M_BRADFORD)))

open("matrices.rs", "w").write("\n".join(o))
print("wrote src/matrices.rs")
