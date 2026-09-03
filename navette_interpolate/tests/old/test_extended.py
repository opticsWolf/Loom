#!/usr/bin/env python3
"""
Test extended features: unsorted queries, derivatives, extrapolation modes,
pickling, and knot access.
"""

import numpy as np
import pickle
import pytest
import navette_interpolator

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def test_unsorted_queries():
    """Check that unsorted target_x works correctly (using general kernel)."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 4.0, 9.0])   # y = x^2
    spline = navette_interpolator.UniSpline(x, y, method="pchip")

    # Unsorted query
    t_unsorted = np.array([2.5, 0.5, 1.5, 3.5])
    result = spline.eval(t_unsorted)
    
    # PCHIP does not perfectly reconstruct quadratics. 
    # These are the exact mathematical outputs of the PCHIP algorithm (matches SciPy):
    expected = np.array([6.21875, 0.3125, 2.21875, 12.0])   
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    print("✅ unsorted queries work")

def test_derivatives():
    """Check first derivative against analytical derivative."""
    x = np.linspace(0, 2, 10)
    y = x ** 3
    spline = navette_interpolator.UniSpline(x, y, method="pchip")
    t = np.linspace(0, 2, 100)
    deriv = spline.derivative(t)
    expected = 3 * t ** 2
    
    # PCHIP is an approximation; its derivative won't exactly match the 
    # analytical derivative of the sampled function (especially near zero).
    # We use a relaxed tolerance (atol) for this sanity check.
    np.testing.assert_allclose(deriv, expected, atol=0.15, rtol=0.15)
    print("✅ derivatives work")

def test_extrapolation_modes():
    """Test linear, clamp, and error extrapolation."""
    # Add .0 to ensure these are created as float64 arrays
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    t_out = np.array([-1.0, 3.0])

    # Linear extrapolation (default)
    spline_lin = navette_interpolator.UniSpline(x, y, extrap="linear")
    res_lin = spline_lin.eval(t_out)
    # linear extrapolation using end slopes: left slope = (1-0)/(1-0)=1 → -1; right slope = (4-1)/(2-1)=3 → 4+3*(3-2)=7
    expected_lin = np.array([-1.0, 7.0])
    np.testing.assert_allclose(res_lin, expected_lin)

    # Clamp
    spline_clamp = navette_interpolator.UniSpline(x, y, extrap="clamp")
    res_clamp = spline_clamp.eval(t_out)
    np.testing.assert_allclose(res_clamp, [0.0, 4.0])

    # Error (should raise)
    spline_err = navette_interpolator.UniSpline(x, y, extrap="error")
    with pytest.raises(Exception):
        spline_err.eval(t_out)
    print("✅ extrapolation modes work")

def test_pickling():
    """Save and load spline via pickle."""
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    spline = navette_interpolator.UniSpline(x, y, method="pchip")

    # Pickle
    data = pickle.dumps(spline)
    spline2 = pickle.loads(data)

    # Compare evaluation
    t = np.linspace(0, 10, 50)
    np.testing.assert_allclose(spline.eval(t), spline2.eval(t))
    print("✅ pickling works")

def test_knot_access():
    """Get x, y, slopes from the spline."""
    # Ensure arrays are float64
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([[0.0, 1.0, 4.0], [1.0, 2.0, 5.0]])   # two signals
    spline = navette_interpolator.UniSpline(x, y, method="pchip")

    # Get x
    np.testing.assert_equal(spline.get_x(), x)

    # Get y (full 2D)
    np.testing.assert_equal(spline.get_y(), y)

    # Get y (single signal)
    np.testing.assert_equal(spline.get_y(0), y[0])

    # Get slopes (if available for PCHIP)
    slopes = spline.get_slopes()
    assert slopes is not None
    assert slopes.shape == (2, 3)
    print("✅ knot access works")

def test_against_scipy_numpy():
    """Test interpolation results directly against SciPy and NumPy."""
    try:
        import scipy.interpolate as si
    except ImportError:
        print("⚠️ SciPy not installed, skipping test_against_scipy_numpy.")
        return

    # Create a noisy sine wave to interpolate
    x = np.linspace(0, 10, 15)
    y = np.sin(x) + np.random.default_rng(42).normal(scale=0.1, size=x.shape)
    t = np.linspace(x[0] - 1, x[-1] + 1, 100)

    # 1. Linear vs NumPy (numpy.interp clamps out of bounds by default)
    spline_lin = navette_interpolator.UniSpline(x, y, method="linear", extrap="clamp")
    res_lin = spline_lin.eval(t)
    res_np = np.interp(t, x, y)
    np.testing.assert_allclose(res_lin, res_np, rtol=1e-10, atol=1e-10)

    # 2. PCHIP vs SciPy PchipInterpolator (linear extrapolation by default)
    spline_pchip = navette_interpolator.UniSpline(x, y, method="pchip", extrap="linear")
    res_pchip = spline_pchip.eval(t)
    pchip_scipy = si.PchipInterpolator(x, y, extrapolate=True)
    np.testing.assert_allclose(res_pchip, pchip_scipy(t), rtol=1e-10, atol=1e-10)

    # 3. Makima vs SciPy Akima1DInterpolator (if supported by the SciPy version)
    spline_makima = navette_interpolator.UniSpline(x, y, method="makima", extrap="linear")
    res_makima = spline_makima.eval(t)
    try:
        # SciPy >= 1.7.0 supports the explicit 'makima' method argument
        makima_scipy = si.Akima1DInterpolator(x, y, method="makima", extrapolate=True)
        np.testing.assert_allclose(res_makima, makima_scipy(t), rtol=1e-10, atol=1e-10)
    except TypeError:
        # Fallback for older SciPy versions that don't support method="makima" natively
        pass

    print("✅ scipy/numpy comparison works")

if __name__ == "__main__":
    test_unsorted_queries()
    test_derivatives()
    test_extrapolation_modes()
    test_pickling()
    test_knot_access()
    test_against_scipy_numpy()
    print("\n🎉 All extended tests passed!")