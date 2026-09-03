# -*- coding: utf-8 -*-
"""
UniSpline Cross-Implementation Benchmark Suite (PySide6 Edition)
Compares: NumPy, SciPy, Python (Numba), Rust (PyO3)
"""
import sys
import time
import numpy as np
import gc
from typing import Dict, Tuple, Callable

# --- 1. Qt Binding Selection ---
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
except ImportError:
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
import pyqtgraph as pg

# --- 2. Import Backends ---
# Python (Numba) -> Rename your file to loom_unispline_py.py!
try:
    import loom_unispline as unispline_py
    HAS_PY = True
except ImportError:
    HAS_PY = False
    print("⚠️ loom_unispline_py.py not found. Skipping Python backend.")

# Rust (PyO3) -> Built via maturin
try:
    import navette_interpolator
    HAS_RS = True
except ImportError:
    HAS_RS = False
    print("⚠️ navette_interpolator (Rust) not found. Skipping Rust backend.")

# SciPy Reference
try:
    from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ SciPy not found. Skipping SciPy references.")

# --- 3. Wrappers ---
def run_numpy(tgt, src, val): return np.interp(tgt, src, val)
def run_scipy_pchip(tgt, src, val): return PchipInterpolator(src, val)(tgt)
def run_scipy_akima(tgt, src, val): return Akima1DInterpolator(src, val)(tgt)

def run_py(tgt, src, val, method): return unispline_py.UniSpline(src, val, method=method)(tgt)
def run_rs(tgt, src, val, method): return navette_interpolator.UniSpline(src, val, method=method)(tgt)

# --- 4. Dataset Generation ---
def generate_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    datasets = {}
    sx = np.linspace(380, 780, 21)
    datasets["1. Smooth Peak (CIE)"] = (sx, 100 * np.exp(-0.5 * ((sx - 550) / 40)**2), np.linspace(380, 780, 401))
    
    sx = np.linspace(0, 10, 11)
    datasets["2. Hard Step"] = (sx, np.array([0,0,0,1,1,1,1,0,0,0,0], dtype=float), np.linspace(0, 10, 500))
    
    sx = np.linspace(-1, 1, 15)
    datasets["3. Runge Function"] = (sx, 1.0 / (1.0 + 25.0 * sx**2), np.linspace(-1, 1, 500))
    
    rng = np.random.default_rng(42)
    sx = np.linspace(0, 100, 20)
    datasets["4. Spiky Random"] = (sx, rng.random(20) * 100, np.linspace(0, 100, 1000))
    return datasets

def benchmark_kernel(func: Callable, src, val, tgt, n_loops=2000) -> float:
    try: func(tgt, src, val)
    except Exception: return -1.0
    
    gc_old = gc.isenabled(); gc.disable()
    try:
        t0 = time.perf_counter()
        for _ in range(n_loops): func(tgt, src, val)
        t1 = time.perf_counter()
    finally:
        if gc_old: gc.enable()
    return ((t1 - t0) * 1000.0) / n_loops

def run_suite():
    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(background='k', foreground='d', antialias=True)
    win = pg.GraphicsLayoutWidget(show=True, title="UniSpline Cross-Implementation Suite")
    win.resize(1400, 900)

    datasets = generate_datasets()
    
    # Dynamically build methods list based on available backends
    methods = [
        ("NumPy (Ref)", lambda t,s,v: run_numpy(t,s,v), (150,150,150), Qt.SolidLine, "linear"),
    ]
    if HAS_SCIPY:
        methods.extend([
            ("SciPy PCHIP", lambda t,s,v: run_scipy_pchip(t,s,v), (255,128,0), Qt.DashLine, "pchip"),
            ("SciPy Akima", lambda t,s,v: run_scipy_akima(t,s,v), (128,255,0), Qt.DashLine, "makima"),
        ])
    if HAS_PY:
        methods.extend([
            ("Py PCHIP", lambda t,s,v: run_py(t,s,v,"pchip"), (0,255,255), Qt.DashDotLine, "pchip"),
            ("Py Makima", lambda t,s,v: run_py(t,s,v,"makima"), (255,255,0), Qt.SolidLine, "makima"),
            ("Py Sprague", lambda t,s,v: run_py(t,s,v,"sprague"), (0,255,0), Qt.DashLine, "sprague"),
        ])
    if HAS_RS:
        methods.extend([
            ("Rust PCHIP", lambda t,s,v: run_rs(t,s,v,"pchip"), (0,128,255), Qt.DashDotLine, "pchip"),
            ("Rust Makima", lambda t,s,v: run_rs(t,s,v,"makima"), (255,0,255), Qt.SolidLine, "makima"),
            ("Rust Sprague", lambda t,s,v: run_rs(t,s,v,"sprague"), (255,0,128), Qt.DashLine, "sprague"),
            ("Rust FH", lambda t,s,v: run_rs(t,s,v,"fh"), (64,128,255), Qt.DashDotLine, "fh"),
        ])

    print(f"\n{'Dataset':<25} | {'Method':<15} | {'Time (ms)':<10} | {'Max Error vs Ref':<15} | {'Status'}")
    print("="*85)

    for i, (ds_name, (sx, sy, tx)) in enumerate(datasets.items()):
        p = win.addPlot(title=ds_name)
        p.addLegend()
        p.plot(sx, sy, pen=None, symbol='o', symbolBrush=(200, 50, 50), symbolSize=9, name="Knots")
        
        # Calculate Reference Output for Accuracy Checking
        ref_out = run_numpy(tx, sx, sy) if "linear" in ds_name.lower() else (run_scipy_pchip(tx, sx, sy) if HAS_SCIPY else run_numpy(tx, sx, sy))

        for m_name, m_func, m_col, m_style, m_type in methods:
            if m_type == "sprague" and "Spiky" in ds_name: continue # Avoid Runge explosion in GUI
            
            avg_ms = benchmark_kernel(m_func, sx, sy, tx, n_loops=1000 if "Sprague" in m_name else 2000)
            try:
                y_out = m_func(tx, sx, sy)
                max_err = np.max(np.abs(y_out - ref_out))
                err_str = f"{max_err:.2e}"
                status = "✅ OK" if max_err < 1e-6 else ("⚠️ Dev" if max_err < 1e-2 else "❌ Fail")
                
                pen = pg.mkPen(color=m_col, width=2, style=m_style)
                p.plot(tx, y_out, pen=pen, name=m_name)
                print(f"{ds_name:<25} | {m_name:<15} | {avg_ms:6.4f} ms | {err_str:<15} | {status}")
            except Exception as e:
                print(f"{ds_name:<25} | {m_name:<15} | FAILED     | N/A             | {e}")

        if (i + 1) % 2 == 0: win.nextRow()

    if hasattr(app, 'exec'): app.exec()
    else: app.exec_()

if __name__ == "__main__":
    run_suite()