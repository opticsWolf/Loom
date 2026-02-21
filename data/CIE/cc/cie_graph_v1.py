import sys
import json
import enum
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QComboBox, 
                               QHBoxLayout, QLabel)
from PySide6.QtGui import QPainterPath, QBrush, QPen, QColor
from PySide6.QtCore import Qt, QPointF

# --- Constants for Color Science ---

# XYZ to sRGB Matrix (Standard D65 Illuminant)
M_XYZ_TO_SRGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Applies the sRGB Gamma Correction (OETF) to linear RGB values.
    
    Args:
        linear: A numpy array of linear RGB values (0.0 to 1.0).
        
    Returns:
        A numpy array of gamma-corrected sRGB values.
    """
    srgb = np.zeros_like(linear)
    threshold = 0.0031308
    
    mask_low = linear <= threshold
    srgb[mask_low] = 12.92 * linear[mask_low]
    
    mask_high = ~mask_low
    srgb[mask_high] = 1.055 * np.power(linear[mask_high], 1.0/2.4) - 0.055
    
    return srgb

# --- Enums for Configuration ---

class CIESystem(enum.Enum):
    CIE1931 = "CIE 1931 (x, y)"
    CIE1976 = "CIE 1976 (u', v')"

class CIEObserver(enum.Enum):
    Degree2 = "2° Standard Observer"
    Degree10 = "10° Supplementary Observer"

@dataclass
class DiagramConfig:
    system: CIESystem
    observer: CIEObserver
    # Range of the graph view
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    x_label: str
    y_label: str

# --- Math & Transformation Engine ---

class CIETransform:
    """Handles vector transformations between Color Spaces."""

    @staticmethod
    def xy_to_uv_1976(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts CIE 1931 (x, y) to CIE 1976 (u', v').
        
        Math:
            u' = 4x / (-2x + 12y + 3)
            v' = 9y / (-2x + 12y + 3)
        """
        denom = -2.0 * x + 12.0 * y + 3.0
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 4.0 * x / denom
            v = 9.0 * y / denom
            
        # Clean up singularities
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        return u, v

    @staticmethod
    def uv_1976_to_xy(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts CIE 1976 (u', v') back to CIE 1931 (x, y).
        
        Used for generating the background gradient in the correct color space.
        Math:
            x = 9u' / (6u' - 16v' + 12)
            y = 4v' / (6u' - 16v' + 12)
        """
        denom = 6.0 * u - 16.0 * v + 12.0
        with np.errstate(divide='ignore', invalid='ignore'):
            x = 9.0 * u / denom
            y = 4.0 * v / denom
        
        x[~np.isfinite(x)] = 0.0
        y[~np.isfinite(y)] = 0.0
        return x, y

# --- Data Handling ---

class CIEDataHandler:
    """Manages loading and transforming spectral locus data."""
    
    def __init__(self, json_path_2deg: str, json_path_10deg: str):
        self.paths = {
            CIEObserver.Degree2: json_path_2deg,
            CIEObserver.Degree10: json_path_10deg
        }
        self.cache = {} # Cache loaded JSONs to avoid re-reading disk

    def get_locus(self, system: CIESystem, observer: CIEObserver) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns wavelength, x_coords, y_coords for the requested configuration."""
        
        # 1. Load Base Data (x, y)
        if observer not in self.cache:
            self._load_data(observer)
            
        wl, x, y = self.cache.get(observer, (None, None, None))
        
        if wl is None: 
            # Fallback if file missing
            return self._generate_fallback()

        # 2. Transform if necessary
        if system == CIESystem.CIE1976:
            x, y = CIETransform.xy_to_uv_1976(x, y)
            
        return wl, x, y

    def _load_data(self, observer: CIEObserver):
        path = self.paths[observer]
        try:
            with open(path, 'r') as f:
                content = json.load(f)
            
            # Note: JSON structure assumed based on provided v1 example.
            # Real 10deg JSONs might have key names like 'x_10(lambda)'. 
            # We normalize this logic here.
            data_block = content.get('data', content) 
            
            # Try 2deg keys first, then 10deg keys
            wl_key = 'lambda'
            x_key = 'x(lambda)' if 'x(lambda)' in data_block else 'x_10(lambda)'
            y_key = 'y(lambda)' if 'y(lambda)' in data_block else 'y_10(lambda)'
            
            # If still missing, check raw keys (sometimes data is just lists)
            if x_key not in data_block:
                # Fallback for unexpected JSON structures
                raise KeyError("Could not identify x/y keys")

            wl = np.array(data_block[wl_key]['values'], dtype=np.float64)
            x = np.array(data_block[x_key]['values'], dtype=np.float64)
            y = np.array(data_block[y_key]['values'], dtype=np.float64)
            
            self.cache[observer] = (wl, x, y)
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load data for {observer.value} from {path}. Using fallback.")
            self.cache[observer] = (None, None, None)

    def _generate_fallback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates a synthetic horseshoe for testing without files."""
        wl = np.linspace(380, 780, 100)
        t = np.linspace(0, np.pi, 100)
        x = 0.1 + 0.6 * np.sin(t)
        y = 0.1 + 0.7 * np.sin(t) * np.cos(t) + 0.4
        return wl, x, y

# --- Main Widget ---

class CIEDiagramWidget(QWidget):
    def __init__(self, data_handler: CIEDataHandler, parent=None):
        super().__init__(parent)
        self.data_handler = data_handler
        
        # UI Setup
        self.main_layout = QVBoxLayout(self)
        self._setup_controls()
        
        # Plot Setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Styling
        axis_pen = pg.mkPen(color='k', width=1)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')

        self.main_layout.addWidget(self.plot_widget)
        
        # State items
        self.current_image_item = None
        self.current_mask_item = None
        self.current_locus_curve = None
        self.label_items = []
        
        # Initial Render
        self.refresh_diagram()

    def _setup_controls(self):
        control_layout = QHBoxLayout()
        
        # System Selector
        self.combo_system = QComboBox()
        self.combo_system.addItem(CIESystem.CIE1931.value, CIESystem.CIE1931)
        self.combo_system.addItem(CIESystem.CIE1976.value, CIESystem.CIE1976)
        self.combo_system.currentIndexChanged.connect(self.refresh_diagram)
        
        # Observer Selector
        self.combo_observer = QComboBox()
        self.combo_observer.addItem(CIEObserver.Degree2.value, CIEObserver.Degree2)
        self.combo_observer.addItem(CIEObserver.Degree10.value, CIEObserver.Degree10)
        self.combo_observer.currentIndexChanged.connect(self.refresh_diagram)
        
        control_layout.addWidget(QLabel("System:"))
        control_layout.addWidget(self.combo_system)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Observer:"))
        control_layout.addWidget(self.combo_observer)
        control_layout.addStretch()
        
        self.main_layout.addLayout(control_layout)

    def refresh_diagram(self):
        """Re-calculates and renders the entire diagram based on selection."""
        system = self.combo_system.currentData()
        observer = self.combo_observer.currentData()
        
        # Fetch Data
        wl, x_locus, y_locus = self.data_handler.get_locus(system, observer)
        
        # Configure View
        self.plot_widget.clear()
        self.label_items.clear()
        
        if system == CIESystem.CIE1931:
            self.plot_widget.setRange(xRange=(-0.1, 0.9), yRange=(-0.1, 1.0))
            self.plot_widget.setLabel('bottom', "CIE x")
            self.plot_widget.setLabel('left', "CIE y")
            bounds = (-0.1, 0.9, -0.1, 0.9)
        else: # 1976
            self.plot_widget.setRange(xRange=(-0.1, 0.7), yRange=(-0.1, 0.7))
            self.plot_widget.setLabel('bottom', "CIE u'")
            self.plot_widget.setLabel('left', "CIE v'")
            bounds = (-0.1, 0.7, -0.1, 0.7)

        # 1. Generate Background Gradient
        self._render_gradient(system, bounds)
        
        # 2. Render Mask
        self._render_mask(x_locus, y_locus, bounds)
        
        # 3. Render Locus Line
        self._render_locus_outline(x_locus, y_locus)
        
        # 4. Render Labels
        self._render_labels(wl, x_locus, y_locus)

    def _render_gradient(self, system: CIESystem, bounds: Tuple[float, float, float, float]):
        """Generates the chromaticity gradient."""
        x_min, x_max, y_min, y_max = bounds
        resolution = 512 # Performance optimization: 512 is usually sufficient visually
        
        # Create grid in the *current* coordinate system
        u = np.linspace(x_min, x_max, resolution)
        v = np.linspace(y_min, y_max, resolution)
        uu, vv = np.meshgrid(u, v)
        
        uf = uu.ravel()
        vf = vv.ravel()
        
        # Convert grid to CIE 1931 (x, y) to calculate XYZ
        if system == CIESystem.CIE1976:
            xf, yf = CIETransform.uv_1976_to_xy(uf, vf)
        else:
            xf, yf = uf, vf
            
        # Calculate XYZ (Assuming Y=1 for max brightness)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_lum = np.ones_like(yf)
            X_lum = (xf / yf) * Y_lum
            Z_lum = ((1.0 - xf - yf) / yf) * Y_lum
            
            XYZ = np.vstack((X_lum, Y_lum, Z_lum)).T
            XYZ[~np.isfinite(XYZ)] = 0.0
            XYZ[XYZ < 0] = 0.0

        # XYZ -> sRGB
        RGB_linear = XYZ @ M_XYZ_TO_SRGB.T
        
        # Gamut Mapping / Normalization
        max_val = np.max(RGB_linear, axis=1, keepdims=True)
        max_val[max_val < 1.0] = 1.0 # Only scale down, don't scale up noise
        RGB_linear = RGB_linear / max_val
        RGB_linear = np.clip(RGB_linear, 0, 1)
        
        RGB_srgb = linear_to_srgb(RGB_linear)
        
        # Construct Image
        img_rgb = (RGB_srgb.reshape(resolution, resolution, 3) * 255).astype(np.uint8)
        alpha = np.full((resolution, resolution), 255, dtype=np.uint8)
        img_rgba = np.dstack((img_rgb, alpha))
        
        # PyqtGraph expects (x, y, rgba), so we transpose
        img_data = np.transpose(img_rgba, (1, 0, 2))
        
        image_item = pg.ImageItem(img_data)
        image_item.setRect(pg.QtCore.QRectF(x_min, y_min, x_max-x_min, y_max-y_min))
        image_item.setZValue(-20)
        self.plot_widget.addItem(image_item)

    def _render_mask(self, x_locus: np.ndarray, y_locus: np.ndarray, bounds: Tuple[float, float, float, float]):
        """Creates the inverted white mask."""
        if len(x_locus) == 0: return

        # Locus Path
        locus_path = QPainterPath()
        start_pt = QPointF(x_locus[0], y_locus[0])
        locus_path.moveTo(start_pt)
        for x, y in zip(x_locus[1:], y_locus[1:]):
            locus_path.lineTo(QPointF(x, y))
        locus_path.closeSubpath()
        
        # Outer Bounding Box (Giant Rectangle covering the view)
        outer_path = QPainterPath()
        # Make outer rect significantly larger than view bounds
        bx_min, bx_max, by_min, by_max = bounds
        margin = 1.0
        outer_rect = pg.QtCore.QRectF(bx_min - margin, by_min - margin, 
                                      (bx_max - bx_min) + 2*margin, 
                                      (by_max - by_min) + 2*margin)
        outer_path.addRect(outer_rect)
        
        # Subtract
        mask_path = outer_path.subtracted(locus_path)
        
        mask_item = pg.QtWidgets.QGraphicsPathItem(mask_path)
        mask_item.setBrush(QBrush(Qt.white))
        mask_item.setPen(QPen(Qt.NoPen))
        mask_item.setZValue(-10)
        self.plot_widget.addItem(mask_item)

    def _render_locus_outline(self, x_locus: np.ndarray, y_locus: np.ndarray):
        # Close the curve for the outline
        x_closed = np.append(x_locus, x_locus[0])
        y_closed = np.append(y_locus, y_locus[0])
        
        curve = pg.PlotCurveItem(
            x_closed, y_closed,
            pen=pg.mkPen(color='k', width=2),
            antialias=True
        )
        self.plot_widget.addItem(curve)

    def _render_labels(self, wl_data: np.ndarray, x_data: np.ndarray, y_data: np.ndarray):
        if wl_data is None: return

        labels_to_show = [390, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]
        
        for label_wl in labels_to_show:
            # Find index
            idx = (np.abs(wl_data - label_wl)).argmin()
            x, y = x_data[idx], y_data[idx]
            
            # Calculate Normal for text placement
            # Look neighbors (+/- 5 indices) to get decent tangent
            search_dist = 5
            prev_idx = max(0, idx - search_dist)
            next_idx = min(len(x_data) - 1, idx + search_dist)
            
            dx = x_data[next_idx] - x_data[prev_idx]
            dy = y_data[next_idx] - y_data[prev_idx]
            
            # Normal: (-dy, dx)
            nx, ny = -dy, dx
            norm = np.hypot(nx, ny)
            if norm == 0: 
                nx, ny = 1.0, 1.0
                norm = 1.414
            nx, ny = nx/norm, ny/norm
            
            # Ensure normal points outward relative to center of diagram
            # Approx center for 1931 is (0.33, 0.33), for 1976 is (0.2, 0.45)
            # A rough centroid approximation works fine
            cx, cy = np.mean(x_data), np.mean(y_data)
            if (nx * (x - cx) + ny * (y - cy)) < 0:
                nx, ny = -nx, -ny
                
            # Offset
            offset = 0.04
            text_x = x + nx * offset
            text_y = y + ny * offset
            
            text = pg.TextItem(text=str(label_wl), color='k', anchor=(0.5, 0.5))
            text.setPos(text_x, text_y)
            text.setFont(pg.QtGui.QFont("Arial", 8))
            self.plot_widget.addItem(text)
            self.label_items.append(text)
            
            # Tick
            tick = pg.PlotCurveItem(
                [x, x + nx * 0.015], [y, y + ny * 0.015],
                pen=pg.mkPen('k', width=1)
            )
            self.plot_widget.addItem(tick)

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # NOTE: You must provide paths to both your 2deg and 10deg JSON files here.
    # If 10deg file is missing, the code will gracefully fail to load it (showing fallback or empty).
    # You can reuse the 2deg file for 10deg just to see the math work, though it is physically inaccurate.
    json_2deg = "CIE_cc_1931_2deg.json" 
    json_10deg = "CIE_cc_1964_10deg.json" 
    
    data_handler = CIEDataHandler(json_2deg, json_10deg)
    
    window = CIEDiagramWidget(data_handler)
    window.resize(1000, 900)
    window.setWindowTitle("CIE Chromaticity Diagram: 1931 / 1964 / 1976")
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()