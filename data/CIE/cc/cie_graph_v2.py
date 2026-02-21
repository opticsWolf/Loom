import sys
import json
import enum
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QComboBox, QLabel)
from PySide6.QtGui import QPainterPath, QBrush, QPen, QColor, QPolygonF
from PySide6.QtCore import Qt, QPointF

# --- Constants for Color Science ---

# XYZ to sRGB Matrix (Standard D65 Illuminant)
M_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Applies the sRGB Gamma Correction (OETF) to linear RGB values."""
    srgb = np.zeros_like(linear)
    threshold = 0.0031308
    
    mask_low = linear <= threshold
    srgb[mask_low] = 12.92 * linear[mask_low]
    
    mask_high = ~mask_low
    srgb[mask_high] = 1.055 * np.power(linear[mask_high], 1.0/2.4) - 0.055
    
    return srgb

# --- Enums & Math ---

class CIESystem(enum.Enum):
    CIE1931 = "CIE 1931 (x, y)"
    CIE1960 = "CIE 1960 (u, v)"
    CIE1976 = "CIE 1976 (u', v')"

class CIEObserver(enum.Enum):
    Degree2 = "2° Standard Observer"
    Degree10 = "10° Supplementary Observer"

class CIETransform:
    """Handles vector transformations between Color Spaces."""

    @staticmethod
    def xy_to_uv1960(x, y):
        denom = -2 * x + 12 * y + 3
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 4 * x / denom
            v = 6 * y / denom
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        return u, v

    @staticmethod
    def xy_to_uv1976(x, y):
        denom = -2 * x + 12 * y + 3
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 4 * x / denom
            v = 9 * y / denom
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        return u, v

    @staticmethod
    def uv1960_to_xy(u, v):
        """Inverse 1960 -> 1931"""
        denom = 2 * u - 8 * v + 4
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (3 * u) / denom
            y = (2 * v) / denom
        x[~np.isfinite(x)] = 0.0
        y[~np.isfinite(y)] = 0.0
        return x, y

    @staticmethod
    def uv1976_to_xy(u, v):
        """Inverse 1976 -> 1931"""
        denom = 6 * u - 16 * v + 12
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (9 * u) / denom
            y = (4 * v) / denom
        x[~np.isfinite(x)] = 0.0
        y[~np.isfinite(y)] = 0.0
        return x, y

# --- Data Handler ---

class CIEDataHandler:
    """Handles loading and transforming chromaticity coordinates."""
    
    def __init__(self, json_path_2deg: str, json_path_10deg: str):
        self.paths = {
            CIEObserver.Degree2: json_path_2deg,
            CIEObserver.Degree10: json_path_10deg
        }
        self.cache = {} 

    def get_data(self, system: CIESystem, observer: CIEObserver):
        """Returns (wavelengths, x_coords, y_coords) for the config."""
        
        # 1. Load Base Data (x, y) if not cached
        if observer not in self.cache:
            self._load_data(observer)
            
        wl, x, y = self.cache.get(observer)
        
        # 2. Transform if necessary
        if system == CIESystem.CIE1931:
            return wl, x, y
        elif system == CIESystem.CIE1960:
            u, v = CIETransform.xy_to_uv1960(x, y)
            return wl, u, v
        elif system == CIESystem.CIE1976:
            u, v = CIETransform.xy_to_uv1976(x, y)
            return wl, u, v
            
        return wl, x, y

    def _load_data(self, observer: CIEObserver):
        path = self.paths[observer]
        try:
            with open(path, 'r') as f:
                content = json.load(f)
            
            data_block = content.get('data', content)
            
            # Smart key detection for 2deg vs 10deg
            wl_key = 'lambda'
            x_key = 'x(lambda)' if 'x(lambda)' in data_block else 'x_10(lambda)'
            y_key = 'y(lambda)' if 'y(lambda)' in data_block else 'y_10(lambda)'
            
            wl = np.array(data_block[wl_key]['values'], dtype=np.float64)
            x = np.array(data_block[x_key]['values'], dtype=np.float64)
            y = np.array(data_block[y_key]['values'], dtype=np.float64)
            
            self.cache[observer] = (wl, x, y)
            
        except Exception as e:
            print(f"Error loading '{path}': {e}")
            # Fallback
            wl = np.linspace(380, 780, 100)
            t = np.linspace(0, np.pi, 100)
            x = 0.1 + 0.6 * np.sin(t)
            y = 0.1 + 0.7 * np.sin(t) * np.cos(t) + 0.4
            self.cache[observer] = (wl, x, y)

# --- Main Widget ---

class CIEWidget(QWidget):
    def __init__(self, data_handler: CIEDataHandler, parent=None):
        super().__init__(parent)
        self.data_handler = data_handler
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Controls
        control_layout = QHBoxLayout()
        self.combo_sys = QComboBox()
        self.combo_sys.addItems([e.name for e in CIESystem])
        self.combo_sys.currentIndexChanged.connect(self.refresh_diagram)
        
        self.combo_obs = QComboBox()
        self.combo_obs.addItems([e.name for e in CIEObserver])
        self.combo_obs.currentIndexChanged.connect(self.refresh_diagram)
        
        control_layout.addWidget(QLabel("System:"))
        control_layout.addWidget(self.combo_sys)
        control_layout.addWidget(QLabel("Observer:"))
        control_layout.addWidget(self.combo_obs)
        control_layout.addStretch()
        self.layout.addLayout(control_layout)

        # 2. Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        
        self.layout.addWidget(self.plot_widget)
        
        # Initial Draw
        self.refresh_diagram()

    def refresh_diagram(self):
        self.plot_widget.clear()
        
        sys_enum = CIESystem[self.combo_sys.currentText()]
        obs_enum = CIEObserver[self.combo_obs.currentText()]
        
        # 1. Get Transformed Locus
        wl, x_locus, y_locus = self.data_handler.get_data(sys_enum, obs_enum)
        
        # 2. Configure View Bounds & Labels
        if sys_enum == CIESystem.CIE1931:
            bounds = (0.0, 0.8, 0.0, 0.9) # x_min, x_max, y_min, y_max
            self.plot_widget.setLabel('bottom', "CIE x")
            self.plot_widget.setLabel('left', "CIE y")
        else: # 1960 or 1976
            bounds = (0.0, 0.65, 0.0, 0.65)
            lbl_u = "CIE u" if sys_enum == CIESystem.CIE1960 else "CIE u'"
            lbl_v = "CIE v" if sys_enum == CIESystem.CIE1960 else "CIE v'"
            self.plot_widget.setLabel('bottom', lbl_u)
            self.plot_widget.setLabel('left', lbl_v)
            
        self.plot_widget.setRange(xRange=(bounds[0], bounds[1]), 
                                  yRange=(bounds[2], bounds[3]))

        # 3. Generate Background (Inverse Mapped Gradient)
        # Note: We generate a FULL rectangular gradient here without alpha masking.
        # The vector mask in step 4 will hide the invalid areas.
        img_rgba, rect = self._generate_chromaticity_grid(
            sys_enum, bounds
        )
        
        image_item = pg.ImageItem(img_rgba)
        image_item.setRect(pg.QtCore.QRectF(rect[0], rect[2], 
                                            rect[1]-rect[0], 
                                            rect[3]-rect[2]))
        image_item.setZValue(-20) # Deep background
        self.plot_widget.addItem(image_item)

        # 4. Vector Masking (The "Cookie Cutter")
        # Create a white overlay that covers everything *except* the locus
        self._apply_vector_mask(x_locus, y_locus, bounds)

        # 5. Draw Spectral Locus Line
        x_closed = np.append(x_locus, x_locus[0])
        y_closed = np.append(y_locus, y_locus[0])
        self.plot_widget.addItem(pg.PlotCurveItem(
            x_closed, y_closed, 
            pen=pg.mkPen(color='k', width=2), 
            antialias=True
        ))
        
        # 6. Add Labels
        self._add_labels(wl, x_locus, y_locus)

    def _generate_chromaticity_grid(self, system, bounds, resolution=512):
        """
        Generates the background gradient.
        1. Creates grid in u,v space.
        2. Inverse maps to x,y to get RGB.
        3. Returns full rectangular texture (no alpha masking).
        """
        x_min, x_max, y_min, y_max = bounds
        
        # 1. Create Grid in Current System Coordinates
        u = np.linspace(x_min, x_max, resolution)
        v = np.linspace(y_min, y_max, resolution)
        uu, vv = np.meshgrid(u, v)
        
        uf = uu.ravel()
        vf = vv.ravel()
        
        # 2. Inverse Map to CIE 1931 (x, y) for Color Calculation
        if system == CIESystem.CIE1931:
            xf, yf = uf, vf
        elif system == CIESystem.CIE1960:
            xf, yf = CIETransform.uv1960_to_xy(uf, vf)
        elif system == CIESystem.CIE1976:
            xf, yf = CIETransform.uv1976_to_xy(uf, vf)
            
        # 3. xy -> XYZ (assuming Y=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_lum = np.ones_like(yf)
            X_lum = (xf / yf) * Y_lum
            Z_lum = ((1.0 - xf - yf) / yf) * Y_lum
            
            XYZ = np.vstack((X_lum, Y_lum, Z_lum)).T
            XYZ[~np.isfinite(XYZ)] = 0.0
            XYZ[XYZ < 0] = 0.0

        # 4. XYZ -> sRGB
        RGB_linear = XYZ @ M_XYZ_TO_SRGB.T
        max_val = np.max(RGB_linear, axis=1, keepdims=True)
        max_val[max_val < 1.0] = 1.0
        RGB_linear = RGB_linear / max_val
        RGB_linear = np.clip(RGB_linear, 0, 1)
        RGB_srgb = linear_to_srgb(RGB_linear)
        
        # 5. Format Texture
        img_rgb = (RGB_srgb.reshape(resolution, resolution, 3) * 255).astype(np.uint8)
        
        # Add opaque alpha channel
        alpha = np.full((resolution, resolution), 255, dtype=np.uint8)
        img_rgba = np.dstack((img_rgb, alpha))
        img_rgba = np.transpose(img_rgba, (1, 0, 2)) # Transpose for PyQtGraph
        
        return img_rgba, (x_min, x_max, y_min, y_max)

    def _apply_vector_mask(self, x_locus, y_locus, bounds):
        """Creates an inverted Vector Mask (White overlay with a hole)."""
        if len(x_locus) == 0: return

        # 1. Locus Path (The "Hole")
        locus_path = QPainterPath()
        locus_path.moveTo(QPointF(x_locus[0], y_locus[0]))
        for x, y in zip(x_locus[1:], y_locus[1:]):
            locus_path.lineTo(QPointF(x, y))
        locus_path.closeSubpath()
        
        # 2. Outer Bounding Box (The "Paper")
        # Make it much larger than the visible area to ensure coverage
        outer_path = QPainterPath()
        bx_min, bx_max, by_min, by_max = bounds
        margin = 1.0
        rect = pg.QtCore.QRectF(bx_min - margin, by_min - margin, 
                                (bx_max - bx_min) + 2*margin, 
                                (by_max - by_min) + 2*margin)
        outer_path.addRect(rect)
        
        # 3. Geometric Subtraction: Mask = Outer - Locus
        mask_path = outer_path.subtracted(locus_path)
        
        # 4. Add to Scene
        mask_item = pg.QtWidgets.QGraphicsPathItem(mask_path)
        mask_item.setBrush(QBrush(Qt.white)) # Background color
        mask_item.setPen(QPen(Qt.NoPen))     # No outline
        mask_item.setZValue(-10)             # Above image, below lines
        self.plot_widget.addItem(mask_item)

    def _add_labels(self, wl, x_locus, y_locus):
        if wl is None: return
        labels = [390, 450, 460, 470, 480, 490, 500, 510, 520, 540, 560, 580, 600, 620, 700]
        
        for label_wl in labels:
            idx = (np.abs(wl - label_wl)).argmin()
            x, y = x_locus[idx], y_locus[idx]
            
            # Simple centroid-based offset
            cx, cy = np.mean(x_locus), np.mean(y_locus)
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy)
            dx, dy = dx/dist, dy/dist
            
            text = pg.TextItem(text=str(label_wl), color='k', anchor=(0.5, 0.5))
            text.setPos(x + dx * 0.04, y + dy * 0.04)
            self.plot_widget.addItem(text)
            self.plot_widget.addItem(pg.PlotCurveItem(
                [x, x + dx*0.015], [y, y + dy*0.015], pen=pg.mkPen('k', width=1)
            ))

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Ensure you have both files or copies of the 2deg file named appropriately
    json_2deg = "CIE_cc_1931_2deg.json"
    json_10deg = "CIE_cc_1964_10deg.json"
    
    handler = CIEDataHandler(json_2deg, json_10deg)
    
    window = CIEWidget(handler)
    window.resize(900, 900)
    window.setWindowTitle("CIE Diagram (Vector Masked)")
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()