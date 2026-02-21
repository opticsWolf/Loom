import sys
import json
import enum
import numpy as np
import pyqtgraph as pg
from typing import Tuple, Dict, Optional, List
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QComboBox, QLabel, QMessageBox, QFrame)
from PySide6.QtGui import QPainterPath, QBrush, QPen, QColor
from PySide6.QtCore import Qt, QRectF

# --- Constants & Math ---

# XYZ to sRGB Matrix (Standard D65 Illuminant)
M_XYZ_TO_SRGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Applies sRGB Gamma Correction to linear RGB values.
    
    Args:
        linear: A numpy array of linear RGB values.

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

class CIESystem(enum.Enum):
    CIE1931 = "CIE 1931 (x, y)"
    CIE1960 = "CIE 1960 (u, v)"
    CIE1976 = "CIE 1976 (u', v')"

class CIEObserver(enum.Enum):
    Degree2 = "2° Observer"
    Degree10 = "10° Observer"

class CIEDataSource(enum.Enum):
    ConeFundamental = "Cone Fundamental (2015)"
    StandardXYZ = "Standard XYZ (1931/1964)"

class CIETransform:
    """Vector transformations between Color Spaces."""
    
    @staticmethod
    def xy_to_uv1960(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = -2 * x + 12 * y + 3
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 4 * x / denom
            v = 6 * y / denom
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        return u, v

    @staticmethod
    def xy_to_uv1976(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = -2 * x + 12 * y + 3
        with np.errstate(divide='ignore', invalid='ignore'):
            u = 4 * x / denom
            v = 9 * y / denom
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        return u, v

    @staticmethod
    def uv1960_to_xy(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = 2 * u - 8 * v + 4
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (3 * u) / denom
            y = (2 * v) / denom
        x[~np.isfinite(x)] = 0.0
        y[~np.isfinite(y)] = 0.0
        return x, y

    @staticmethod
    def uv1976_to_xy(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        denom = 6 * u - 16 * v + 12
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (9 * u) / denom
            y = (4 * v) / denom
        x[~np.isfinite(x)] = 0.0
        y[~np.isfinite(y)] = 0.0
        return x, y

# --- Data Handler ---

class CIEDataHandler:
    """
    Parses CIE JSONs and manages data retrieval based on Source, Observer, and System.
    """
    
    # Configuration map to handle the different JSON key naming conventions
    KEY_MAPPINGS = {
        (CIEDataSource.ConeFundamental, CIEObserver.Degree2):  ('xbar_F(lambda)', 'ybar_F(lambda)', 'zbar_F(lambda)'),
        (CIEDataSource.ConeFundamental, CIEObserver.Degree10): ('xbar_F,10(lambda)', 'ybar_F,10(lambda)', 'zbar_F,10(lambda)'),
        (CIEDataSource.StandardXYZ, CIEObserver.Degree2):      ('x_bar(lambda)', 'y_bar(lambda)', 'z_bar(lambda)'),
        (CIEDataSource.StandardXYZ, CIEObserver.Degree10):     ('xbar_10(lambda)', 'ybar_10(lambda)', 'zbar_10(lambda)'),
    }

    def __init__(self, file_paths: Dict[Tuple[CIEDataSource, CIEObserver], str]):
        """
        Args:
            file_paths: Dictionary mapping (Source, Observer) tuples to file paths.
        """
        self.paths = file_paths
        self.cache: Dict[Tuple[CIEDataSource, CIEObserver], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {} 

    def get_data(self, source: CIEDataSource, observer: CIEObserver, system: CIESystem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves and transforms data for the requested configuration.
        """
        cache_key = (source, observer)
        
        # 1. Load Raw XYZ Data (if not cached)
        if cache_key not in self.cache:
            success = self._load_data(source, observer)
            if not success:
                return np.array([]), np.array([]), np.array([])
            
        wl, x, y = self.cache[cache_key]
        
        # 2. Transform to requested Coordinate System
        if system == CIESystem.CIE1931:
            return wl, x, y
        elif system == CIESystem.CIE1960:
            u, v = CIETransform.xy_to_uv1960(x, y)
            return wl, u, v
        elif system == CIESystem.CIE1976:
            u, v = CIETransform.xy_to_uv1976(x, y)
            return wl, u, v
            
        return wl, x, y

    def _load_data(self, source: CIEDataSource, observer: CIEObserver) -> bool:
        """Loads JSON data and normalizes XYZ to xy."""
        key = (source, observer)
        path = self.paths.get(key)
        
        if not path:
            print(f"No file path defined for {source.name} - {observer.name}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            data_block = content.get('data', content)
            
            # Retrieve the specific keys for this dataset
            keys_x, keys_y, keys_z = self.KEY_MAPPINGS[key]
            
            # Extract Wavelengths and Tristimulus Values
            wl = np.array(data_block['lambda']['values'], dtype=np.float64)
            X = np.array(data_block[keys_x]['values'], dtype=np.float64)
            Y = np.array(data_block[keys_y]['values'], dtype=np.float64)
            Z = np.array(data_block[keys_z]['values'], dtype=np.float64)
            
            # Calculate Chromaticity Coordinates (xy)
            sum_xyz = X + Y + Z
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                x_norm = X / sum_xyz
                y_norm = Y / sum_xyz
            
            x_norm[~np.isfinite(x_norm)] = 0.0
            y_norm[~np.isfinite(y_norm)] = 0.0
            
            self.cache[key] = (wl, x_norm, y_norm)
            return True
            
        except FileNotFoundError:
            print(f"File not found: {path}")
            return False
        except KeyError as e:
            print(f"JSON Key Error in {path}: Missing key {e}")
            return False
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return False

# --- Main GUI ---

class CIEWidget(QWidget):
    def __init__(self, data_handler: CIEDataHandler, parent=None):
        super().__init__(parent)
        self.data_handler = data_handler
        self.init_ui()
        self.refresh_diagram()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(5)
        
        # --- Control Panel ---
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # 1. Data Source Selection
        self.combo_source = QComboBox()
        self.combo_source.addItems([e.name for e in CIEDataSource])
        self.combo_source.currentIndexChanged.connect(self.refresh_diagram)
        
        # 2. Observer Selection
        self.combo_obs = QComboBox()
        self.combo_obs.addItems([e.name for e in CIEObserver])
        self.combo_obs.currentIndexChanged.connect(self.refresh_diagram)
        
        # 3. Coordinate System Selection
        self.combo_sys = QComboBox()
        self.combo_sys.addItems([e.name for e in CIESystem])
        self.combo_sys.currentIndexChanged.connect(self.refresh_diagram)
        
        control_layout.addWidget(QLabel("<b>Data Source:</b>"))
        control_layout.addWidget(self.combo_source)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("<b>Observer:</b>"))
        control_layout.addWidget(self.combo_obs)
        control_layout.addSpacing(15)
        control_layout.addWidget(QLabel("<b>System:</b>"))
        control_layout.addWidget(self.combo_sys)
        control_layout.addStretch()
        
        self.layout.addWidget(control_frame)

        # --- Plot Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getPlotItem().setMouseEnabled(x=False, y=False) # Optional: Lock zoom for stability
        
        # Styling Axes
        pen = pg.mkPen('k', width=1)
        for axis in ['bottom', 'left']:
            self.plot_widget.getAxis(axis).setPen(pen)
            self.plot_widget.getAxis(axis).setTextPen('k')

        self.layout.addWidget(self.plot_widget)

    def refresh_diagram(self):
        self.plot_widget.clear()
        
        # Get Enum values from ComboBox text
        source_enum = CIEDataSource[self.combo_source.currentText()]
        obs_enum = CIEObserver[self.combo_obs.currentText()]
        sys_enum = CIESystem[self.combo_sys.currentText()]
        
        # 1. Fetch Data
        wl, x_locus, y_locus = self.data_handler.get_data(source_enum, obs_enum, sys_enum)
        
        if len(x_locus) == 0:
            # Handle empty data gracefully
            return 

        # 2. Setup Bounds and Labels
        if sys_enum == CIESystem.CIE1931:
            bounds = (0.0, 0.8, 0.0, 0.9)
            self.plot_widget.setLabel('bottom', "x")
            self.plot_widget.setLabel('left', "y")
        else:
            bounds = (0.0, 0.65, 0.0, 0.65)
            self.plot_widget.setLabel('bottom', "u'" if sys_enum == CIESystem.CIE1976 else "u")
            self.plot_widget.setLabel('left', "v'" if sys_enum == CIESystem.CIE1976 else "v")
            
        self.plot_widget.setRange(xRange=bounds[0:2], yRange=bounds[2:4])

        # 3. Draw Gradient Background (Full Rect)
        # Optimization: Only regenerate gradient if resolution/bounds change (not implemented here for simplicity)
        img_rgba, rect = self._generate_gradient(sys_enum, bounds)
        img_item = pg.ImageItem(img_rgba)
        img_item.setRect(QRectF(rect[0], rect[2], rect[1]-rect[0], rect[3]-rect[2]))
        img_item.setZValue(-20)
        self.plot_widget.addItem(img_item)

        # 4. Apply Vector Mask (The "Cookie Cutter")
        self._apply_vector_mask(x_locus, y_locus, bounds)

        # 5. Draw Spectral Locus Outline
        # Close the loop visually
        x_closed = np.append(x_locus, x_locus[0])
        y_closed = np.append(y_locus, y_locus[0])
        self.plot_widget.addItem(pg.PlotCurveItem(x_closed, y_closed, pen=pg.mkPen('k', width=2), antialias=True))
        
        # 6. Add Wavelength Labels
        self._add_labels(wl, x_locus, y_locus)

    def _generate_gradient(self, system: CIESystem, bounds: Tuple[float, float, float, float], res: int = 400) -> Tuple[np.ndarray, Tuple]:
        """Generates the colored spectral background."""
        x_min, x_max, y_min, y_max = bounds
        u = np.linspace(x_min, x_max, res)
        v = np.linspace(y_min, y_max, res)
        uu, vv = np.meshgrid(u, v)
        
        # Flatten for vectorized calculation
        uf, vf = uu.ravel(), vv.ravel()
        
        # Inverse Map to xy for Color Calculation
        if system == CIESystem.CIE1931:
            xf, yf = uf, vf
        elif system == CIESystem.CIE1960:
            xf, yf = CIETransform.uv1960_to_xy(uf, vf)
        elif system == CIESystem.CIE1976:
            xf, yf = CIETransform.uv1976_to_xy(uf, vf)

        # Calculate XYZ from xy (assuming Y=1.0 for maximum brightness/color)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y = np.ones_like(yf)
            X = (xf / yf) * Y
            Z = ((1.0 - xf - yf) / yf) * Y
            XYZ = np.vstack((X, Y, Z)).T
            
            # Clean invalid values
            XYZ[~np.isfinite(XYZ)] = 0.0
            XYZ[XYZ < 0] = 0.0

        # Convert to sRGB
        rgb_linear = XYZ @ M_XYZ_TO_SRGB.T
        
        # Normalize by max component to keep colors bright/visible
        # (This differs from strict colorimetry but is standard for diagrams)
        max_val = np.max(rgb_linear, axis=1, keepdims=True)
        max_val[max_val == 0] = 1.0 # Avoid div by zero
        rgb_normalized = np.clip(rgb_linear / max_val, 0, 1)
        
        rgb_gamma = linear_to_srgb(rgb_normalized)
        
        # Reshape to image format (H, W, 3)
        img = (rgb_gamma.reshape(res, res, 3) * 255).astype(np.uint8)
        
        # Add Alpha Channel (Opaque)
        alpha = np.full((res, res), 255, dtype=np.uint8)
        
        # PyQtGraph expects (W, H, 4) transpose
        return np.transpose(np.dstack((img, alpha)), (1, 0, 2)), bounds

    def _apply_vector_mask(self, x_locus: np.ndarray, y_locus: np.ndarray, bounds: Tuple[float, float, float, float]):
        """Creates a white mask outside the spectral locus using QPainterPath."""
        # 1. Path for the Spectral Locus (Inner Shape)
        locus_path = QPainterPath()
        if len(x_locus) > 0:
            locus_path.moveTo(x_locus[0], y_locus[0])
            for x, y in zip(x_locus[1:], y_locus[1:]):
                locus_path.lineTo(x, y)
        locus_path.closeSubpath()
        
        # 2. Path for the Outer Bounding Box
        outer = QPainterPath()
        bx_min, bx_max, by_min, by_max = bounds
        margin = 0.5
        outer.addRect(QRectF(bx_min - margin, by_min - margin, 
                             (bx_max - bx_min) + 2*margin, (by_max - by_min) + 2*margin))
        
        # 3. Subtract Locus from Outer Box -> Hole
        mask_path = outer.subtracted(locus_path)
        
        mask_item = pg.QtWidgets.QGraphicsPathItem(mask_path)
        mask_item.setBrush(QBrush(Qt.white))
        mask_item.setPen(QPen(Qt.NoPen))
        mask_item.setZValue(-10) # Place between background and grid
        self.plot_widget.addItem(mask_item)

    def _add_labels(self, wl: np.ndarray, x_locus: np.ndarray, y_locus: np.ndarray):
        """Adds tick marks and wavelength numbers along the locus."""
        if wl is None or len(wl) == 0: return
        
        targets = [450, 480, 500, 520, 540, 560, 580, 600, 620, 700]
        
        # Center of locus for vector calculation
        cx, cy = np.mean(x_locus), np.mean(y_locus)
        
        for t in targets:
            # Find closest index
            idx = (np.abs(wl - t)).argmin()
            x, y = x_locus[idx], y_locus[idx]
            
            # Calculate normal vector pointing outwards
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx, dy)
            dx, dy = dx/dist, dy/dist
            
            # Tick mark
            tick_len = 0.015
            self.plot_widget.addItem(pg.PlotCurveItem(
                [x, x + dx*tick_len], 
                [y, y + dy*tick_len], 
                pen=pg.mkPen('k', width=1.5)
            ))
            
            # Text label
            txt = pg.TextItem(str(t), color='k', anchor=(0.5, 0.5))
            # Offset text slightly further out
            txt.setPos(x + dx*0.045, y + dy*0.045)
            self.plot_widget.addItem(txt)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Define file paths for all supported configurations
    files = {
        (CIEDataSource.ConeFundamental, CIEObserver.Degree2):  "CIE_cfb_stv_2deg.json",
        (CIEDataSource.ConeFundamental, CIEObserver.Degree10): "CIE_cfb_stv_10deg.json",
        (CIEDataSource.StandardXYZ, CIEObserver.Degree2):      "CIE_xyz_1931_2deg.json",
        (CIEDataSource.StandardXYZ, CIEObserver.Degree10):     "CIE_xyz_1964_10deg.json",
    }
    
    handler = CIEDataHandler(files)
    
    window = CIEWidget(handler)
    window.resize(1000, 900)
    window.setWindowTitle("Advanced CIE Color Diagram Explorer")
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()