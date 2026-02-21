import sys
import json
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from matplotlib.path import Path

# --- Constants for Color Science ---

# XYZ to sRGB Matrix (Standard D65 Illuminant)
M_XYZ_TO_SRGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
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

class CIEDataHandler:
    """Handles loading and processing of CIE JSON Data."""
    
    def __init__(self, json_path: str):
        self.wavelengths = None
        self.x_bar = None
        self.y_bar = None
        self.z_bar = None
        self.locus_x = None
        self.locus_y = None
        
        self._load_data(json_path)
        self._compute_spectral_locus()

    def _load_data(self, path: str):
        try:
            with open(path, 'r') as f:
                content = json.load(f)
            
            data_block = content['data']
            self.wavelengths = np.array(data_block['lambda']['values'], dtype=np.float64)
            self.x_bar = np.array(data_block['x_bar(lambda)']['values'], dtype=np.float64)
            self.y_bar = np.array(data_block['y_bar(lambda)']['values'], dtype=np.float64)
            self.z_bar = np.array(data_block['z_bar(lambda)']['values'], dtype=np.float64)
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load '{path}' ({e}). Using synthetic fallback data.")
            # Synthetic fallback for demonstration
            self.wavelengths = np.linspace(380, 780, 100)
            t = np.linspace(0, 1, 100)
            self.x_bar = np.exp(-((t-0.2)**2)*20) + 0.3*np.exp(-((t-0.8)**2)*20)
            self.y_bar = np.exp(-((t-0.5)**2)*20)
            self.z_bar = np.exp(-((t-0.8)**2)*20) + 0.1*np.exp(-((t-0.2)**2)*20)

    def _compute_spectral_locus(self):
        sum_xyz = self.x_bar + self.y_bar + self.z_bar
        with np.errstate(divide='ignore', invalid='ignore'):
            self.locus_x = np.divide(self.x_bar, sum_xyz)
            self.locus_y = np.divide(self.y_bar, sum_xyz)
            
        valid = sum_xyz > 1e-6
        self.locus_x = self.locus_x[valid]
        self.locus_y = self.locus_y[valid]
        self.wavelengths = self.wavelengths[valid]

class CIE1931Widget(QWidget):
    def __init__(self, json_path: str, parent=None):
        super().__init__(parent)
        self.data = CIEDataHandler(json_path)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', "CIE x")
        self.plot_widget.setLabel('left', "CIE y")
        self.plot_widget.setRange(xRange=(-0.1, 0.9), yRange=(-0.1, 1.0))
        
        self.layout.addWidget(self.plot_widget)
        self._render_diagram()

    def _render_diagram(self):
        # 1. Generate Background
        img_rgba, bounds = self._generate_chromaticity_grid(resolution=512)
        
        image_item = pg.ImageItem(img_rgba)
        # Map the image rect to the coordinate system
        image_item.setRect(pg.QtCore.QRectF(bounds[0], bounds[2], 
                                            bounds[1]-bounds[0], 
                                            bounds[3]-bounds[2]))
        image_item.setZValue(-10)
        self.plot_widget.addItem(image_item)

        # 2. Draw Spectral Locus Line
        locus_x_closed = np.append(self.data.locus_x, self.data.locus_x[0])
        locus_y_closed = np.append(self.data.locus_y, self.data.locus_y[0])
        
        locus_curve = pg.PlotCurveItem(
            locus_x_closed, locus_y_closed, 
            pen=pg.mkPen(color='k', width=2),
            antialias=True
        )
        self.plot_widget.addItem(locus_curve)
        
        # 3. Add Labels
        labels_to_show = [450, 480, 500, 520, 540, 560, 580, 600, 620]
        for wl in labels_to_show:
            idx = (np.abs(self.data.wavelengths - wl)).argmin()
            x, y = self.data.locus_x[idx], self.data.locus_y[idx]
            
            dx, dy = x - 0.33, y - 0.33
            dist = np.hypot(dx, dy)
            dx, dy = dx/dist, dy/dist
            
            text = pg.TextItem(text=str(wl), color='k', anchor=(0.5, 0.5))
            text.setPos(x + dx * 0.04, y + dy * 0.04)
            font = text.textItem.font()
            font.setPointSize(8)
            text.setFont(font)
            self.plot_widget.addItem(text)
            
            self.plot_widget.addItem(pg.PlotCurveItem(
                [x, x + dx*0.015], [y, y + dy*0.015],
                pen=pg.mkPen('k', width=1)
            ))

    def _generate_chromaticity_grid(self, resolution: int = 512):
        """Generates the RGB texture, ensuring correct Y-axis orientation."""
        x_min, x_max = 0.0, 0.8
        y_min, y_max = 0.0, 0.9
        
        # 1. Create Grid
        # Meshgrid by default produces arrays where:
        # xx increases with column index
        # yy increases with row index (0 -> min, -1 -> max)
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        
        # FIX: Removed np.flipud(yy).
        # We want index 0 to correspond to y_min (bottom of plot)
        # and index -1 to correspond to y_max (top of plot).
        
        # Flatten
        xf = xx.ravel()
        yf = yy.ravel()
        
        # 2. xy -> XYZ (assuming Y=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Y_lum = np.ones_like(yf)
            X_lum = (xf / yf) * Y_lum
            Z_lum = ((1.0 - xf - yf) / yf) * Y_lum
            
            XYZ = np.vstack((X_lum, Y_lum, Z_lum)).T
            XYZ[~np.isfinite(XYZ)] = 0.0
            XYZ[XYZ < 0] = 0.0

        # 3. XYZ -> sRGB
        RGB_linear = XYZ @ M_XYZ_TO_SRGB.T
        
        # Gamut mapping (normalization)
        max_val = np.max(RGB_linear, axis=1, keepdims=True)
        max_val[max_val < 1.0] = 1.0
        RGB_linear = RGB_linear / max_val
        
        RGB_linear = np.clip(RGB_linear, 0, 1)
        RGB_srgb = linear_to_srgb(RGB_linear)
        
        # 4. Masking
        locus_points = np.column_stack((self.data.locus_x, self.data.locus_y))
        locus_poly = np.vstack([locus_points, [locus_points[0]]])
        path = Path(locus_poly)
        
        grid_points = np.column_stack((xf, yf))
        mask = path.contains_points(grid_points)
        
        # 5. Format for PyQtGraph
        # Reshape to (rows, cols, 3) -> (y, x, 3) effectively
        img_rgb = (RGB_srgb.reshape(resolution, resolution, 3) * 255).astype(np.uint8)
        mask = mask.reshape(resolution, resolution)
        
        alpha = np.zeros((resolution, resolution), dtype=np.uint8)
        alpha[mask] = 255
        
        img_rgba = np.dstack((img_rgb, alpha))
        
        # Transpose to (cols, rows, rgba) -> (x, y, rgba)
        # PyQtGraph expects x-axis to be the first dimension
        img_rgba = np.transpose(img_rgba, (1, 0, 2))
        
        return img_rgba, (x_min, x_max, y_min, y_max)

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Ensure this file is in the working directory
    json_filename = "CIE_xyz_1931_2deg.json"
    
    window = CIE1931Widget(json_filename)
    window.resize(900, 900)
    window.setWindowTitle("CIE 1931 Diagram (Y-Flip Corrected)")
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()