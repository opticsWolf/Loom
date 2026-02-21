# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 20:55:02 2025

@author: Frank
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from typing import Tuple, Dict, Any

def load_cie_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CIE 1931 data from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Tuple containing wavelengths, x_bar, y_bar, z_bar as numpy arrays.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Navigate the specific JSON structure provided
    raw_data = data['data']
    wavelengths = np.array(raw_data['lambda']['values'])
    x_bar = np.array(raw_data['x_bar(lambda)']['values'])
    y_bar = np.array(raw_data['y_bar(lambda)']['values'])
    z_bar = np.array(raw_data['z_bar(lambda)']['values'])
    
    return wavelengths, x_bar, y_bar, z_bar

def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """
    Converts XYZ coordinates to sRGB (D65 standard).
    
    Args:
        xyz: Shape (..., 3) array of XYZ values.
        
    Returns:
        Shape (..., 3) array of sRGB values (clipped to [0, 1]).
    """
    # Standard sRGB D65 transformation matrix
    # [ 3.2404542 -1.5371385 -0.4985314]
    # [-0.9692660  1.8760108  0.0415560]
    # [ 0.0556434 -0.2040259  1.0572252]
    matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    # Linear RGB
    rgb_linear = np.tensordot(xyz, matrix, axes=([len(xyz.shape)-1], [1]))
    
    # Gamma correction for sRGB
    mask = rgb_linear > 0.0031308
    rgb_srgb = np.zeros_like(rgb_linear)
    rgb_srgb[mask] = 1.055 * np.power(rgb_linear[mask], 1.0/2.4) - 0.055
    rgb_srgb[~mask] = 12.92 * rgb_linear[~mask]
    
    return np.clip(rgb_srgb, 0, 1)

def plot_cie_chromaticity(
    wavelengths: np.ndarray, 
    x_bar: np.ndarray, 
    y_bar: np.ndarray, 
    z_bar: np.ndarray
) -> None:
    """
    Generates and saves the CIE 1931 Chromaticity Diagram.
    """
    # 1. Calculate Chromaticity Coordinates (spectral locus)
    denominator = x_bar + y_bar + z_bar
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        x_locus = np.divide(x_bar, denominator, where=denominator!=0)
        y_locus = np.divide(y_bar, denominator, where=denominator!=0)
    
    # 2. Generate Grid for the Colored Background
    resolution = 1000
    x_grid = np.linspace(0, 0.8, resolution)
    y_grid = np.linspace(0, 0.9, resolution)
    xv, yv = np.meshgrid(x_grid, y_grid)
    
    # 3. Convert Grid (x, y) -> XYZ -> RGB
    # We assume Y (Luminance) = 1 for brightness, then calculate X and Z
    # X = (x/y) * Y
    # Z = ((1-x-y)/y) * Y
    # Note: This map shows the color IF that x,y existed with full luminance.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Y_grid = np.ones_like(xv)
        X_grid = (xv / yv) * Y_grid
        Z_grid = ((1 - xv - yv) / yv) * Y_grid
        
    # Stack into shape (H, W, 3)
    xyz_grid = np.dstack((X_grid, Y_grid, Z_grid))
    
    # Handle NaNs/Infs from y=0
    xyz_grid[~np.isfinite(xyz_grid)] = 0
    
    rgb_grid = xyz_to_srgb(xyz_grid)

    # 4. Masking (Point in Polygon)
    # Define the boundary polygon: Spectral locus + connect ends
    locus_points = np.column_stack((x_locus, y_locus))
    # Close the loop (Purple line)
    locus_points = np.vstack((locus_points, locus_points[0]))
    
    path = Path(locus_points)
    
    # Create flat points array for checking
    points_flat = np.column_stack((xv.ravel(), yv.ravel()))
    mask_flat = path.contains_points(points_flat)
    mask = mask_flat.reshape(xv.shape)
    
    # Apply mask (set outside to white or black)
    # Using alpha channel for transparency is cleaner
    rgba_grid = np.dstack((rgb_grid, mask.astype(float)))

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    
    # Plot the colored image
    # Note: imshow origin is top-left, we need 'lower'
    ax.imshow(
        rgba_grid, 
        extent=[0, 0.8, 0, 0.9], 
        origin='lower', 
        interpolation='bilinear'
    )
    
    # Plot the spectral locus line
    ax.plot(x_locus, y_locus, color='black', linewidth=1.5, label='Spectral Locus')
    
    # Plot the purple line (connect start and end)
    ax.plot(
        [x_locus[0], x_locus[-1]], 
        [y_locus[0], y_locus[-1]], 
        color='purple', 
        linestyle='--', 
        linewidth=1.2, 
        label='Line of Purples'
    )

    # Annotate Wavelengths
    # Filter wavelengths to plot text sparsely to avoid clutter
    target_lambdas = [380, 450, 480, 500, 520, 540, 560, 580, 600, 620, 700]
    
    for wl in target_lambdas:
        # Find closest index
        idx = (np.abs(wavelengths - wl)).argmin()
        x_pt, y_pt = x_locus[idx], y_locus[idx]
        
        ax.plot(x_pt, y_pt, 'ko', markersize=3)
        
        # Simple heuristic for label offset
        offset_x, offset_y = 0.0, 0.0
        if y_pt > 0.8: offset_y = 0.02
        elif x_pt < 0.2: offset_x = -0.04
        elif x_pt > 0.6: offset_x = 0.01; offset_y = 0.01
        else: offset_x = 0.01; offset_y = 0.01
        
        # Adjust specific positions for readability
        if wl == 520: offset_y = 0.02; offset_x = 0
        if wl == 380: offset_x = 0.01; offset_y = -0.01
        
        ax.text(
            x_pt + offset_x, 
            y_pt + offset_y, 
            f'{wl}nm', 
            fontsize=9, 
            color='black',
            fontweight='bold'
        )

    ax.set_title('CIE 1931 Chromaticity Diagram', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.05, 0.95)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('cie_1931_chromaticity.png')
    # plt.show() is handled by the environment or omitted per instructions

if __name__ == "__main__":
    try:
        wl, x, y, z = load_cie_data('CIE_xyz_1931_2deg.json')
        plot_cie_chromaticity(wl, x, y, z)
        print("CIE graph generated successfully: cie_1931_chromaticity.png")
    except FileNotFoundError:
        print("Error: JSON file not found. Please ensure CIE_xyz_1931_2deg.json is in the working directory.")