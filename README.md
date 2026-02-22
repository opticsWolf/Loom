<img width="280" height="280" alt="loom_logo" src="https://github.com/user-attachments/assets/e08705cf-d221-4aeb-bfe2-56ef9f23bf17" />

# Loom - Weaving the mathematics of light in thin film systems

**Loom** is a high-performance, physically rigorous 1D optical engine designed for the simulation of light propagation in stratified media. Built on a modern **Scattering Matrix (S-matrix)** architecture, it offers a numerically stable and vectorized alternative to traditional Transfer Matrix Methods (TMM).

### 1. Unconditional Numerical Stability

Traditional TMM suffers from numerical divergence (exponentially growing evanescent waves) when dealing with thick layers or highly absorbing materials. Loom utilizes the **Redheffer Star Product** to propagate scattering matrices, ensuring that all matrix elements remain bounded and physically meaningful, regardless of layer thickness.

### 2. High-Concurrency Performance

As a Principal Performance Engineer, you need tools that scale. Loom is built for speed:

- **Parallel Execution**: Utilizes Numba’s `@njit(parallel=True)` to saturate all available CPU cores.
    
- **Vectorized Engine**: Operations are performed across the entire (wavelength × angle) coordinate space in a single pass, eliminating Python's loop overhead.
    
- **Memory Efficiency**: Collapses multi-layer stacks into a compact global S-matrix to minimize cache misses.
    

### 3. Partial Coherence Support

Real-world systems often involve thick substrates (like a 1mm glass slide) where phase information is lost. Loom features a **Hybrid Coherence Engine**:

- **Coherent Blocks**: Preserves phase for thin-film interference.
    
- **Incoherent Interfaces**: Switches to intensity-based propagation for thick layers, preventing the "unphysical ringing" caused by assuming perfect coherence across a macroscopic substrate.
    

### 4. Advanced Physics Modeling

Loom goes beyond simple Fresnel equations to provide research-grade accuracy:

- **Interface Roughness**: Implements the **Névot-Croce** model, providing superior accuracy for high-frequency or X-ray reflectometry compared to standard Gaussian approximations.
    
- **Ellipsometric Rigor**: Outputs (Ψ,Δ) parameters that strictly follow the **Azzam & Bashara** convention, ensuring direct compatibility with commercial ellipsometers (e.g., Woollam, Horiba).
### Technical Specifications

|**Feature**|**Implementation & Engineering Benefit**|
|---|---|
|**Core Algorithm**|**1D Scattering Matrix ($S$-matrix)**: Utilizes the Redheffer Star Product to eliminate numerical divergence and precision loss in thick or highly absorbing layers.|
|**Propagation Logic**|**Hybrid Mixed Coherence**: Sophisticated dual-stage engine supporting phase-accurate (coherent) and intensity-only (incoherent) layers within a single pass.|
|**Coherent Blocks**|**$2 \times 2$ Complex Field Matrices**: Maintains full phase and amplitude information, ensuring rigorous calculation of thin-film interference and ellipsometric parameters.|
|**Incoherent Blocks**|**Stokes-Mueller / Intensity Redheffer**: Prevents unphysical interference artifacts in macroscopic substrates by utilizing intensity-based propagation.|
|**Roughness Model**|**Névot-Croce (Exact Wavevector)**: Achieves research-grade accuracy for X-ray and UV interfaces by modeling exact wavevector correlations across boundaries.|
|**Optimization**|**Numba-JIT / Parallelized `prange`**: Delivers near-native C performance with Pythonic flexibility, optimized for high-concurrency simulation and real-time GUI responsiveness.|
|**Polarization**|**Full $s$ and $p$ Support**: Comprehensive Jones and Stokes calculus integration, following standard commercial ellipsometry conventions (Azzam & Bashara).|
|**Complexity**|**$O(N)$ Scaling**: Optimized linear time complexity relative to the number of layers, ensuring stable performance for complex multi-stack architectures.|
