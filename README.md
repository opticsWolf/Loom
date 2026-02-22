<img width="280" height="280" alt="loom_logo" src="https://github.com/user-attachments/assets/e08705cf-d221-4aeb-bfe2-56ef9f23bf17" />

# Loom: 
## Weaving the mathematics of light in thin film systems
**Loom** is a high-performance, physically rigorous 1D optical engine designed for the simulation of light propagation in stratified media. Built on a modern **Scattering Matrix (S-matrix)** architecture, it offers a numerically stable and vectorized alternative to traditional Transfer Matrix Methods (TMM).

### 1. Unconditional Numerical Stability

Traditional TMM suffers from "numerical swamping" (exponentially growing evanescent waves) when dealing with thick layers or highly absorbing materials (like metals in the UV/X-ray range). Loom utilizes the **Redheffer Star Product** to propagate scattering matrices, ensuring that all matrix elements remain bounded and physically meaningful, regardless of layer thickness.

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
