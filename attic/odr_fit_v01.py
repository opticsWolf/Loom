# -*- coding: utf-8 -*-
"""
Loom: Weaving the mathematics of light in thin film systems
Copyright (c) 2026 opticsWolf

SPDX-License-Identifier: LGPL-3.0-or-later
"""

import numpy as np
from numba import njit, prange
# Assuming core_engine is available from your previous code
# from your_module import core_engine, POL_S, POL_P 

@njit(parallel=True)
def calculate_spectral_jacobian(beta, wavls, theta_rad, n_layers, base_indices, 
                                base_thicknesses, incoherent_flags, 
                                rough_types, rough_vals, 
                                param_mapping_indices, param_mapping_thick,
                                perturbation=1e-8):
    """
    Calculates the Jacobian of the Reflectance spectrum with respect to parameters beta.
    
    Args:
        beta: The optimization parameters.
        perturbation: Step size for finite difference.
        param_mapping_*: Arrays defining which parameter in 'beta' controls which layer property.
                         -1 implies the layer property is fixed/constant.
    
    Returns:
        J_matrix: (N_wavs, N_params) array.
    """
    num_wavs = len(wavls)
    num_params = len(beta)
    
    # 1. Calculate Baseline Spectrum (Unperturbed)
    # Reconstruct arrays based on current beta
    # NOTE: We operate on copies to avoid modifying shared memory in parallel loops
    
    # We need a helper to reconstruct (inlined logic for speed)
    # This part reconstructs the full inputs for core_engine
    curr_indices = base_indices.copy()
    curr_thick = base_thicknesses.copy()
    
    # Update properties based on beta
    for p_idx in range(num_params):
        val = beta[p_idx]
        
        # Update Thicknesses
        for i in range(n_layers):
            if param_mapping_thick[i] == p_idx:
                curr_thick[i] = val
        
        # Update Indices (Real part only for this simple example)
        for i in range(n_layers):
            if param_mapping_indices[i] == p_idx:
                # Assuming non-dispersive optimization for simplicity here
                # For dispersion, you would call a dispersion func here
                curr_indices[i, :] = val + 0j 

    # Run Baseline
    # We only care about Unpolarized (avg) for this example, but you can choose S or P
    Rs, Rp, _, _ = core_engine(wavls, theta_rad, n_layers, curr_indices, curr_thick, 
                               incoherent_flags, rough_types, rough_vals, True, True)
    R_base = (Rs + Rp) / 2.0
    
    # 2. Calculate Jacobian Columns (Parallelized over Parameters)
    J = np.empty((num_wavs, num_params), dtype=np.float64)
    
    for p in prange(num_params):
        # Create local copies for perturbation
        p_indices = base_indices.copy()
        p_thick = base_thicknesses.copy()
        
        # Perturb one parameter
        beta_perturbed_val = beta[p] + perturbation
        
        # Apply perturbation to mapped layers
        # (Repeat the mapping logic for the perturbed value)
        # 1. Update perturbed parameter in the full set
        for i in range(n_layers):
            if param_mapping_thick[i] == p:
                p_thick[i] = beta_perturbed_val
            if param_mapping_indices[i] == p:
                p_indices[i, :] = beta_perturbed_val + 0j
        
        # Run Perturbed
        Rs_p, Rp_p, _, _ = core_engine(wavls, theta_rad, n_layers, p_indices, p_thick, 
                                       incoherent_flags, rough_types, rough_vals, True, True)
        R_perturbed = (Rs_p + Rp_p) / 2.0
        
        # Finite Difference
        for w in range(num_wavs):
            J[w, p] = (R_perturbed[w] - R_base[w]) / perturbation
            
    return J, R_base

import numpy as np
from scipy.odr import ODR, Model, Data

class ODR_TMM_Fitter:
    def __init__(self, wavls, R_meas, theta_deg, structure_definition, constraints):
        """
        Args:
            wavls: Wavelength array (nm).
            R_meas: Measured Reflectance array (0-1).
            structure_definition: Dict containing base structure and mapping logic.
            constraints: List of dicts {'param_idx': 0, 'min': 10, 'max': 100, 'weight': 1e4}
        """
        self.wavls = np.ascontiguousarray(wavls)
        self.R_meas = np.ascontiguousarray(R_meas)
        self.theta_rad = np.radians(theta_deg)
        self.constraints = constraints
        
        # Unpack structure definition
        self.n_layers = structure_definition['n_layers']
        self.base_indices = structure_definition['base_indices']
        self.base_thick = structure_definition['base_thick']
        self.inc_flags = structure_definition['inc_flags']
        self.r_types = structure_definition['r_types']
        self.r_vals = structure_definition['r_vals']
        
        # Mapping arrays: -1 means fixed, 0+ is index in beta vector
        self.map_thick = structure_definition['map_thick']
        self.map_ind = structure_definition['map_ind']

    def _apply_constraints(self, beta):
        """Calculates penalty residuals and their Jacobian rows."""
        penalty_res = []
        penalty_jac_rows = []
        
        for c in self.constraints:
            idx = c['param_idx']
            w_sqrt = np.sqrt(c['weight'])
            val = beta[idx]
            
            # Row for Jacobian (initially zero)
            j_row = np.zeros(len(beta))
            
            p_val = 0.0
            
            # Check Min
            if 'min' in c and val < c['min']:
                diff = c['min'] - val
                p_val += w_sqrt * diff
                j_row[idx] -= w_sqrt # Derivative of (Min - x) is -1
            
            # Check Max
            elif 'max' in c and val > c['max']:
                diff = val - c['max']
                p_val += w_sqrt * diff
                j_row[idx] += w_sqrt # Derivative of (x - Max) is +1
                
            penalty_res.append(p_val)
            penalty_jac_rows.append(j_row)
            
        return np.array(penalty_res), np.array(penalty_jac_rows)

    def fcn_odr(self, beta, x):
        """
        The Model Function.
        Returns concatenated vector: [Spectral_Residuals, Penalty_Residuals]
        
        Note: x is unused here because wavls are stored in class, 
        but ODR requires the signature f(beta, x).
        """
        # 1. Calculate Spectrum (via Numba wrapper)
        # We reuse the Jacobian function because it computes the baseline R_base essentially for free
        # But for cleaner logic, we can just run the forward model logic here.
        # However, to save code duplication, let's call the Jacobian func which returns (J, R)
        
        # Note: In a production environment, split these to save the Jacobian calculation cost 
        # when ODR only requests the function value.
        _, R_calc = calculate_spectral_jacobian(
            beta, self.wavls, self.theta_rad, self.n_layers, 
            self.base_indices, self.base_thick, self.inc_flags, 
            self.r_types, self.r_vals, self.map_ind, self.map_thick
        )
        
        # 2. Calculate Penalties
        P_vals, _ = self._apply_constraints(beta)
        
        # 3. Formulate Output
        # ODR expects Y_est. We are tricking it.
        # We want to minimize: sum((R_calc - R_meas)^2) + sum(P^2)
        # We tell ODR our "Y_data" is [R_meas, 0, 0...]
        # So our Function must return [R_calc, P_vals]
        
        return np.hstack((R_calc, P_vals))

    def jac_odr(self, beta, x):
        """
        The Jacobian Function.
        Returns stacked matrix: [Spectral_Jacobian; Penalty_Jacobian]
        """
        # 1. Spectral Jacobian (N_wavs x N_params)
        J_spec, _ = calculate_spectral_jacobian(
            beta, self.wavls, self.theta_rad, self.n_layers, 
            self.base_indices, self.base_thick, self.inc_flags, 
            self.r_types, self.r_vals, self.map_ind, self.map_thick
        )
        
        # 2. Penalty Jacobian (N_constraints x N_params)
        _, J_pen = self._apply_constraints(beta)
        
        if len(J_pen) > 0:
            return np.vstack((J_spec, J_pen))
        return J_spec

    def run_fit(self, beta0):
        # 1. Construct Augmented Data
        # Y data = Measured Spectrum + Zeros for penalties
        num_constraints = len(self.constraints)
        y_augmented = np.hstack((self.R_meas, np.zeros(num_constraints)))
        
        # X data = Wavelengths + Dummy zeros (lengths must match Y)
        # (ODR requires X and Y to be same length usually, but for explicit models we can cheat slightly
        # depending on implementation, but safest is to pad X too)
        x_augmented = np.zeros_like(y_augmented)
        x_augmented[:len(self.wavls)] = self.wavls
        
        # 2. Setup ODR
        # We use RealData. sx and sy (weights) can be added here.
        data = Data(x_augmented, y_augmented)
        
        model = Model(self.fcn_odr, fjac=self.jac_odr)
        
        my_odr = ODR(data, model, beta0=beta0)
        my_odr.set_job(fit_type=2, deriv=1) # 2 = Least Squares (Explicit ODR), 1=User Jacobian
        
        output = my_odr.run()
        return output
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # --- 1. Generate Mock Data ---
    # Let's say we have 3 layers: Air / Thin Film / Substrate
    # We want to fit the Thickness of Layer 1 and Index of Layer 1.
    wavls = np.linspace(400, 800, 100)
    
    # Ground Truth
    true_d = 150.0 # nm
    true_n = 1.6
    
    # Create Indices Array (3 layers x N_wavs)
    # Layer 0: Air
    idx_0 = np.full((3, 100), 1.0+0j, dtype=np.complex128)
    # Layer 1: The Film (True value)
    idx_0[1, :] = true_n + 0j
    # Layer 2: Substrate (BK7ish)
    idx_0[2, :] = 1.5 + 0j
    
    th_0 = np.array([0.0, true_d, 0.0]) # Thicknesses
    
    # Generate "Measured" Data
    Rs, Rp, _, _ = core_engine(wavls, np.radians(0), 3, idx_0, th_0, 
                               np.zeros(3, bool), np.zeros(3, int), np.zeros(3, float), 
                               True, True)
    R_true = (Rs + Rp)/2.0
    # Add Noise
    R_meas = R_true + np.random.normal(0, 0.005, size=len(R_true))
    
    # --- 2. Setup Optimization Structure ---
    
    # Guess values
    guess_d = 100.0
    guess_n = 2.0
    beta0 = [guess_d, guess_n]
    
    # Define Mapping (-1 = fixed, 0 = beta[0], 1 = beta[1])
    map_thick = np.array([-1, 0, -1], dtype=np.int32) # Layer 1 thickness is beta[0]
    map_ind   = np.array([-1, 1, -1], dtype=np.int32) # Layer 1 index is beta[1]
    
    struct_def = {
        'n_layers': 3,
        'base_indices': idx_0, # This holds the fixed values
        'base_thick': th_0,    # This holds the fixed values
        'inc_flags': np.zeros(3, bool),
        'r_types': np.zeros(3, int),
        'r_vals': np.zeros(3, float),
        'map_thick': map_thick,
        'map_ind': map_ind
    }
    
    # Define Constraints
    # Thickness must be > 0. Index must be between 1.4 and 1.8.
    constraints = [
        {'param_idx': 0, 'min': 0.0, 'weight': 1e6},      # Thickness constraint
        {'param_idx': 1, 'min': 1.4, 'max': 1.8, 'weight': 1e6} # Index constraint
    ]
    
    # --- 3. Run ODR ---
    print("Starting Optimization...")
    fitter = ODR_TMM_Fitter(wavls, R_meas, 0.0, struct_def, constraints)
    res = fitter.run_fit(beta0)
    
    print("\n--- Results ---")
    print(f"True Params:   d={true_d}, n={true_n}")
    print(f"Fitted Params: d={res.beta[0]:.4f}, n={res.beta[1]:.4f}")
    print(f"Stop Reason: {res.stopreason}")
    
    # --- 4. Verify Constraints ---
    # Let's try to fit with a starting guess that violates constraints 
    # to prove the penalty works.
    print("\nTest: Starting with invalid index guess (n=3.0, max is 1.8)")
    res_constrained = fitter.run_fit([100.0, 3.0])
    print(f"Fitted Params (Constrained): d={res_constrained.beta[0]:.4f}, n={res_constrained.beta[1]:.4f}")
    
    # Plotting
    # Re-calculate spectrum with fitted beta
    # Note: We just use the first part of the output (exclude penalty residuals)
    R_fit_aug = fitter.fcn_odr(res.beta, None) 
    R_fit = R_fit_aug[:len(wavls)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(wavls, R_meas, 'k.', label='Measured (Noisy)')
    plt.plot(wavls, R_fit, 'r-', linewidth=2, label='ODR Fit')
    plt.plot(wavls, R_true, 'g--', alpha=0.5, label='True Spectrum')
    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Numba TMM + ODR with Constraints')
    plt.show()