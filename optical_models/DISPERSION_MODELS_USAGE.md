# Dispersion Models - Usage Examples (v3)

## Summary of Updated Classes

All dispersion model classes now use the hybrid parameter API:

1. **Cauchy** - Simple Cauchy dispersion (transparent materials)
2. **CauchyUrbach** - Cauchy with Urbach absorption tail
3. **Sellmeier** - Sellmeier dispersion (optical glasses)
4. **SellmeierUrbach** - Sellmeier with Urbach absorption tail

---

## Cauchy Model

### Basic Usage

```python
import numpy as np
from cauchy_sellmeier import Cauchy

# ===== METHOD 1: Individual Parameters =====
mat = Cauchy(A=1.5, B=0.01, C=0.0001)

# ===== METHOD 2: Dict-based (Config Files) =====
params = {
    'A': 1.5,
    'B': 0.01,
    'C': 0.0001
}
mat = Cauchy(params=params)

# ===== METHOD 3: With Wavelength Range =====
wavelength = np.linspace(400, 700, 100)
mat = Cauchy(A=1.5, B=0.01, C=0.0001, wavelength=wavelength)

# ===== ACCESS PARAMETERS =====
print(f"A = {mat.A}")  # 1.5
print(f"B = {mat.B}")  # 0.01
print(f"C = {mat.C}")  # 0.0001

# Get all parameters as dict
params = mat.get_params()
print(params)  # {'A': 1.5, 'B': 0.01, 'C': 0.0001}

# ===== MODIFY PARAMETERS =====
mat.set_param('A', 1.6)
print(mat.A)  # 1.6 (updated)

# ===== COMPUTE REFRACTIVE INDEX =====
wavelength = np.linspace(400, 700, 100)
nk = mat.complex_refractive_index(wavelength)

print(nk.shape)  # (100,)
print(nk[0])     # (1.60... + 0j) - complex, but k=0
print(nk.real)   # n values
print(nk.imag)   # k values (all zeros for Cauchy)
```

### Material Database Integration

```python
import json
import numpy as np
from cauchy_sellmeier import Cauchy

# ===== CREATE MATERIAL DATABASE =====
database = {
    "PMMA": {
        "model": "Cauchy",
        "params": {
            "A": 1.4893,
            "B": 0.00356,
            "C": 0.0
        },
        "wavelength_range": [400, 800],
        "description": "Poly(methyl methacrylate)"
    },
    "Polycarbonate": {
        "model": "Cauchy",
        "params": {
            "A": 1.5750,
            "B": 0.0045,
            "C": 0.0
        },
        "wavelength_range": [400, 800],
        "description": "Polycarbonate"
    }
}

# Save to JSON
with open('materials_database.json', 'w') as f:
    json.dump(database, f, indent=2)

# ===== LOAD FROM DATABASE =====
with open('materials_database.json', 'r') as f:
    db = json.load(f)

# Create material from database
mat_pmma = Cauchy(params=db["PMMA"]["params"])

print(f"PMMA: A={mat_pmma.A}, B={mat_pmma.B}")

# Compute refractive index
wavelength = np.linspace(*db["PMMA"]["wavelength_range"], 100)
nk = mat_pmma.complex_refractive_index(wavelength)
```

---

## CauchyUrbach Model

### Basic Usage

```python
import numpy as np
from cauchy_sellmeier import CauchyUrbach

# ===== INDIVIDUAL PARAMETERS =====
mat = CauchyUrbach(
    A=2.5,           # Cauchy constant
    B=0.02,          # Cauchy 1/λ² term
    C=0.0005,        # Cauchy 1/λ⁴ term
    alpha0=1e4,      # Absorption at band gap (1/cm)
    Eu=0.05,         # Urbach energy (eV)
    lambda_g=400     # Band gap wavelength (nm)
)

# ===== DICT-BASED =====
params = {
    'A': 2.5,
    'B': 0.02,
    'C': 0.0005,
    'alpha0': 1e4,
    'Eu': 0.05,
    'lambda_g': 400
}
mat = CauchyUrbach(params=params)

# ===== ACCESS PARAMETERS =====
print(f"Cauchy A = {mat.A}")
print(f"Band gap = {mat.lambda_g} nm")
print(f"Urbach energy = {mat.Eu} eV")

# ===== COMPUTE REFRACTIVE INDEX =====
wavelength = np.linspace(300, 700, 200)
nk = mat.complex_refractive_index(wavelength)

# Now k is non-zero near band edge
print(f"n at 500nm: {nk[100].real:.4f}")
print(f"k at 380nm (near band edge): {nk[40].imag:.6f}")
```

### Modeling Band Gap Absorption

```python
import numpy as np
import matplotlib.pyplot as plt
from cauchy_sellmeier import CauchyUrbach

# Create material with band gap at 400nm
mat = CauchyUrbach(
    A=2.5, B=0.02, C=0.0005,
    alpha0=1e4, Eu=0.05, lambda_g=400
)

wavelength = np.linspace(300, 700, 400)
nk = mat.complex_refractive_index(wavelength)

# Plot n and k
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(wavelength, nk.real)
ax1.axvline(mat.lambda_g, color='r', linestyle='--', label='Band gap')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Refractive Index (n)')
ax1.legend()
ax1.grid(True)

ax2.plot(wavelength, nk.imag)
ax2.axvline(mat.lambda_g, color='r', linestyle='--', label='Band gap')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Extinction Coefficient (k)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Study effect of Urbach energy
Eu_values = [0.03, 0.05, 0.07, 0.10]
plt.figure(figsize=(10, 6))

for Eu in Eu_values:
    mat.set_param('Eu', Eu)
    nk = mat.complex_refractive_index(wavelength)
    plt.plot(wavelength, nk.imag, label=f'Eu = {Eu} eV')

plt.axvline(400, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Extinction Coefficient (k)')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.title('Effect of Urbach Energy on Absorption')
plt.show()
```

---

## Sellmeier Model

### BK7 Glass Example

```python
import numpy as np
from cauchy_sellmeier import Sellmeier

# ===== BK7 OPTICAL GLASS (Schott) =====
# Individual parameters
mat_bk7 = Sellmeier(
    B1=1.03961212,
    C1=0.00600069867,
    B2=0.231792344,
    C2=0.0200179144,
    B3=1.01046945,
    C3=103.560653
)

# Or dict-based
params_bk7 = {
    'B1': 1.03961212, 'C1': 0.00600069867,
    'B2': 0.231792344, 'C2': 0.0200179144,
    'B3': 1.01046945, 'C3': 103.560653
}
mat_bk7 = Sellmeier(params=params_bk7)

# ===== COMPUTE DISPERSION =====
wavelength = np.linspace(400, 700, 100)
nk = mat_bk7.complex_refractive_index(wavelength)

# Print refractive indices at standard wavelengths
standard_wavelengths = {
    'd-line': 587.56,  # Yellow (Helium)
    'F-line': 486.13,  # Blue (Hydrogen)
    'C-line': 656.27   # Red (Hydrogen)
}

for name, wvl in standard_wavelengths.items():
    nk_temp = mat_bk7.complex_refractive_index(np.array([wvl]))
    print(f"{name} ({wvl:.2f} nm): n = {nk_temp[0].real:.6f}")

# Expected for BK7:
# d-line (587.56 nm): n ≈ 1.5168
# F-line (486.13 nm): n ≈ 1.5224
# C-line (656.27 nm): n ≈ 1.5143
```

### Two-Term Sellmeier

```python
# Many materials only need B1, B2, C1, C2 (B3, C3 default to 0)
mat_simple = Sellmeier(
    B1=1.0,
    C1=0.01,
    B2=0.3,
    C2=0.05
)
# B3 and C3 automatically set to 0.0

print(mat_simple.B3)  # 0.0
print(mat_simple.C3)  # 0.0
```

### Material Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from cauchy_sellmeier import Sellmeier

# Define several optical glasses
glasses = {
    'BK7': {
        'B1': 1.03961212, 'C1': 0.00600069867,
        'B2': 0.231792344, 'C2': 0.0200179144,
        'B3': 1.01046945, 'C3': 103.560653
    },
    'SF11': {
        'B1': 1.73759695, 'C1': 0.0131887070,
        'B2': 0.313747346, 'C2': 0.0623068142,
        'B3': 1.89878101, 'C3': 155.23629
    },
    'Fused Silica': {
        'B1': 0.6961663, 'C1': 0.0684043**2,
        'B2': 0.4079426, 'C2': 0.1162414**2,
        'B3': 0.8974794, 'C3': 9.896161**2
    }
}

wavelength = np.linspace(400, 1000, 300)

plt.figure(figsize=(12, 6))
for name, params in glasses.items():
    mat = Sellmeier(params=params)
    nk = mat.complex_refractive_index(wavelength)
    plt.plot(wavelength, nk.real, label=name, linewidth=2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive Index (n)')
plt.title('Dispersion Curves for Common Optical Glasses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## SellmeierUrbach Model

### Semiconductor Material Example

```python
import numpy as np
from cauchy_sellmeier import SellmeierUrbach

# ===== ZnO SEMICONDUCTOR =====
mat_zno = SellmeierUrbach(
    # Sellmeier coefficients
    B1=1.4313,
    C1=0.01,
    B2=0.65,
    C2=0.025,
    B3=0.0,
    C3=0.0,
    # Urbach parameters
    alpha0=1e5,      # High absorption at band edge
    Eu=0.06,         # Urbach energy
    lambda_g=380     # UV band gap
)

# ===== COMPUTE FULL SPECTRUM =====
wavelength = np.linspace(300, 800, 500)
nk = mat_zno.complex_refractive_index(wavelength)

# Analyze transparency window
transparent_region = nk.imag < 1e-3
print(f"Transparent from {wavelength[transparent_region].min():.1f} nm")
print(f"to {wavelength[transparent_region].max():.1f} nm")

# Refractive index in transparent region
n_avg = nk[transparent_region].real.mean()
print(f"Average n in transparent region: {n_avg:.3f}")
```

### Parameter Optimization Example

```python
import numpy as np
from scipy.optimize import minimize
from cauchy_sellmeier import SellmeierUrbach

# Experimental data (wavelength in nm, refractive index)
wvl_exp = np.array([400, 450, 500, 550, 600, 650, 700])
n_exp = np.array([2.45, 2.42, 2.40, 2.38, 2.37, 2.36, 2.35])

# Initial guess for parameters
initial_params = {
    'B1': 1.0, 'C1': 0.01,
    'B2': 0.5, 'C2': 0.02,
    'B3': 0.0, 'C3': 0.0,
    'alpha0': 1e4, 'Eu': 0.05, 'lambda_g': 380
}

def objective_function(x):
    """Minimize difference between model and experiment."""
    params = {
        'B1': x[0], 'C1': x[1],
        'B2': x[2], 'C2': x[3],
        'B3': 0.0, 'C3': 0.0,
        'alpha0': initial_params['alpha0'],
        'Eu': initial_params['Eu'],
        'lambda_g': initial_params['lambda_g']
    }
    
    mat = SellmeierUrbach(params=params)
    nk_model = mat.complex_refractive_index(wvl_exp)
    
    # Mean squared error
    mse = np.mean((nk_model.real - n_exp)**2)
    return mse

# Optimize
x0 = [initial_params['B1'], initial_params['C1'],
      initial_params['B2'], initial_params['C2']]

result = minimize(objective_function, x0, method='Nelder-Mead')

print(f"Optimized parameters:")
print(f"B1 = {result.x[0]:.6f}")
print(f"C1 = {result.x[1]:.6f}")
print(f"B2 = {result.x[2]:.6f}")
print(f"C2 = {result.x[3]:.6f}")
print(f"Final MSE = {result.fun:.6e}")
```

---

## Config File Best Practices

### JSON Material Library

```json
{
  "materials": {
    "PMMA": {
      "model": "Cauchy",
      "params": {
        "A": 1.4893,
        "B": 0.00356,
        "C": 0.0
      },
      "wavelength_range": [400, 800],
      "category": "polymer"
    },
    "BK7": {
      "model": "Sellmeier",
      "params": {
        "B1": 1.03961212,
        "C1": 0.00600069867,
        "B2": 0.231792344,
        "C2": 0.0200179144,
        "B3": 1.01046945,
        "C3": 103.560653
      },
      "wavelength_range": [365, 2325],
      "category": "optical_glass"
    },
    "GaN": {
      "model": "SellmeierUrbach",
      "params": {
        "B1": 1.75,
        "C1": 0.01,
        "B2": 0.87,
        "C2": 0.03,
        "B3": 0.0,
        "C3": 0.0,
        "alpha0": 5e4,
        "Eu": 0.07,
        "lambda_g": 365
      },
      "wavelength_range": [350, 800],
      "category": "semiconductor"
    }
  }
}
```

### Python Loader

```python
import json
import numpy as np
from cauchy_sellmeier import Cauchy, CauchyUrbach, Sellmeier, SellmeierUrbach

# Map model names to classes
MODEL_MAP = {
    'Cauchy': Cauchy,
    'CauchyUrbach': CauchyUrbach,
    'Sellmeier': Sellmeier,
    'SellmeierUrbach': SellmeierUrbach
}

def load_material(filename, material_name):
    """Load material from JSON database."""
    with open(filename, 'r') as f:
        db = json.load(f)
    
    mat_data = db['materials'][material_name]
    model_class = MODEL_MAP[mat_data['model']]
    
    # Create material
    mat = model_class(params=mat_data['params'])
    
    return mat, mat_data

# Usage
mat_bk7, info = load_material('materials.json', 'BK7')
wavelength = np.linspace(*info['wavelength_range'], 200)
nk = mat_bk7.complex_refractive_index(wavelength)

print(f"Loaded {info['category']} material")
print(f"Valid range: {info['wavelength_range'][0]}-{info['wavelength_range'][1]} nm")
```

---

## Migration from v2 to v3

### Old Code (v2)
```python
# BEFORE: Dict required, redundant attributes
params = {'A': 1.5, 'B': 0.01, 'C': 0.0001}
mat = Cauchy(params)

# Direct attribute access (could cause sync issues)
mat.A = 1.6  # params['A'] still 1.5 - BUG!
```

### New Code (v3)
```python
# AFTER: Multiple options, single source of truth
mat = Cauchy(A=1.5, B=0.01, C=0.0001)  # NEW
mat = Cauchy(params={'A': 1.5, 'B': 0.01, 'C': 0.0001})  # Still works

# Safe modification through set_param
mat.set_param('A', 1.6)  # Cache invalidated, always consistent
print(mat.A)  # 1.6 - property reflects self.params['A']
```

---

## Performance Notes

- Property access (`mat.A`) vs dict lookup (`self.params['A']`): ~2-3 ns difference
- Negligible for optical calculations (dominated by interpolation/computation)
- Single source of truth prevents bugs worth the tiny overhead
- JIT-compiled numba functions handle heavy computation efficiently
