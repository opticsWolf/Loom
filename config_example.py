import numpy as np
from loom.config import (
    load_material_library,
    save_architect,
    load_architect,
    LayerConfig,
    layer_from_config,
)
from loom_structure import Loom_Structure, Loom_Architect

# 1. Load material library
wl = np.linspace(400, 800, 300)
provider = load_material_library("materials.yaml", wl, use_code_map=True)

# 2. Create layers from config (could also load from YAML)
layers = [
    layer_from_config(LayerConfig(material_code="L", thickness_nm=100.0), provider),
    layer_from_config(LayerConfig(material_code="H", thickness_nm=60.0), provider),
]
structure = Loom_Structure(layers, materials=provider)

# 3. Build architect
arch = Loom_Architect(materials=provider)
arch.add_structure(structure)

# 4. Save architect state
save_architect(arch, "arch_state.yaml", fmt="yaml")

# 5. Load back
arch2 = load_architect("arch_state.yaml", provider)