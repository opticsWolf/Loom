# Install maturin
pip install maturin

# Build and install the Python module
maturin develop

# Test from Python
python -c "
import navette_spectralweave
import numpy as np

weaver = navette_spectralweave.OpticalWeaver(cache_size=64)
key = (550.0, 'R', 's')
wl = np.linspace(400, 800, 100)
data = np.sin(wl / 100)
weaver.set_data(key, data, wl)

wl_out, data_out = weaver.get_weaved(key)
print(wl_out[:5], data_out[:5])
"