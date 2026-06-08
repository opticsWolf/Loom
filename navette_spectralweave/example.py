import navette.spectralweave
import numpy as np

# Create a weaver
weaver = navette.spectralweave.OpticalWeaver(cache_size=64)

# Define a key
key = (550.0, "R", "s")

# Set data
wavelength = np.linspace(400, 800, 100)
values = np.sin(wavelength / 100)
weaver.set_data(key, values, wavelength)

# Retrieve weaved
wl, data = weaver.get_weaved(key)
print(wl[:5], data[:5])

# Unweave (full curve)
full_wl = np.linspace(400, 800, 500)
full_data = np.cos(full_wl / 100)
updated = weaver.unweave(key, full_wl, full_data)
print(f"Updated {updated} frames")