import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# NDVI file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ndvi_2020_path = os.path.join(script_dir, '../1.data/NDVI_Rochester_2020.tif')
ndvi_2024_path = os.path.join(script_dir, '../1.data/NDVI_Rochester_2024.tif')

# Function to read NDVI data
def load_ndvi(path):
    with rasterio.open(path) as src:
        ndvi = src.read(1)
        ndvi = np.where((ndvi >= -1) & (ndvi <= 1), ndvi, np.nan)
        return ndvi

# Load data
ndvi_2020 = load_ndvi(ndvi_2020_path)
ndvi_2024 = load_ndvi(ndvi_2024_path)

# Calculate NDVI change
ndvi_diff = ndvi_2024 - ndvi_2020

# === Statistical Analysis ===
valid_mask = ~np.isnan(ndvi_diff)
valid_diff = ndvi_diff[valid_mask]

mean_change = np.nanmean(valid_diff)
min_change = np.nanmin(valid_diff)
max_change = np.nanmax(valid_diff)

increase_mask = valid_diff > 0
decrease_mask = valid_diff < 0

increase_count = np.sum(increase_mask)
decrease_count = np.sum(decrease_mask)
total_count = increase_count + decrease_count

increase_ratio = (increase_count / total_count) * 100
decrease_ratio = (decrease_count / total_count) * 100

print("=== NDVI Change Statistics (2024 - 2020) ===")
print(f"Mean change: {mean_change:.4f}")
print(f"Min change: {min_change:.4f}")
print(f"Max change: {max_change:.4f}")
print(f"Increase count: {increase_count} pixels ({increase_ratio:.2f}%)")
print(f"Decrease count: {decrease_count} pixels ({decrease_ratio:.2f}%)")

# Visualization
plt.figure(figsize=(6, 5))
plt.imshow(ndvi_2020, cmap='YlGn', vmin=0, vmax=1)
plt.title("NDVI 2020")
plt.colorbar()
plt.tight_layout()


plt.figure(figsize=(6, 5))
plt.imshow(ndvi_2024, cmap='YlGn', vmin=0, vmax=1)
plt.title("NDVI 2024")
plt.colorbar()
plt.tight_layout()


plt.figure(figsize=(6, 5))
plt.imshow(ndvi_diff, cmap='bwr', vmin=-0.5, vmax=0.5)
plt.title("NDVI Difference (2024 - 2020)")
plt.colorbar()
plt.tight_layout()
plt.show()

# Create directory for saving images if it doesn't exist
os.makedirs("../3.report", exist_ok=True)

# Save 
plt.figure(figsize=(6, 5))
plt.imshow(ndvi_2020, cmap='YlGn', vmin=0, vmax=1)
plt.title("NDVI 2020")
plt.colorbar()
plt.tight_layout()
plt.savefig("../3.report/ndvi_2020.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(ndvi_2024, cmap='YlGn', vmin=0, vmax=1)
plt.title("NDVI 2024")
plt.colorbar()
plt.tight_layout()
plt.savefig("../3.report/ndvi_2024.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(ndvi_diff, cmap='bwr', vmin=-0.5, vmax=0.5)
plt.title("NDVI Difference (2024 - 2020)")
plt.colorbar()
plt.tight_layout()
plt.savefig("../3.report/ndvi_diff.png", dpi=150)
plt.close()