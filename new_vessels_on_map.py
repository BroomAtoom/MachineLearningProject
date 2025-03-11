import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

output_directory = r'C:\Users\jarri\Desktop\Spyder_Conda\bigproject'

# Load the data
df = pd.read_csv(os.path.join(output_directory, 'aarburg_data.csv'))
df = df[df['status'] == 'under-way-using-engine']
df = df[:100]  # Number of datapoints to plot

# Ensure longitude and latitude are numeric
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

# Drop NaN values
df = df.dropna(subset=['longitude', 'latitude'])

# Extract coordinates
longitudes = df['longitude']
latitudes = df['latitude']

# Create figure and map
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())  # Use consistent CRS

# Set map extent with padding
ax.set_extent([longitudes.min() - 0.1, longitudes.max() + 0.1,
               latitudes.min() - 0.1, latitudes.max() + 0.1], crs=ccrs.PlateCarree())
land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor=cfeature.COLORS['land'])
borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', edgecolor='black', linestyle=':')
coastlines = cfeature.NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black')
rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m', edgecolor='blue', linewidth=1.2)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m', facecolor='lightblue')

# Add map features
ax.add_feature(land)
ax.add_feature(borders)
ax.add_feature(coastlines)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(rivers)


# Plot vessel path
ax.plot(longitudes, latitudes, marker='o', color='red', linewidth=0.5, markersize=1, label='Vessel Path', transform=ccrs.PlateCarree())

# Add title and legend
plt.title('Vessel Trip Path')
plt.legend()

# Show plot
plt.show()
