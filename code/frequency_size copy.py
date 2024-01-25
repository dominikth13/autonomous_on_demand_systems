import os


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from geopandas import GeoSeries, GeoDataFrame

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from geopandas import GeoSeries, GeoDataFrame
from program.program_params import Mode, ProgramParams

# Create a map centered around a specific location
# Here I'm using the coordinates for New York City as an example

file_path2 = 'code/data_output/driver_data.csv'
dr_data = pd.read_csv(file_path2)

# Read the CSV file


# Create a figure and axis


# Function to update the animation
def update(frame):
    frame = int(frame)
    ax.clear()
    ax.coastlines()
    
    
    ax.scatter(dr_data['lon'][frame:frame+9], dr_data['lat'][frame:frame+9])
    ax.set_xlim(grid_cells_df['long'].min(), grid_cells_df['long'].max())
    ax.set_ylim(grid_cells_df['lat'].min(), grid_cells_df['lat'].max())
    
            
        
        

        
        

    # Assuming 'Time' is your frame and 'X', 'Y' are coordinates
    

# Create an animation









#sns.heatmap(dfo['counter'], cmap='coolwarm')

# Overlay the scatter points. The 'zorder' parameter ensures the points are on top of the heatmap.


grid_cells_path = 'code/data/grid_cells.csv'
    

grid_cells_df = pd.read_csv(grid_cells_path)
print(grid_cells_df.head())    

    # Funktion zur Berechnung der Grenzen einer Zone
def compute_zone_boundaries(zone_id, df):
    zone_data = df[df['zone_id'] == zone_id]
    points = MultiPoint(list(zip(zone_data['long'], zone_data['lat'])))
    polygon = points.convex_hull
    return polygon

    # Berechnen der Grenzen für jede Zone
unique_zones = grid_cells_df['zone_id'].unique()
zone_boundaries = [compute_zone_boundaries(zone, grid_cells_df) for zone in unique_zones]

    # Erstellen eines GeoDataFrame aus den Polygonen
zone_polygons = GeoSeries(zone_boundaries)
zone_polygons_gdf = GeoDataFrame(geometry=zone_polygons)

    # Erstellen der Karte
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Zeichnen der Zonengrenzen
zone_polygons_gdf.boundary.plot(ax=ax, color='blue', alpha=0.5, label='Zonengrenzen')
sequence = np.arange(0, len(dr_data), 10)
ax.set_global()  # Or set_extent to your area of interest
ax.coastlines()
ani = animation.FuncAnimation(fig, update, frames=sequence, repeat=False, interval = 1000)


# Binning the longitude and latitude to create discrete zones
# You can adjust the bin sizes as per your requirement
#df['longitude_bin'] = pd.cut(df['longitude'], bins=np.linspace(min(df['longitude']), max(df['longitude']), 39))
#df['latitude_bin'] = pd.cut(df['latitude'], bins=np.linspace(min(df['latitude']), max(df['latitude']), 36))

# Aggregating the frequencies by these bins
# This will give us the absolute frequency for each bin
#frequency_table = df.groupby(['latitude_bin', 'longitude_bin']).frequency.sum().reset_index()

# Pivot the aggregated frequency data
#heatmap_data = frequency_table.pivot(index='latitude_bin', columns='longitude_bin', values='frequency').fillna(0)



# Create the heatmap using seaborn
#plt.figure(figsize=(12, 6))
#sns.heatmap(heatmap_data, cmap='viridis', annot=True)

# Adding labels and title
plt.title('Heatmap of Absolute Frequencies by Longitude and Latitude')
plt.xlabel('Longitude Bin')
plt.ylabel('Latitude Bin')

# Show the heatmap
plt.show()
