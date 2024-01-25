import os

from shapely.geometry import MultiPoint
from geopandas import GeoSeries, GeoDataFrame
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint





file_path = 'code/data_output/cell_id.csv'
file_path1 = 'code/data/grid_cells.csv'
# Import the CSV file into a DataFrame
df_cell_id = pd.read_csv(file_path)
df_grid_cells = pd.read_csv(file_path1)




cell_id_list = df_cell_id['cell_id'].to_list()
zone_dictionary = {}
for p in df_grid_cells['zone_id']:
    if p not in zone_dictionary and p != 9999.0:
        index = df_grid_cells[df_grid_cells['zone_id'] == p].idxmax()[0]
        zone_dictionary[p] = {'counter':0, 'lat':df_grid_cells.at[index, 'zone_center_lat'] , 'lon': df_grid_cells.at[index, 'zone_center_lon']}


    
    
print(zone_dictionary)



for n in cell_id_list:
    zone = df_grid_cells.at[n, 'zone_id']
    if zone not in zone_dictionary and zone != 9999.0:
        zone_dictionary[zone]['counter'] = 1


    else:
        if zone != 9999.0:
            zone_dictionary[zone]['counter'] += 1



    

# Sort by keys
sorted_dict = {k: zone_dictionary[k] for k in sorted(zone_dictionary)}
dfo = pd.DataFrame.from_dict(zone_dictionary, orient='index')
 
print(dfo.head())



print(df_cell_id.head())
print(df_grid_cells.head())




#sns.heatmap(dfo['counter'], cmap='coolwarm')

# Overlay the scatter points. The 'zorder' parameter ensures the points are on top of the heatmap.


grid_cells_path = 'code/data/grid_cells.csv'
    

grid_cells_df = pd.read_csv(grid_cells_path)
    

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
fig, ax = plt.subplots(figsize=(8, 6))

    # Zeichnen der Zonengrenzen
zone_polygons_gdf.boundary.plot(ax=ax, color='blue', alpha=0.5, label='Zonengrenzen')
plt.scatter(dfo['lon'], dfo['lat'], c=dfo['counter'],  cmap='viridis', zorder=2)
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
plt.colorbar()
# Show the heatmap
plt.show()
