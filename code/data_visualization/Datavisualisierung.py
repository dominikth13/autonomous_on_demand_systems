import pandas as pd
from shapely.geometry import MultiPoint
from geopandas import GeoSeries, GeoDataFrame
import matplotlib.pyplot as plt

def visualize_drivers():
    # Daten laden
    grid_cells_path = 'code/data/grid_cells.csv'
    subway_data_path = 'code/data/continuous_subway_data.csv'
    drivers_data_path = 'code/data/drivers.csv'

    grid_cells_df = pd.read_csv(grid_cells_path)
    subway_data_df = pd.read_csv(subway_data_path)
    drivers_data_df = pd.read_csv(drivers_data_path)

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
    fig, ax = plt.subplots(figsize=(12, 12))

    # Zeichnen der Zonengrenzen
    zone_polygons_gdf.boundary.plot(ax=ax, color='blue', alpha=0.5, label='Zonengrenzen')

    # Stationen einzeichnen
    plt.scatter(subway_data_df['LONG'], subway_data_df['LAT'], c='red', label='Stationen', alpha=0.8)

    # Fahrer einzeichnen
    plt.scatter(drivers_data_df['lon'], drivers_data_df['lat'], c='green', label='Fahrer', alpha=0.8)

    # Legende hinzufügen
    plt.legend()

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title("Karte mit Zonengrenzen, Stationen und Fahrern")
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.savefig('code\Data Visualization\Karte.png', dpi=600)
    # Karte anzeigen
    plt.show()

