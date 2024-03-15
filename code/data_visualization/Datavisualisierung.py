import pandas as pd
from shapely.geometry import MultiPoint
from geopandas import GeoSeries, GeoDataFrame
import matplotlib.pyplot as plt
from driver.drivers import Drivers
from order import Order

def visualize_drivers(output_file_name: str):
    # Daten laden
    grid_cells_path = 'code/data/grid_cells.csv'
    subway_data_path = 'code/data/continuous_subway_data.csv'

    grid_cells_df = pd.read_csv(grid_cells_path)
    subway_data_df = pd.read_csv(subway_data_path)
    drivers_data_df = pd.DataFrame(list(map(lambda x: {"lat": x.current_position.lat, "lon": x.current_position.lon}, Drivers.get_drivers())))
    print(len(drivers_data_df[drivers_data_df['lat'] <= 41.65]))
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
    plt.scatter(subway_data_df['LONG'], subway_data_df['LAT'], c='darkgrey', label='Stationen', alpha=0.9, s = 5)

    # Fahrer einzeichnen
    plt.scatter(drivers_data_df['lon'], drivers_data_df['lat'], c='green', label='Fahrer', alpha=0.6, s = 25)

    # Legende hinzufügen
    plt.legend()

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title("Karte mit Zonengrenzen, Stationen und Fahrern")
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.savefig(f'code/data_visualization/{output_file_name}', dpi=600)
    # Karte anzeigen
    plt.show()


def visualize_orders(output_file_name: str):
    # Daten laden
    grid_cells_path = 'code/data/grid_cells.csv'
    subway_data_path = 'code/data/continuous_subway_data.csv'

    grid_cells_df = pd.read_csv(grid_cells_path)
    subway_data_df = pd.read_csv(subway_data_path)
    #drivers_data_df = pd.DataFrame(list(map(lambda x: {"lat": x.current_position.lat, "lon": x.current_position.lon}, Drivers.get_drivers())))

    # Daten der Bestellungen abrufen
    orders_by_time = Order.get_orders_by_time()
    orders_data = []
    for time, orders in orders_by_time.items():
        for order in orders:
            orders_data.append({"lat": order.start.lat, "lon": order.start.lon})

    orders_data_df = pd.DataFrame(orders_data)

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
    #plt.scatter(drivers_data_df['lon'], drivers_data_df['lat'], c='green', label='Fahrer', alpha=0.8)

    # Bestellungen einzeichnen
    plt.scatter(orders_data_df['lon'], orders_data_df['lat'], c='blue', label='Bestellungen', alpha=0.15, s = 5)

    # Legende hinzufügen
    plt.legend()

    # Titel und Achsenbeschriftungen hinzufügen
    plt.title("Karte mit Zonengrenzen, Stationen und Fahrern")
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.savefig(f'code/data_visualization/{output_file_name}', dpi=600)

    # Karte anzeigen
    #plt.show()
