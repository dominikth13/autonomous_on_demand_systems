import numpy as np
import pandas as pd
from grid.grid_cell import GridCell
from location.location import Location
from location.zone import Zone
from logger import LOGGER

# erstellt ein Grid aus 100*100Meter "Kacheln" für ganz NewYork, außer Staten Island
def to_zone(location: Location):

    from copy import deepcopy
    zones = sorted(deepcopy(Zone.get_zones()[:-1]), key=lambda zone: zone.central_location.distance_to(location))
    default = Zone.get_zones()[-1]
    from shapely.geometry import Point

    point = Point(location.lon, location.lat)

    for zone in zones:
        if zone.polygon.contains(point):
            return zone 
    
    return default

def to_zone_center_lat(location: Location):

    from copy import deepcopy
    zones = sorted(deepcopy(Zone.get_zones()[:-1]), key=lambda zone: zone.central_location.distance_to(location))
    default = Zone.get_zones()[-1]
    from shapely.geometry import Point

    point = Point(location.lon, location.lat)

    for zone in zones:
        if zone.polygon.contains(point):
            return zone.central_location.lat 
    
    return default

def to_zone_center_lon(location: Location):

    from copy import deepcopy
    zones = sorted(deepcopy(Zone.get_zones()[:-1]), key=lambda zone: zone.central_location.distance_to(location))
    default = Zone.get_zones()[-1]
    from shapely.geometry import Point

    point = Point(location.lon, location.lat)

    for zone in zones:
        if zone.polygon.contains(point):
            return zone.central_location.lon 
    
    return default

def create_cell_grid():
    min_lat = 40.534522
    min_lon = -74.050826
    max_lat = 40.925205
    max_lon = -73.685841
    step_distance = 0.001

    cells = []
    counter = 0
    total_amount = len(np.arange(min_lat, max_lat, step_distance)) * len(np.arange(min_lon, max_lon, step_distance))
    for lat in np.arange(min_lat, max_lat, step_distance):
        for lon in np.arange(min_lon, max_lon, step_distance):
            LOGGER.debug(f" Generate cell {counter}/{total_amount}")
            cells.append(GridCell(Location(lat, lon), to_zone(Location(lat, lon)) ))
            counter += 1
    
    export_table = pd.DataFrame(
        columns=["id", "lat", "long", "zone_id", "zone_center_lat" , "zone_center_lon"]
    )
    for cell in cells:
        export_table.loc[len(export_table)] = [
            cell.id,
            cell.center.lat,
            cell.center.lon,
            cell.zone.id,
            cell.zone.central_location.lat,
            cell.zone.central_location.lon
        ]

    export_table.to_csv("code/data/grid_cells.csv")