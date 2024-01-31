from __future__ import annotations
import csv
import sys
from location.location import Location
from logger import LOGGER
import shapely

# csv.field_size_limit(sys.maxsize)
if sys.maxsize > 2**32:
    csv.field_size_limit(2**31 - 1)  # für 64-Bit-Systeme
else:
    csv.field_size_limit(2**15 - 1)


# We have the problem that zones not all the time match some straight lines
# We define our zones as sets of smaller squares where it is super easy to
# find the fitting square in a coordinate system (2-layer binary search)
# Initial setup in static_data_generation/grid_builder.py
class Zone:
    _zones: list[Zone] = None

    def get_zones() -> list[Zone]:
        if Zone._zones == None:
            LOGGER.debug("Starting to create zones")
            Zone._zones = []
            csv_file_path = "code/data/taxi_zones.csv"
            with open(csv_file_path, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    zone_id = int(row["LocationID"])

                    # [(long, lat), (long, lat), ...]
                    polygon = shapely.from_wkt(row["the_geom"])
                    point = polygon.centroid
                    Zone._zones.append(
                        Zone(zone_id, Location(point.y, point.x), polygon)
                    )
            # Add a default zone for not found
            Zone._zones.append(Zone(9999, Location(40.749751, -74.016235), None))
            LOGGER.debug("Finished to create zones")

            LOGGER.debug("Set adjacent zones")
            csv_file_path = "code/data/zone_neighborhoods_by_extended_distance.csv"
            with open(csv_file_path, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    zone_id = int(row["Zone"])
                    zone = list(filter(lambda x: x.id == zone_id, Zone._zones))[0]

                    for i in range(1, 21):
                        zone.adjacent_zones_dict[i].extend(
                            list(
                                map(
                                    lambda x: int(x),
                                    filter(
                                        lambda x: x != "",
                                        row[f"{i}"].strip("][").split(", "),
                                    ),
                                )
                            )
                        )

        return Zone._zones

    def __init__(self, id: int, central_location: Location, polygon) -> None:
        self.id = id
        self.central_location = central_location
        self.polygon = polygon
        self.adjacent_zones_dict: dict[int, list[int]] = {i: [] for i in range(1, 21)}

    # distance in meters
    def find_adjacent_zone_ids(self, distance: int) -> list[int]:
        zones = []
        for i in range(1, (distance // 1000) + 1):
            zones.extend(self.adjacent_zones_dict[i])

        return zones
