import csv
from driver.driver import Driver
from location.location import Location
from state.state_value_table import StateValueTable

# Singleton class containing all the drivers
class Drivers:
    _drivers: list[Driver] = None

    def get_drivers() -> list[Driver]:
        if Drivers._drivers == None:
            Drivers._drivers = []
            csv_file_path = "code/data/drivers.csv"
            with open(csv_file_path, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    zone_id = int(row["Zone_ID"])
                    location = StateValueTable.get_state_value_table().grid.zones_dict[zone_id].central_location
                    Drivers._drivers.append(Driver(location))

        return Drivers._drivers