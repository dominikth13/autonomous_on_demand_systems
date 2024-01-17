import csv
from driver.driver import Driver
from grid.grid import Grid
from location.location import Location


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
                    location = Location(float(row["lat"]), float(row["lon"]))
                    Drivers._drivers.append(Driver(location))

        return Drivers._drivers

    def export_drivers() -> None:
        drivers = Drivers.get_drivers()
        csv_file_path = "code/data/drivers.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["driver_id", "lat", "lon"])
            for driver in drivers:
                writer.writerow(
                    [
                        driver.id,
                        driver.current_position.lat,
                        driver.current_position.lon,
                    ]
                )
