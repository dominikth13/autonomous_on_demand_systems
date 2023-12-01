import csv
from driver.driver import Driver
from location.location import Location

# Singleton class containing all the drivers
class Drivers:
    _drivers: list[Driver] = None

    def get_drivers() -> list[Driver]:
        if Driver._drivers == None:
            Driver._drivers = []
            csv_file_path = "code/data/drivers.csv"
            with open(csv_file_path, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    lat = int(row["X"])
                    lon = int(row["Y"])
                    Driver._drivers.append(Driver(start_position=Location(lat, lon)))

        return Driver._drivers