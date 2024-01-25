from __future__ import annotations

from location.location import Location
from program.program_params import ProgramParams

class DriverJob:
    # time in seconds
    def __init__(self, total_driving_time: int, passenger_pickup_time: int, pre_pickup_position: Location, pickup_position: Location, final_position: Location, is_relocation: bool) -> None:
        self.is_relocation = is_relocation
        
        positions = []
        if not is_relocation:
            pickup_stops = passenger_pickup_time / ProgramParams.SIMULATION_UPDATE_RATE
            if pickup_stops > 0:
                lat_steps = (pickup_position.lat - pre_pickup_position.lat) / pickup_stops
                lon_steps = (pickup_position.lon - pre_pickup_position.lon) / pickup_stops
                pickup_stops = int(pickup_stops)
                for i in range(1, pickup_stops + 1):
                    lat = i*lat_steps + pre_pickup_position.lat
                    lon = i*lon_steps + pre_pickup_position.lon
                    positions.append((lat, lon))
        
        dropoff_stops = (total_driving_time - passenger_pickup_time) / ProgramParams.SIMULATION_UPDATE_RATE
        if dropoff_stops > 0:
            lat_steps = (final_position.lat - pickup_position.lat) / dropoff_stops
            lon_steps = (final_position.lon - pickup_position.lon) / dropoff_stops
            dropoff_stops = int(dropoff_stops)
            for i in range(1, dropoff_stops + 1):
                lat = i*lat_steps + pickup_position.lat
                lon = i*lon_steps + pickup_position.lon
                positions.append((lat, lon))
        
        self.positions = iter(positions)
        self.final_position = final_position
        self.open_trip_time = total_driving_time
    
    def of_trip(total_driving_time: int, passenger_pickup_time: int, driver_position: Location, pickup_position: Location, final_position: Location) -> DriverJob:
        return DriverJob(total_driving_time, passenger_pickup_time, driver_position, pickup_position, final_position, False)

    def of_relocation(total_driving_time: int, driver_position: Location, final_position: Location) -> DriverJob:
        return DriverJob(total_driving_time, 0, None, driver_position, final_position, True)
    
    def get_next_position(self) -> Location:
        (lat, lon) = next(self.positions, (self.final_position.lat, self.final_position.lon))
        return Location(lat, lon)

