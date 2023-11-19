from utils import IdProvider
from location import Location
from program_params import *

ID_PROVIDER = IdProvider()

class Order:
    def __init__(self, start: Location, end: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.start = start
        self.end = end
        self.expires = ORDER_EXPIRY_DURATION

        fastest_connection = None
        from station import FastestStationConnectionNetwork
        fastest_connection_network = FastestStationConnectionNetwork.get_instance()
        # Find the most fastest connection without any autonomous on-demand services
        for origin in fastest_connection_network.stations:
            for destination in fastest_connection_network.stations:
                if origin == destination:
                    continue
                connection = fastest_connection_network.get_fastest_connection(origin, destination)
                total_walking_time = (start.distance_to(origin.position) + destination.position.distance_to(end)) / WALKING_SPEED

                if fastest_connection == None or fastest_connection[1] > total_walking_time + connection[1]:
                    fastest_connection = (connection[0], connection[1] + total_walking_time)
        
        self.direct_connection = fastest_connection