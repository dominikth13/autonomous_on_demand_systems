from utils import IdProvider
from customer import Customer
from location import Location

ID_PROVIDER = IdProvider()

class Order:
    def __init__(self, customer: Customer, start: Location, end: Location) -> None:
        self.id = ID_PROVIDER.get_id()
        self.customer = customer
        self.start = start
        self.end = end