from customer import Customer
from location import Location

class Order:
    def __init__(self, customer: Customer, start: Location, end: Location) -> None:
        self.customer = customer
        self.start = start
        self.end = end