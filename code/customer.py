from utils import IdProvider

ID_PROVIDER = IdProvider()

class Customer:
    def __init__(self) -> None:
        self.id = IdProvider().get_id()