from utils import IdProvider

ID_PROVIDER = IdProvider()

class Action:
    from route import Route

    def __init__(self, route: Route) -> None:
        self.id = ID_PROVIDER.get_id()
        self.route = route
        self.idling = route == None

    def is_idling(self) -> bool:
        return self.idling

    def is_route(self) -> bool:
        return not self.idling

    def __str__(self):
        return f"{f'Route {self.route.id}' if self.is_route() else 'Idling'}"
