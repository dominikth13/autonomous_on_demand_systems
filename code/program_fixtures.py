STATE_VALUE_TABLE = None

DRIVERS = None

STATE = None


def initialize():
    from state import State as State
    from driver import Driver as Driver
    from state_value_table import StateValueTable, Time, TimeSeries, Grid
    from location import Location
    import random
    STATE_VALUE_TABLE = StateValueTable(
        Grid(0, 0, 10, 10, 1), TimeSeries(Time(3, 0), Time(12, 0), 1)
    )
    DRIVERS = [
        Driver(
            Location(
                random.Random(i).randint(0, 10000),
                random.Random(i * i).randint(0, 10000),
            )
        )
        for i in range(100)
    ]
    STATE = State()
