import random
import time
from algorithm.algorithm import generate_driver_action_pairs, generate_routes, solve_optimization_problem
from interval.time_series import Time
from location.location import Location
from order import Order
import pandas as pd
from logger import LOGGER
from state.state import State
from state.state_value_table import StateValueTable
import numpy as np

grid_cells = pd.read_csv('code\data\grid_cells.csv') 
grid_cells.columns = ['Index', 'id', 'lat', 'long', 'zone_id']

np.random.seed(42)
random.seed(42)
def get_random_coordinates_for_zone(zone):
    
    possible_locations = grid_cells[grid_cells['zone_id'] == zone]
    if not possible_locations.empty:
        selected_row = possible_locations.sample(n=1).iloc[0]
        return (selected_row['lat'], selected_row['long'])
    else:
        # Entscheiden, wie mit fehlenden Koordinaten umgegangen werden soll
        raise ValueError(f"Keine Koordinaten für Zone {zone} gefunden")

def q_learning():
    start_time = time.time()
    import_state_values_from_csv()

    # 1. Find all shortest paths in public transport network
    # Is done automatically in station.py
    np.random.seed(42)
    random.seed(42)
    # 2. Run Q-Learning algorithm to train state value table
    counter = 1
    df = pd.read_csv('code\data\orders_2015-07-01.csv')

    for start_minutes in range(
        StateValueTable.get_state_value_table().time_series.start_time.to_total_minutes(),
        StateValueTable.get_state_value_table().time_series.end_time.to_total_minutes() - 360,
    ):
        current_time = Time.of_total_minutes(start_minutes)
        LOGGER.info(f"Simulate time {current_time}")

        # Filtern der Daten basierend auf dem Zeitfenster
        hours, minutes, seconds = current_time.to_hours_minutes_seconds()
        df['pickup_time'] = pd.to_datetime(df['pickup_time'], format='%H:%M:%S').dt.time
        filtered_orders = df[(df['pickup_time'].apply(lambda x: x.hour) == hours) &(df['pickup_time'].apply(lambda x: x.minute) == minutes)]

        orders = [
            Order(
            Location(*get_random_coordinates_for_zone(row.PULocationID)),
            Location(*get_random_coordinates_for_zone(row.DOLocationID))
        ) for row in filtered_orders.itertuples()]     

        # Add orders to state
        State.get_state().add_orders(orders)
        # Generate routes
        LOGGER.debug("Generate routes")
        order_routes_dict = generate_routes(orders)
        # Generate Action-Driver pairs with all available routes and drivers
        LOGGER.debug("Generate driver-action-pairs")
        driver_action_pairs = generate_driver_action_pairs(order_routes_dict)
        # Find Action-Driver matches based on a min-cost-flow problem
        LOGGER.debug("Generate driver-action matches")
    
        matches = solve_optimization_problem(driver_action_pairs)

        # Apply state changes based on Action-Driver matches and existing driver jobs
      
        LOGGER.debug("Apply state-value changes")
        State.get_state().apply_state_change(matches)
        # Update the expiry durations of still open orders
        State.get_state().update_order_expiry_duration()
        # Increment to next interval
        State.get_state().increment_time_interval(current_time)
        counter += 1

    LOGGER.info("Exporting results...")
    export_epoch_to_csv()
    LOGGER.info(f"Algorithm took {time.time() - start_time} seconds to run.")

def export_epoch_to_csv():
    export_table = pd.DataFrame(
        columns=["start_time", "end_time", "zone_name", "state_value"]
    )
    for time_interval in StateValueTable.get_state_value_table().value_grid:
        for zone in StateValueTable.get_state_value_table().value_grid[time_interval]:
            export_table.loc[len(export_table)] = [
                time_interval.start.to_total_seconds(),
                time_interval.end.to_total_seconds(),
                zone.id,
                StateValueTable.get_state_value_table().value_grid[time_interval][zone],
            ]

    export_table.to_csv("code/training_data/state_value_table.csv")


def import_state_values_from_csv():
    import_table = pd.read_csv("code/training_data/state_value_table.csv")

    for _, row in import_table.iterrows():
        start_time = Time.of_total_seconds(int(row["start_time"]))
        interval = StateValueTable.get_state_value_table().time_series.find_interval(start_time)
        zone = StateValueTable.get_state_value_table().grid.zones_dict[row["zone_name"]]
        state_value = float(row["state_value"])
        StateValueTable.get_state_value_table().value_grid[interval][zone] = state_value

q_learning()


# INFO:algorithm:Simulate time 03:00:00
# Laufzeit filtered_orders: 0.0 Sekunden
# Laufzeit orders: 620.99849152565 Sekunden
# Laufzeit add_orders: 0.029900312423706055 Sekunden
# Laufzeit generate Routes: 345.75252294540405 Sekunden
# Laufzeit generate_driver_action_pairs: 794.6348361968994 Sekunden
# Laufzeit solve optimization_problem: 31.162750244140625 Sekunden
# Laufzeit rest: 0.7596573829650879 Sekunden
# INFO:algorithm:Simulate time 03:01:00
# Laufzeit filtered_orders: 0.0 Sekunden