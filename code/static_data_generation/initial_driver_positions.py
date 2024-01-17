import csv
import random
from driver.driver import Driver
from grid.grid import Grid
from interval.time import Time
from order import Order



def initialize_driver_positions() -> None:
    # We do a similar approach as in the Feng et al. (2022) paper: 
    # the distribution of drivers positions follows the distribution 
    # of orders in the first 30min of the studied period.
    # In our case we randomly take orders from the first 30mins 
    # and do a one to one mapping with drivers, where each driver 
    # is positioned randomly in a radius of two GridCells from the order.
    grid = Grid.get_instance()
    orders_by_time = Order.get_orders_by_time()
    first_orders = []
    for minute in range(30):
        first_orders.extend(orders_by_time[Time.of_total_minutes(minute)])
    from program.program_params import ProgramParams
    sampled_orders = random.Random(42).choices(first_orders, k=ProgramParams.AMOUNT_OF_DRIVERS)

    counter = 0
    drivers = []
    for order in sampled_orders:
        cell = grid.find_cell(order.start)
        cells = list(grid.find_n_adjacent_cells(cell, 2))
        driver_cell = random.Random(counter).choice(cells)
        drivers.append(Driver(driver_cell.center))
        counter += 1
    
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

