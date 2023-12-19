import torch
import torch.nn as nn
import torch.optim as optim
from deep_rl_setup import NeuroNet , td_error , generate_driver_action_pairs_without_weights
import random
import time
from algorithm import (
    generate_driver_action_pairs,
    generate_routes,
    solve_optimization_problem,
)
from time_interval import Time
from state import STATE
from location import Location
from order import Order
import pandas as pd
from state_value_table import STATE_VALUE_TABLE
from logger import LOGGER




initialization_first_time = True
# Define a simple neural network



# Initialize the network
net = NeuroNet()
# Initialize a new network
target_net = NeuroNet()

if initialization_first_time == False:
    net.load_state_dict(torch.load('net_state_dict.pth'))
    #either model.eval() or model.train() depending on what we currently doing
    target_net.load_state_dict(torch.load('target_net_state_dict.pth'))
    #either model.eval() or model.train() depending on what we currently doing



 # Commonly used for classification problems
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent




# Training loop
for epocch_target in range(100):
    original_state_dict = net.state_dict() # Save the state dict of the original network
    # Load the state dict of the original network into the new network
    target_net.load_state_dict(original_state_dict)
    # Ensure that the new network is in the same mode (train or eval) as the original
    target_net.train(mode=net.training)
      

    start_time = time.time()
    

    # 1. Find all shortest paths in public transport network
    # Is done automatically in station.py

    # 2. Run Q-Learning algorithm to train state value table
    counter = 1
    for start_minutes in range(
        STATE_VALUE_TABLE.time_series.start_time.to_total_minutes(),
        # TimeSeries is two hours longer than the simulation
        STATE_VALUE_TABLE.time_series.end_time.to_total_minutes() - 120,
    ):
        current_time = Time.of_total_minutes(start_minutes)
        LOGGER.info(f"Starting iteration {counter}")
        LOGGER.debug("Collecting orders")
        # Collect new orders
        orders = [
            Order(
                Location(
                    random.randint(0, 10000),
                    random.randint(0, 10000),
                ),
                Location(
                    random.randint(0, 10000),
                    random.randint(0, 10000),
                ),
            )
            for i in range(random.randint(0, 50))
        ]

        if counter == 232:
            pass

        # Add orders to state
        STATE.add_orders(orders)
        # Generate routes
        LOGGER.debug("Generate routes")
        order_routes_dict = generate_routes(orders)
        # Generate Action-Driver pairs with all available routes and drivers
        LOGGER.debug("Generate driver-action-pairs")
        driver_action_pairs = generate_driver_action_pairs_without_weights(order_routes_dict)

    print(type(driver_action_pairs))   

    for epoch in range(100):  # loop over the dataset multiple times
        index = random.randint(0,len(driver_action_pairs)-1)

        inputs_next_state = torch.randn(1, 10)  # Example input vector (batch size 1, 10 features)
        inputs_current_state = torch.randn(1, 10)
        optimizer.zero_grad()  # zero the parameter gradients
        output_target_net = target_net(driver_action_pairs.loc[index,'Target Destination'],driver_action_pairs.loc[index,'Target Arrival'])
        # Forward pass
        output = net(driver_action_pairs.loc[index,'Current Time'], driver_action_pairs.loc[index,'Current Position'])

        # Compute loss
        loss = td_error(output, output_target_net, driver_action_pairs.loc[index,'Reward'])
        #loss = loss.pow(2)  # Squaring the TD error (if needed) I don`t want negative losses

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print statistics
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print('Finished Training')

torch.save(net.state_dict(), 'net_state_dict.pth')
torch.save(target_net.state_dict(), 'target_net_state_dict.pth')
