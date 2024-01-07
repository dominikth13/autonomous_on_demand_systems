import csv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from deep_reinforcement_learning.deep_rl_setup import NeuroNet, td_error
from driver.driver import Driver
from driver.drivers import Drivers
from grid.grid import Grid
from torch.optim.lr_scheduler import StepLR
from logger import LOGGER

def import_trajectories() -> dict[int, dict[str, float]]:
    trajectories = {}
    csv_file_path = "code/data/trajectories.csv"
    with open(csv_file_path, mode="r") as file:
        reader = csv.DictReader(file)
        counter = 0
        for row in reader:
            trajectory = {}
            trajectory["reward"] = float(row["reward"])
            trajectory["target_time"] = int(row["target_time"])
            trajectory["target_lat"] = float(row["target_lat"])
            trajectory["target_lon"] = float(row["target_lon"])
            trajectory["current_time"] = int(row["current_time"])
            trajectory["current_lat"] = float(row["current_lat"])
            trajectory["current_lon"] = float(row["current_lon"])
            trajectories[counter] = trajectory
            counter += 1
    return trajectories
loss_list = []

def train() -> None:
    LOGGER.info("Initialize training environment and data")
    initialization_first_time = False
    # Define a simple neural network

    # Initialize the network
    net = NeuroNet()
    # Initialize a new network
    target_net = NeuroNet()

    if initialization_first_time == False:
        net.load_state_dict(torch.load('code/training_data/net_state_dict.pth'))
        #either model.eval() or model.train() depending on what we currently doing
        target_net.load_state_dict(torch.load('code/training_data/target_net_state_dict.pth'))
        #either model.eval() or model.train() depending on what we currently doing

    

    # Commonly used for classification problems
    optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    trajectories = import_trajectories()
    # Training loop
    for epoch_target in range(1):
        LOGGER.info(f"Epoch {epoch_target}")
        original_state_dict = net.state_dict() # Save the state dict of the original network
        # Load the state dict of the original network into the new network
        target_net.load_state_dict(original_state_dict)
        # Ensure that the new network is in the same mode (train or eval) as the original
        target_net.train(mode=net.training)
        
        training_data = random.sample(trajectories.keys(), len(trajectories) // 10)
        counter = 0
        net.train()
        target_net.train()
        for key in training_data:
            trajectory = trajectories[key]
            if counter % 100 == 0:
                target_net.load_state_dict(net.state_dict())
                scheduler.step()
            if counter % 50000 == 0:
                LOGGER.info(f"Processed {counter}/{len(training_data)} trajectories")
            optimizer.zero_grad()  # zero the parameter gradients
            output_target_net = target_net(torch.Tensor([trajectory["target_lat"], trajectory["target_lon"], trajectory["target_time"]]))
            # Forward pass
            LOGGER.debug("Forwardpropagation")
            output = net(torch.Tensor([trajectory["current_lat"], trajectory["current_lon"], trajectory["current_time"]]))

            # Compute loss
            LOGGER.debug("TD Loss")
            loss = td_error(output, output_target_net, trajectory["reward"])
            if counter % 1000 == 0:

                loss_list.append(loss.item())
            #loss = loss.pow(2)  # Squaring the TD error (if needed) I don`t want negative losses
            LOGGER.debug("Backpropagation")
            # Backward pass
            loss.backward()
            LOGGER.debug("Optimize")
            # Optimize
            optimizer.step()

            # Print statistics
            LOGGER.debug(f"Loss: {loss.item()}")
            counter += 1

    LOGGER.info('Finished Training')

    torch.save(net.state_dict(), 'code/training_data/net_state_dict.pth')
    torch.save(target_net.state_dict(), 'code/training_data/target_net_state_dict.pth')
