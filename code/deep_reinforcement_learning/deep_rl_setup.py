import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from action.action import Action
from action.driver_action_pair import DriverActionPair
from algorithm.algorithm import generate_routes
from driver.driver import Driver
from driver.drivers import Drivers
from order import Order
from program_params import *
import pandas as pd
import torch.nn.functional as F

from route import Route
from logger import LOGGER



import torch.nn.functional as F


class NeuroNet(nn.Module):
    def __init__(self):
        super(NeuroNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)  # Assuming x is a scalar
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))

        return output


# Example usage:
# model = NeuroNet()
# output = model(input_x, input_y, input_time)


# Define a loss function and optimizer
def td_error(output_net, output_target_net, reward):
    loss = (reward + output_target_net - output_net) ** 2
    return loss