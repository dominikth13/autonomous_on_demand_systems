# This file contains the code for the offline policy evaluation (training with random samples of driver trajectories)

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from deep_reinforcement_learning.deep_rl_setup import NeuroNet
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from logger import LOGGER
from program_params import DISCOUNT_FACTOR


class TemporalDifferenceLoss(nn.Module):
    def __init__(self):
        super(TemporalDifferenceLoss, self).__init__()

    def forward(self, trajectories_and_state_values):
        loss = 0
        for x in trajectories_and_state_values:
            start_state_value = x[1]
            end_state_value = x[2]
            start_t = x[0]["current_time"]
            end_t = x[0]["target_time"]
            duration = end_t - start_t
            reward = x[0]["reward"]
            loss += (
                (
                    (reward * (DISCOUNT_FACTOR(duration) - 1))
                    / (duration * (DISCOUNT_FACTOR(1) - 1))
                )
                + DISCOUNT_FACTOR(duration) * end_state_value
                - start_state_value
            ) ** 2
        
        # TODO find out what this Lipschitz constant is
        return loss #+ math.exp(-4) * 


def train_ope() -> None:
    LOGGER.info("Initialize environment")
    ope_net = NeuroNet()

    if False:
        ope_net.load_state_dict(torch.load('net_state_dict.pth'))

    trajectories = import_trajectories()

    # Used in Tang et al. (2021)
    optimizer = optim.Adam(
        ope_net.parameters(), lr=3 * math.exp(-4)
    )  # Stochastic Gradient Descent
    # Loss function
    loss_fn = TemporalDifferenceLoss()

    LOGGER.info("Start training")
    # Training loop
    N = 100
    for i in range(N):
        LOGGER.debug(f"Training loop {i}")
        # Sample random batch of trajectories
        M = 100
        batch = random.Random(i).sample(trajectories, k=M)

        LOGGER.debug("Forward propagation")
        state_values = []
        for trajectory in batch:
            output_current = ope_net(
                torch.Tensor(
                    [
                        trajectory["current_lat"],
                        trajectory["current_lon"],
                        trajectory["current_time"],
                    ]
                )
            )
            output_target = ope_net(
                torch.Tensor(
                    [
                        trajectory["target_lat"],
                        trajectory["target_lon"],
                        trajectory["target_time"],
                    ]
                )
            )
            state_values.append((trajectory, output_current, output_target))

        LOGGER.debug("Backward propagation and optimization")
        # Backward and optimize
        optimizer.zero_grad()
        # Compute loss
        loss = loss_fn(state_values)
        loss.backward()
        optimizer.step()
    LOGGER.info('Finished Training')

    torch.save(ope_net.state_dict(), "code/data/ope_net_state_dict.pth")
