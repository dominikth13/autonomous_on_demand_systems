# This file contains the code for the offline policy evaluation (training with random samples of driver trajectories)

import math
import random
import torch
import torch.optim as optim
from deep_reinforcement_learning.neuro_net import NeuroNet
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from deep_reinforcement_learning.temporal_difference_loss import TemporalDifferenceLoss
from logger import LOGGER

def train_ope() -> None:
    LOGGER.info("Initialize environment")
    state_value_net = NeuroNet()
    target_net = NeuroNet()

    if True:
        state_value_net.load_state_dict(torch.load("code/training_data/ope_state_value_net_state_dict.pth"))
        target_net.load_state_dict(torch.load("code/training_data/ope_target_net_state_dict.pth"))

    trajectories = import_trajectories()
    test_trajectory_keys = set(random.Random(42).sample(trajectories.keys(), len(trajectories) // 100))
    train_trajectory_keys = list(filter(lambda t: t not in test_trajectory_keys, trajectories.keys()))

    # Used in Tang et al. (2021)
    # Loss function
    loss_fn = TemporalDifferenceLoss()
    # Optimizer
    optimizer = optim.Adam(
        state_value_net.parameters(), lr=3 * math.exp(-4)
    )  # Stochastic Gradient Descent
    for epoch in range(10):
        LOGGER.info(f"Epoch {epoch}")
        LOGGER.info("Start training")
        state_value_net.train()
        target_net.train()
        # Training loop
        N = 500
        for i in range(N):
            if i % 100 == 0:
                LOGGER.debug("Transfer weights from main to target network")
                target_net.load_state_dict(state_value_net.state_dict())
            LOGGER.debug(f"Training loop {i}")
            # Sample random batch of trajectories
            M = 100
            batch = random.Random(i).sample(train_trajectory_keys, k=M)

            LOGGER.debug("Forward propagation")
            state_values = []
            for key in batch:
                trajectory = trajectories[key]
                output_current = state_value_net(
                    torch.Tensor(
                        [
                            trajectory["current_lat"],
                            trajectory["current_lon"],
                            trajectory["current_time"],
                        ]
                    )
                )
                output_target = target_net(
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

        LOGGER.info("Start evaluation")
        state_value_net.eval()
        target_net.eval()
        for key in test_trajectory_keys:
            trajectory = trajectories[key]
            output_current = state_value_net(
                torch.Tensor(
                    [
                        trajectory["current_lat"],
                        trajectory["current_lon"],
                        trajectory["current_time"],
                    ]
                )
            )
            output_target = target_net(
                torch.Tensor(
                    [
                        trajectory["target_lat"],
                        trajectory["target_lon"],
                        trajectory["target_time"],
                    ]
                )
            )
            state_values.append((trajectory, output_current, output_target))

        # Compute loss
        loss = loss_fn(state_values)
        LOGGER.info(f"Temporal difference error: {float(loss)}")
        
    LOGGER.info('Finished Training')

    torch.save(state_value_net.state_dict(), "code/training_data/ope_state_value_net_state_dict.pth")
    torch.save(target_net.state_dict(), "code/training_data/ope_target_net_state_dict.pth")
