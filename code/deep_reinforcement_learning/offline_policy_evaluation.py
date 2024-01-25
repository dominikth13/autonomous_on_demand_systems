# This file contains the code for the offline policy evaluation (training with random samples of driver trajectories)

import csv
import math
import random
from numpy import mean
import torch
import torch.optim as optim
from deep_reinforcement_learning.neuro_net import NeuroNet
from deep_reinforcement_learning.deep_rl_training import import_trajectories
from deep_reinforcement_learning.temporal_difference_loss import TemporalDifferenceLoss
from logger import LOGGER
from program.program_params import ProgramParams


def train_ope() -> None:
    # wd, sat or sun
    TRAINING_MODE = "wd"
    time_series_breakpoints = {
        "wd": [0, 150, 300, 450, 1050, 1350],
        "sat": [0, 150, 300, 450, 750, 1050],
        "sun": [0, 150, 300, 450, 600, 1350],
    }
    LOGGER.info("Initialize environment")
    trajectories_by_bp = import_trajectories(
        TRAINING_MODE, time_series_breakpoints[TRAINING_MODE]
    )
    all_losses = []
    for bp in trajectories_by_bp:
        all_losses.append(train_for(TRAINING_MODE, bp, trajectories_by_bp[bp]))

    LOGGER.info("Finished Training")

    with open(f"code/training_data/training_{TRAINING_MODE}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["break_point", "losses"])
        for i, bp in enumerate(time_series_breakpoints[TRAINING_MODE]):
            writer.writerow([bp, all_losses[i]])


def train_for(
    train_mode: str,
    time_series_breakpoint_group: int,
    trajectories: dict[int, dict[str, float]],
) -> list[float]:
    state_value_net = NeuroNet()
    target_net = NeuroNet()
    keys = list(trajectories.keys())
    random.Random(time_series_breakpoint_group).shuffle(keys)

    # Berechnen der Größe jedes Teilsets
    size, extra = divmod(len(keys), 11)

    # Aufteilung in Teilsets
    all_groups = [
        keys[i * size + min(i, extra) : (i + 1) * size + min(i + 1, extra)]
        for i in range(11)
    ]
    epoch_groups = all_groups[:-1]
    test_group = all_groups[-1]

    # Used in Tang et al. (2021)
    # Loss function
    loss_fn = TemporalDifferenceLoss()
    # Optimizer
    optimizer = optim.Adam(
        state_value_net.parameters(), lr=3 * math.exp(-4)
    )  # Stochastic Gradient Descent
    all_losses = []
    for epoch, group in enumerate(epoch_groups):
        LOGGER.info(f"Epoch {epoch}")
        LOGGER.info("Start training")
        state_value_net.train()
        target_net.train()
        # Training loop
        M = 100
        N = len(group) // M
        for i in range(N):
            if i % 100 == 0:
                LOGGER.debug("Transfer weights from main to target network")
                target_net.load_state_dict(state_value_net.state_dict())
            LOGGER.debug(f"Training loop {i}")
            # Sample random batch of trajectories
            batch = random.Random(epoch + i**epoch).sample(group, k=M)

            LOGGER.debug("Forward propagation")
            state_values = []
            for key in batch:
                trajectory = trajectories[key]
                output_current = state_value_net(
                    torch.Tensor(
                        [
                            trajectory["current_zone"],
                        ]
                    )
                )
                output_target = target_net(
                    torch.Tensor(
                        [
                            trajectory["target_zone"],
                        ]
                    )
                )
                state_values.append((trajectory, output_current, output_target))

            LOGGER.debug("Backward propagation and optimization")
            # Backward and optimize
            optimizer.zero_grad()
            # Compute loss
            loss = loss_fn(state_value_net, state_values)
            loss.backward()
            optimizer.step()

        LOGGER.info("Start evaluation")
        state_value_net.eval()
        target_net.eval()
        losses = []
        N = len(test_group) // M // 10
        for i in range(N):
            # Sample random batch of trajectories
            batch = random.Random(epoch + i**epoch).sample(test_group, k=M)
            for key in batch:
                trajectory = trajectories[key]
                output_current = state_value_net(
                    torch.Tensor(
                        [
                            trajectory["current_zone"],
                        ]
                    )
                )
                output_target = target_net(
                    torch.Tensor(
                        [
                            trajectory["target_zone"],
                        ]
                    )
                )
                state_values.append((trajectory, output_current, output_target))

            # Compute loss
            losses.append(loss_fn(state_value_net, state_values).item())

        LOGGER.info(f"Medium difference error: {float(mean(losses))}")
        all_losses.append(float(mean(losses)))
    torch.save(
        state_value_net.state_dict(),
        f"code/training_data/ope_{train_mode}_{time_series_breakpoint_group}.pth",
    )
    torch.save(
        target_net.state_dict(),
        f"code/training_data/ope_target_{train_mode}_{time_series_breakpoint_group}.pth",
    )
    LOGGER.info(
        f"Finished Training for breakpoint {time_series_breakpoint_group} on mode {train_mode}"
    )

    return all_losses
