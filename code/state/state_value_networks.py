from __future__ import annotations
import math
from action.driver_action_pair import DriverActionPair
import torch
import torch.optim as optim
from deep_reinforcement_learning.neuro_net import NeuroNet
from deep_reinforcement_learning.temporal_difference_loss import TemporalDifferenceLoss
from interval.time import Time
from location.location import Location
from logger import LOGGER
from program.program_params import ProgramParams


class StateValueNetworks:
    _state_value_networks = None

    def get_instance() -> StateValueNetworks:
        if StateValueNetworks._state_value_networks == None:
            StateValueNetworks._state_value_networks = StateValueNetworks()
        return StateValueNetworks._state_value_networks

    def main_network() -> NeuroNet:
        return StateValueNetworks.get_instance().main_net

    def target_network() -> NeuroNet:
        return StateValueNetworks.get_instance().target_net

    def __init__(self) -> None:
        self.main_net = NeuroNet()
        self.target_net = NeuroNet()
        self.main_net.train()
        self.target_net.train()

        self.loss_fn = TemporalDifferenceLoss()
        # Optimizer
        self.optimizer = optim.Adam(
            self.main_net.parameters(), lr=3 * math.exp(-4)
        )  # Stochastic Gradient Descent

        self.iteration = 1

    def get_main_state_value(self, location: Location, time: Time) -> float:
        return float(
            self.main_net(
                torch.Tensor([location.lat, location.lon, time.to_total_seconds()])
            ).item()
        )

    def get_target_state_value(self, location: Location, time: Time) -> float:
        return float(
            self.target_net(
                torch.Tensor([location.lat, location.lon, time.to_total_seconds()])
            ).item()
        )

    # We want a list of action tuples here since the error function is calculated in each iteration for all changes
    def adjust_state_values(self, action_tuples: list[tuple]) -> None:
        trajectories = []
        for tup in action_tuples:
            trajectories.append(
                {
                    "reward": tup[0],
                    "current_lat": tup[1].lat,
                    "current_lon": tup[1].lon,
                    "current_time": tup[2].to_total_seconds(),
                    "target_lat": tup[3].lat,
                    "target_lon": tup[3].lon,
                    "target_time": tup[4].to_total_seconds(),
                }
            )
        LOGGER.debug("Adjust weights for deep state value networks")
        if self.iteration % ProgramParams.MAIN_AND_TARGET_NET_SYNC_ITERATIONS == 0:
            LOGGER.debug("Transfer weights from main to target network")
            self.target_net.load_state_dict(self.main_net.state_dict())

        LOGGER.debug("Forward propagation")
        state_values = []
        for trajectory in trajectories:
            output_main = self.main_net(
                torch.Tensor(
                    [
                        trajectory["current_lat"],
                        trajectory["current_lon"],
                        trajectory["current_time"],
                    ]
                )
            )
            output_target = self.target_net(
                torch.Tensor(
                    [
                        trajectory["target_lat"],
                        trajectory["target_lon"],
                        trajectory["target_time"],
                    ]
                )
            )
            state_values.append((trajectory, output_main, output_target))

        LOGGER.debug("Backward propagation and optimization")
        # Backward and optimize
        self.optimizer.zero_grad()
        # Compute loss
        loss = self.loss_fn(state_values)
        LOGGER.debug(f"Temporal difference error: {float(loss)}")
        loss.backward()
        self.optimizer.step()

    def import_weights(self) -> None:
        self.main_net.load_state_dict(
            torch.load("code/training_data/main_net_state_dict.pth")
        )
        self.target_net.load_state_dict(
            torch.load("code/training_data/target_net_state_dict.pth")
        )

    def export_weights(self) -> None:
        torch.save(
            self.main_net.state_dict(), "code/training_data/main_net_state_dict.pth"
        )
        torch.save(
            self.target_net.state_dict(), "code/training_data/target_net_state_dict.pth"
        )

    def load_offline_policy_weights(self, current_total_minutes: int) -> None:
        daystr = ""
        wd = ProgramParams.SIMULATION_DATE.weekday()
        if wd < 5:
            daystr = "wd"
        elif wd == 5:
            daystr = "sat"
        else:
            daystr = "sun"

        ope_net = NeuroNet()
        ope_target_net = NeuroNet()
        ope_optimizer = optim.Adam(self.main_net.parameters(), lr=3 * math.exp(-4))
        ope_net.load_state_dict(
            torch.load(f"code/training_data/ope_{daystr}_{current_total_minutes}.pth")
        )
        ope_target_net.load_state_dict(
            torch.load(
                f"code/training_data/ope_target_{daystr}_{current_total_minutes}.pth"
            )
        )
        ope_optimizer.load_state_dict(
            torch.load(
                f"code/training_data/ope_opt_{daystr}_{current_total_minutes}.pth"
            )
        )
        ope_state = ope_net.state_dict()
        ope_target_state = ope_target_net.state_dict()
        ope_optimizer_state = ope_optimizer.state_dict()
        main_state = self.main_net.state_dict()
        target_state = self.target_net.state_dict()
        optimizer_state = self.optimizer.state_dict()

        new_target_state = {}
        new_main_state = {}
        new_optimizer_state = {}

        for key in main_state:
            new_main_state[key] = (
                ProgramParams.OMEGA * main_state[key]
                + (1 - ProgramParams.OMEGA) * ope_state[key]
            )
        for key in target_state:
            new_target_state[key] = (
                ProgramParams.OMEGA * target_state[key]
                + (1 - ProgramParams.OMEGA) * ope_target_state[key]
            )
        for key in optimizer_state:
            new_optimizer_state[key] = (
                ProgramParams.OMEGA * optimizer_state[key]
                + (1 - ProgramParams.OMEGA) * ope_optimizer_state[key]
            )

        self.main_net.load_state_dict(new_main_state)
        self.target_net.load_state_dict(new_target_state)
        self.optimizer.load_state_dict(new_optimizer_state)
