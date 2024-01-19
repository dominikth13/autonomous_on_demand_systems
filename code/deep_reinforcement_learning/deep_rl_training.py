import csv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from deep_reinforcement_learning.neuro_net import NeuroNet
from driver.driver import Driver
from driver.drivers import Drivers
from grid.grid import Grid
from torch.optim.lr_scheduler import StepLR
from interval.time import Time
from logger import LOGGER

def import_trajectories(train_mode: str, time_series_breakpoints: list[int]) -> dict[int, dict[int, dict[str, float]]]:
    csv_file_names = []
    if train_mode == "wd":
        csv_file_names = ["trajectories_thu_week_28.csv", "trajectories_fri_week_28.csv"]
    elif train_mode == "sat":
        csv_file_names = ["trajectories_sat_week_28.csv"]
    elif train_mode == "sun":
        csv_file_names = ["trajectories_sun_week_28.csv"]
    trajectories = {bp: {} for bp in time_series_breakpoints}
    for name in csv_file_names:
        csv_file_path = f"code/trajectories/{name}"
        with open(csv_file_path, mode="r") as file:
            reader = csv.DictReader(file)
            for counter, row in enumerate(reader):
                total_minutes = Time.of_total_seconds(int(row["current_time"])).to_total_minutes()
                bp_minutes = 0
                for bp in time_series_breakpoints:
                    if bp > total_minutes:
                        break
                    bp_minutes = bp

                trajectory = {}
                trajectory["reward"] = float(row["reward"])
                trajectory["target_time"] = int(row["target_time"])
                trajectory["target_lat"] = float(row["target_lat"])
                trajectory["target_lon"] = float(row["target_lon"])
                trajectory["current_time"] = int(row["current_time"])
                trajectory["current_lat"] = float(row["current_lat"])
                trajectory["current_lon"] = float(row["current_lon"])
                trajectories[bp_minutes][counter] = trajectory

    return trajectories