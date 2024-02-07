import torch.nn as nn

class NeuroNet(nn.Sequential):
    def __init__(self):
        super(NeuroNet, self).__init__(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)
        )