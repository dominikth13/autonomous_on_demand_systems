import torch.nn as nn

class NeuroNet(nn.Sequential):
    def __init__(self):
        super(NeuroNet, self).__init__(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1)
        )