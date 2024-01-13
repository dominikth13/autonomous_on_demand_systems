import torch.nn as nn

class NeuroNet(nn.Sequential):
    def __init__(self):
        super(NeuroNet, self).__init__(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)
        )

def td_error(output_net, output_target_net, reward):
    loss = (reward + output_target_net - output_net) ** 2
    return loss
