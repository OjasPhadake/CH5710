# ------------------------------------------------------
# PINN MODEL MODULE
# Defines the Physics-Informed Neural Network
# ------------------------------------------------------

import torch.nn as nn

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.
    """
    def __init__(self, num_hidden_layers, num_neurons):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.Tanh()]
        layers.append(nn.Linear(num_neurons, 1))
        self.hidden = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.hidden(inputs)
