# ------------------------------------------------------
# CONDITIONS MODULE
# Includes initial and boundary conditions
# ------------------------------------------------------

import torch
import numpy as np

def initial_condition(x):
    return -torch.sin(np.pi * x)

def boundary_condition(x, t):
    return torch.zeros_like(t)
