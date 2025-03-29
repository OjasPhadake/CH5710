# ------------------------------------------------------
# GEOMETRY MODULE
# Generates collocation, boundary, and initial condition points
# ------------------------------------------------------

import torch

def create_geometry():
    """
    Generates collocation, boundary, and initial condition points.

    Returns:
        dict: A dictionary containing tensors for different point types.
    """
    x_values = torch.linspace(-1, 1, 25).view(-1, 1)
    t_values = torch.linspace(0, 1, 25).view(-1, 1)

    x_collocation, t_collocation = torch.meshgrid(x_values.squeeze(), t_values.squeeze(), indexing='xy')
    x_collocation = x_collocation.reshape(-1, 1)
    t_collocation = t_collocation.reshape(-1, 1)

    x_boundary_left = torch.full_like(t_values, -1)
    x_boundary_right = torch.full_like(t_values, 1)
    t_boundary_points = t_values

    t_initial_condition = torch.zeros_like(x_values)
    x_initial_condition = x_values

    return {
        "x_collocation": x_collocation,
        "t_collocation": t_collocation,
        "x_boundary_left": x_boundary_left,
        "x_boundary_right": x_boundary_right,
        "t_boundary_points": t_boundary_points,
        "x_initial_condition": x_initial_condition,
        "t_initial_condition": t_initial_condition
    }
