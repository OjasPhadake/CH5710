# ------------------------------------------------------
# PDE RESIDUAL MODULE
# Computes the residual for Burgers' equation
# ------------------------------------------------------

import torch

def pde_residual(x, t, model, nu=0.01):
    """
    Computes the PDE residual for the Burgers' equation.

    Args:
        x (torch.Tensor): Spatial points.
        t (torch.Tensor): Temporal points.
        model (nn.Module): PINN model.
        nu (float): Viscosity parameter.

    Returns:
        torch.Tensor: Residual values.
    """
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx
