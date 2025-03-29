# Import necessary libraries
import deepxde as dde
import numpy as np

# Define the governing PDEs for the system

def pde(X, Y):
    """
    Defines the PDEs for the velocity field and pressure field.

    Parameters:
    X: Coordinates of the points
    Y: Field values (u, v, p)

    Returns:
    List of residuals for each equation
    """
    # Gradients and Hessians for velocity and pressure components
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    du_xx = dde.grad.hessian(Y, X, i=0, j=0, component=0)
    du_yy = dde.grad.hessian(Y, X, i=1, j=1, component=0)
    dv_xx = dde.grad.hessian(Y, X, i=0, j=0, component=1)
    dv_yy = dde.grad.hessian(Y, X, i=1, j=1, component=1)

    # Equations for u, v, and continuity
    pde_u = Y[:, 0:1] * du_x + Y[:, 1:2] * du_y + 1 / rho * dp_x - (mu / rho) * (du_xx + du_yy)
    pde_v = Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y + 1 / rho * dp_y - (mu / rho) * (dv_xx + dv_yy)
    pde_cont = du_x + dv_y

    return [pde_u, pde_v, pde_cont]

# Create the PDE data
def create_data(geom):
    """Creates the PDE data for training the model."""
    data = dde.data.PDE(
        geom,
        pde,
        [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v],
        num_domain=2000,
        num_boundary=200,
        num_test=100
    )
    return data
