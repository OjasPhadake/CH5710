# Import necessary libraries
import numpy as np
import deepxde as dde

# Constants for the problem
rho = 1  # Density
mu = 1  # Viscosity
u_in = 1  # Inlet velocity
D = 1  # Channel width
L = 2  # Length of the channel

# Define boundary conditions and geometry

def boundary_wall(X, on_boundary):
    """Boundary condition for the walls (no-slip condition)"""
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -D/2), np.isclose(X[1], D/2)), on_boundary)
    return on_wall

def boundary_inlet(X, on_boundary):
    """Boundary condition for the inlet (inlet velocity specified)"""
    return on_boundary and np.isclose(X[0], -L/2)

def boundary_outlet(X, on_boundary):
    """Boundary condition for the outlet (pressure is zero)"""
    return on_boundary and np.isclose(X[0], L/2)

# Define the geometry of the system (rectangle)
geom = dde.geometry.Rectangle(xmin=[-L/2, -D/2], xmax=[L/2, D/2])

# Dirichlet boundary conditions for each part of the system
bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=0)  # No-slip condition for u velocity
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component=1)  # No-slip condition for v velocity
bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component=0)  # Inlet velocity u
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component=1)  # Inlet velocity v
bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=2)  # Zero pressure at outlet
bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component=1)  # Zero v velocity at outlet
