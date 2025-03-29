# ------------------------------------------------------
# VISUALIZATION MODULE
# Includes functions for plotting solutions and geometry
# ------------------------------------------------------

import matplotlib.pyplot as plt

def visualize_reference_solution(x_ref, t_ref, exact):
    """
    Plots the reference solution as a contour plot.

    Args:
        x_ref (np.ndarray): Space meshgrid.
        t_ref (np.ndarray): Time meshgrid.
        exact (np.ndarray): Solution values.
    """
    plt.figure(figsize=(16, 4))
    plt.contourf(x_ref, t_ref, exact, levels=250, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Reference Solution of Burgers' Equation")
    plt.show()

def visualize_geometry(points):
    """
    Plots the collocation and boundary points.

    Args:
        points (dict): Dictionary containing geometry points.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(points["x_collocation"].numpy(), points["t_collocation"].numpy(), label='Domain Points', color='blue', s=10)
    plt.scatter(points["x_initial_condition"].numpy(), points["t_initial_condition"].numpy(), label='Initial Condition Points', color='green', s=30)
    plt.scatter(points["x_boundary_left"].numpy(), points["t_boundary_points"].numpy(), label='Left Boundary Points', color='red', s=30)
    plt.scatter(points["x_boundary_right"].numpy(), points["t_boundary_points"].numpy(), label='Right Boundary Points', color='orange', s=30)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Collocation Points in the Domain and on the Boundaries')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
