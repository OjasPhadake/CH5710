# ------------------------------------------------------
# DATA LOADING MODULE
# Loads reference data for Burgers' equation
# ------------------------------------------------------

import numpy as np

def load_reference_data(file_path):
    """
    Loads reference data for the Burgers' equation.

    Args:
        file_path (str): Path to the .npz file containing reference data.

    Returns:
        tuple: t_ref (time meshgrid), x_ref (space meshgrid), exact (solution values).
    """
    data = np.load(file_path)
    t_ref, x_ref, exact = data["t"], data["x"], data["usol"].T
    x_ref, t_ref = np.meshgrid(x_ref, t_ref)
    return t_ref, x_ref, exact
