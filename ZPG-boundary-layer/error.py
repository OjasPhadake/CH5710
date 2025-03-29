import numpy as np

def l2norm_err(ref, pred):
    """
    Calculate the relative L2-norm of errors for all computational points 
    for each variable.

    Args:
        ref (np.ndarray): Reference values of variables. 
                          Shape should be (N, ..., I), where 
                          N = number of points, 
                          I = number of variables.
                          
        pred (np.ndarray): Predicted values computed by a neural network. 
                           Shape should match that of 'ref'.

    Returns:
        np.ndarray: An array of relative L2-norm errors with shape (N, I), 
                    representing the error for each variable at each point.
    """
    # Calculate the relative L2-norm of the error
    relative_error = (np.linalg.norm(ref - pred, axis=(1, 2)) / 
                      np.linalg.norm(ref, axis=(1, 2))) * 100
    return relative_error
