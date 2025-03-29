import numpy as np

def l2norm_err(ref, pred):
    """
    Calculate the relative L2-norm of errors E_i on all computational points for i-th variable.
    
    This function computes the relative L2-norm error between the reference values and the 
    predicted values across all data points and variables. It returns the error in percentage.

    Args:
        ref (np.ndarray): Reference values of variables. Shape: [N, I, ...]
                          where N is the number of points and I is the number of variables.
                          
        pred (np.ndarray): Predicted values computed by Neural Networks. Same shape as ref.

    Returns:
        np.ndarray: An array of relative L2-norm errors for each point and each variable. 
                    Shape: [N, I] where N is the number of points and I is the number of variables.

    """
    # Calculate the L2-norm of the errors (difference between reference and prediction)
    error_norm = np.linalg.norm(ref - pred, axis=(1, 2))
    
    # Calculate the L2-norm of the reference values
    ref_norm = np.linalg.norm(ref, axis=(1, 2))
    
    # Compute the relative L2-norm error in percentage
    relative_error = (error_norm / ref_norm) * 100
    
    return relative_error
