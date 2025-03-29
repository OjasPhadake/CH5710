import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt
import numpy as np
from train_configs import FS_config

class optimizer():
    """
    Optimizer class for minimizing a given function using a specified method (default is BFGS).
    Handles partitioning and stitching of trainable variables during optimization.
    """
    
    def __init__(self, trainable_vars, method=FS_config.method):
        """
        Initialize the optimizer with trainable variables and optimization method.

        Args:
        - trainable_vars (list): List of trainable variables to optimize.
        - method (str): The optimization method (default is BFGS).
        """
        super(optimizer, self).__init__()
        
        # Store the trainable variables and optimization method
        self.trainable_variables = trainable_vars
        self.method = method
        
        # Get the shapes of the trainable variables
        self.shapes = tf.shape_n(self.trainable_variables)
        self.num_tensors = len(self.shapes)

        # Initialize counters and lists for partitioning and stitching
        count = 0
        stitch_indices = []  # List to store indices for stitching tensors together
        partition_indices = []  # List to store partition indices
    
        # Loop over each tensor shape to create appropriate indices
        for i, shape in enumerate(self.shapes):
            num_elements = np.product(shape)
            stitch_indices.append(tf.reshape(tf.range(count, count + num_elements, dtype=tf.int32), shape))
            partition_indices.extend([i] * num_elements)
            count += num_elements
    
        # Convert partition indices list to a TensorFlow constant
        self.partition_indices = tf.constant(partition_indices)
        self.stitch_indices = stitch_indices
    
    def assign_params(self, params_1d):
        """
        Assign 1D parameters back to the trainable variables.

        Args:
        - params_1d (tf.Tensor): A 1D tensor containing the parameters to be assigned to trainable variables.
        """
        # Ensure the parameters are in the correct data type
        params_1d = tf.cast(params_1d, dtype=tf.float32)

        # Partition the parameters into the respective trainable variables
        partitioned_params = tf.dynamic_partition(params_1d, self.partition_indices, self.num_tensors)

        # Assign each partitioned parameter to the corresponding trainable variable
        for i, (shape, param) in enumerate(zip(self.shapes, partitioned_params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
    def minimize(self, func):
        """
        Minimize the given function using the specified optimization method.

        Args:
        - func (callable): The function to be minimized.
        
        Returns:
        - results: The optimization results from scipy.optimize.minimize.
        """
        # Combine the trainable variables into a single 1D tensor
        initial_params = tf.dynamic_stitch(self.stitch_indices, self.trainable_variables)

        # Perform the optimization using scipy's minimize function
        results = sopt.minimize(
            fun=func, 
            x0=initial_params, 
            method=self.method, 
            jac=True, 
            options={
                'iprint': 0,
                'maxiter': 50000,
                'maxfun': 50000,
                'maxcor': 50,
                'maxls': 50,
                'gtol': 1.0 * np.finfo(float).eps,
                'ftol': 1.0 * np.finfo(float).eps
            }
        )
        return results
