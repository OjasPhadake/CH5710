import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt
import numpy as np
from train_configs import ZPG_config

class Optimizer:
    """
    A custom optimizer class that supports variable stitching and partitioning 
    for optimization using SciPy's optimization methods.
    """
    
    def __init__(self, trainable_vars, method=ZPG_config.method):
        """
        Initialize the optimizer with trainable variables and a specified method.

        Args:
            trainable_vars (list): List of trainable TensorFlow variables.
            method (str): Optimization method (default from ZPG_config).
        """
        super(Optimizer, self).__init__()
        self.trainable_variables = trainable_vars
        self.method = method

        # ---- Shape Extraction ----
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        # ---- Indexing for Stitching and Partitioning ----
        count = 0
        idx = []  # Indices for stitching
        part = []  # Partition labels for variables

        for i, shape in enumerate(self.shapes):
            n = np.product(shape)  # Number of elements in the tensor
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        # Save partition and index data
        self.part = tf.constant(part)
        self.idx = idx

    def assign_params(self, params_1d):
        """
        Assign a 1D parameter vector back to the trainable variables.

        Args:
            params_1d (tf.Tensor): Flattened parameter vector.
        """
        params_1d = tf.cast(params_1d, dtype=tf.float32)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)

        # Assign reshaped parameters back to the trainable variables
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))

    def minimize(self, func):
        """
        Minimize a given function using the SciPy optimizer.

        Args:
            func (callable): The objective function to minimize. Must return 
                             both function value and gradient.
        """
        # ---- Initialization ----
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)

        # ---- Optimization using SciPy ----
        results = sopt.minimize(
            fun=func,
            x0=init_params,
            method=self.method,
            jac=True,
            options={
                'iprint': 0,
                'maxiter': 50000,
                'maxfun': 50000,
                'maxcor': 50,
                'maxls': 50,
                'gtol': 1.0 * np.finfo(float).eps,
                'ftol': 1.0 * np.finfo(float).eps,
            }
        )
