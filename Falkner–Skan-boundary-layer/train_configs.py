class FS_config:
    """
    Train configuration for Falkner-Skan boundary layer problem.

    This class holds the configuration parameters used in the training process for 
    solving the Falkner-Skan boundary layer equations using a neural network.

    Args:
        act (str): Activation function used for the MLP (e.g., "tanh", "relu").
        
        n_adam (int): Number of steps used for supervised training (number of iterations).
        
        n_neural (int): Number of neurons in each MLP layer (N_h).
        
        n_layer (int): Total number of layers in the MLP model (N_l).
        
        cp_step (int): Length of the interval to collect collection points (N_e).
        
        bc_step (int): Length of the interval to collect points on the domain boundaries (N_b).
        
        method (str): Optimization method used for training (e.g., "L-BFGS-B").
    """
    
    # Activation function used in the MLP
    act = "tanh"
    
    # Number of steps used for supervised training
    n_adam = 1000
    
    # Number of neurons in each MLP layer
    n_neural = 20
    
    # Total number of layers used in the MLP model
    n_layer = 8
    
    # Length of interval to collect collection points (N_e)
    cp_step = 100
    
    # Length of interval to collect points on the domain boundaries (N_b)
    bc_step = 10
    
    # Optimization method used for training
    method = "L-BFGS-B"
