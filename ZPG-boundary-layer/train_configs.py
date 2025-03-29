# ==========================
# ZPG Configuration Class
# ==========================
class ZPG_config:
    """
    Configuration for training the ZPG Boundary Layer (BL) model.

    Attributes:
        act (str): Activation function used for the Multi-Layer Perceptron (MLP).
        n_adam (int): Number of steps used for supervised training.
        n_neural (int): Hidden dimension for each MLP layer (denoted as N_h).
        n_layer (int): Total number of MLP layers used in the model (denoted as N_l).
        cp_step (int): Interval length for collecting collection points (denoted as N_e).
        method (str): Optimizer used for unsupervised training.
    """

    # ==========================
    # Default Configurations
    # ==========================
    act = "tanh"          # Activation function
    n_adam = 1000         # Steps for supervised training
    n_neural = 20         # Hidden dimension for MLP layers
    n_layer = 8           # Total number of MLP layers
    cp_step = 500         # Interval for collection points
    method = "L-BFGS-B"   # Optimizer for unsupervised training
