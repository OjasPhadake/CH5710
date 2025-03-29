# ------------------------------------------------------
# OPTIMIZER MODULE
# Provides various optimizers for training
# ------------------------------------------------------

import torch

def get_optimizer(optimizer_name, model_parameters, learning_rate):
    """
    Selects an optimizer based on the provided name.

    Args:
        optimizer_name (str): Name of the optimizer.
        model_parameters: Model parameters for optimization.
        learning_rate (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Selected optimizer.
    """
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'AdamW': torch.optim.AdamW
    }
    return optimizers.get(optimizer_name, torch.optim.Adam)(model_parameters, lr=learning_rate)
