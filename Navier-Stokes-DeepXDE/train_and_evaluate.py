# Import necessary libraries
import deepxde as dde
import random
import shutil
import os
import matplotlib.pyplot as plt

# Random hyperparameter search and model training
def random_search_and_train(data):
    """
    Performs random search for hyperparameters and trains models.
    
    Parameters:
    data: PDE data for the training
    
    Returns:
    None
    """
    layer_sizes = [
        [2] + [64] * 3 + [3],
        [2] + [128] * 3 + [3],
        [2] + [64] * 4 + [3],
        [2] + [128] * 4 + [3],
        [2] + [64] * 5 + [3],
        [2] + [128] * 5 + [3]
    ]
    activations = ["tanh", "relu"]
    initializers = ["Glorot uniform", "He normal"]
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-3, 1e-4]

    # Number of random configurations to try
    num_trials = 40  # You can adjust this value

    for _ in range(num_trials):
        # Sample a random configuration
        layer_size = random.choice(layer_sizes)
        activation = random.choice(activations)
        initializer = random.choice(initializers)
        optimizer = random.choice(optimizers)
        lr = random.choice(learning_rates)

        # Neural network setup
        net = dde.maps.FNN(layer_size, activation, initializer)
        model = dde.Model(data, net)
        model.compile(optimizer, lr=lr)

        # Train the model with early stopping
        losshistory, train_state = model.train(epochs=5000, display_every=10)

        # Prepare configuration name to display in the plot title
        config_name = f"Layers_{layer_size}_Activation_{activation}_Init_{initializer}_Optimizer_{optimizer}_LR_{lr}"

        # Save the trained model
        model_save_path = os.path.join("Navier Stokes Models", f"{config_name}.h5")
        model.save(model_save_path)  # Save the model in the specified folder

        # Save the plot with the configuration name
        plot_save_path = os.path.join("Navier Stokes Plots", f"{config_name}_loss_plot.png")
        deepxde.utils.external.plot_loss_history(losshistory, fname=plot_save_path)
        
        print(f"Model trained with configuration: {config_name}")
