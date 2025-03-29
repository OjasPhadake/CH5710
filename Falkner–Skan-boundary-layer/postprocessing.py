import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import FS_config

# Set color map for the contour plots
color_map = sns.color_palette('RdBu_r', as_cmap=True)
plt.set_cmap(color_map)

# Extract configuration values for neural network setup
activation_function = FS_config.act
num_neurons = FS_config.n_neural
num_layers = FS_config.n_layer
adam_optimizer_params = FS_config.n_adam

# Generate the data file name based on the model configuration
data_filename = f"FS_{num_neurons}_{num_layers}_{activation_function}_{adam_optimizer_params}_1206_500"

# Load prediction data and reference data
prediction_data = np.load(f"pred/res_{data_filename}.npz")
reference_data = np.load('data/Falkner_Skan_Ref_Data.npz')

# Extract grid values for x and y coordinates
x_coordinates = reference_data['x'].T
y_coordinates = reference_data['y'].T

# Plot loss versus epoch data
plt.semilogy(prediction_data["hist"][:, 0], label="Total Loss")
plt.semilogy(prediction_data["hist"][:, 1], label="Boundary Condition Loss")
plt.semilogy(prediction_data["hist"][:, 2], label="Residual Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"figs/{data_filename}_Loss")

# Extract predicted values and reference values for U, V, P components
predicted_values = prediction_data["pred"]
reference_values = prediction_data["ref"]
component_names = ["U", "V", "P"]

# Generate contour plots for U, V, and P components
for index, component in enumerate(component_names):
    fig, axes = plt.subplots(3, figsize=(9, 10))
    
    # Plot predicted values for the component
    axes[0].contourf(x_coordinates, y_coordinates, predicted_values[index, :, :])
    # Plot reference values for the component
    axes[1].contourf(x_coordinates, y_coordinates, reference_values[index, :, :])
    # Plot absolute error between reference and predicted values
    error_contour = axes[2].contourf(x_coordinates, y_coordinates, np.abs(reference_values[index, :, :] - predicted_values[index, :, :]))
    
    # Add color bar for the error contour plot
    colorbar = plt.colorbar(error_contour, orientation="horizontal", pad=0.3)
    axes[0].set_title(f"{component} Prediction", fontdict={"size": 18})
    axes[1].set_title(f"{component} Reference", fontdict={"size": 18})
    axes[2].set_title(f"{component} Error", fontdict={"size": 18})
    
    # Adjust color bar settings
    colorbar.ax.locator_params(nbins=5)
    colorbar.ax.tick_params(labelsize=12)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"figs/{data_filename}_{component}", dpi=300)

# Calculate L2-norm errors for U, V, and P components
error_values = l2norm_err(reference_values, predicted_values)

# Print the error percentages for U, V, and P components
print(f"Error U = {np.round(error_values[0], 3)}%")
print(f"Error V = {np.round(error_values[1], 3)}%")
print(f"Error P = {np.round(error_values[2], 3)}%")
