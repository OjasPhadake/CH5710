import numpy as np
from tensorflow.keras import models, layers, optimizers
from PINN_FS import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import FS_config
from error import l2norm_err

# ==============================================================================
# Data Loading Section
# ==============================================================================
# Load the dataset containing velocity components and pressure
data = np.load('data/Falkner_Skan_Ref_Data.npz')

# Extract data arrays
velocity_u = data['u'].T
velocity_v = data['v'].T
coordinates_x = data['x'].T
coordinates_y = data['y'].T
pressure_p = data['p'].T

# Normalize the x and y coordinates
coordinates_x -= coordinates_x.min()
coordinates_y -= coordinates_y.min()

# Combine velocity and pressure into reference data
reference_data = np.stack((velocity_u, velocity_v, pressure_p))

# ==============================================================================
# Training Parameters Section
# ==============================================================================
# Retrieve training configuration parameters
activation_function = FS_config.act
neurons_per_layer = FS_config.n_neural
num_layers = FS_config.n_layer
adam_steps = FS_config.n_adam
checkpoint_step = FS_config.cp_step
boundary_condition_step = FS_config.bc_step

# ==============================================================================
# Data Preparation Section
# ==============================================================================
# Collection points (sampling locations) at a specified step interval
collection_points = np.concatenate((coordinates_x[:, ::checkpoint_step].reshape((-1, 1)), 
                                    coordinates_y[:, ::checkpoint_step].reshape((-1, 1))), axis=1)
num_collection_points = len(collection_points)

# Boundary condition points (at the edges of the domain)
boundary_condition_flags = np.zeros(coordinates_x.shape, dtype=bool)
boundary_condition_flags[[0, -1], ::boundary_condition_step] = True
boundary_condition_flags[:, [0, -1]] = True

# Extract boundary condition coordinates and velocity components
boundary_x = coordinates_x[boundary_condition_flags].flatten()
boundary_y = coordinates_y[boundary_condition_flags].flatten()

boundary_u = velocity_u[boundary_condition_flags].flatten()
boundary_v = velocity_v[boundary_condition_flags].flatten()

# Combine boundary condition data
boundary_conditions = np.array([boundary_x, boundary_y, boundary_u, boundary_v]).T

# Parameters for boundary conditions and neural network
num_input_vars = 2
num_output_vars = boundary_conditions.shape[1] - num_input_vars + 1
pressure_output = 1

# Randomly select half of the boundary points for training
boundary_condition_indices = np.random.choice([False, True], len(boundary_conditions), p=[1 - pressure_output, pressure_output])
boundary_conditions = boundary_conditions[boundary_condition_indices]

num_boundary_conditions = len(boundary_conditions)
test_name = f'_{neurons_per_layer}_{num_layers}_{activation_function}_{adam_steps}_{num_collection_points}_{num_boundary_conditions}'

# ==============================================================================
# Model Compilation Section
# ==============================================================================
# Define input layer for the neural network model
input_layer = layers.Input(shape=(num_input_vars,))

# Define hidden layers of the neural network
hidden_layer = input_layer
for _ in range(num_layers):
    hidden_layer = layers.Dense(neurons_per_layer, activation=activation_function)(hidden_layer)

# Define output layer
output_layer = layers.Dense(num_output_vars)(hidden_layer)

# Create the model
model = models.Model(input_layer, output_layer)
print(model.summary())

# Set learning rate and optimizer
learning_rate = 1e-3
optimizer = optimizers.Adam(learning_rate)

# Initialize PINNs model with optimizer
pinn_model = PINNs(model, optimizer, adam_steps)

# ==============================================================================
# Training Section
# ==============================================================================
# Start training the model
print(f"INFO: Start training case : {test_name}")
start_time = time()

# Train the model using boundary conditions and collection points
history = pinn_model.fit(boundary_conditions, collection_points)

# Calculate training time
end_time = time()
training_time = end_time - start_time

# ==============================================================================
# Prediction Section
# ==============================================================================
# Create prediction points (flattened grid of x and y coordinates)
prediction_points = np.array([coordinates_x.flatten(), coordinates_y.flatten()]).T

# Get predictions from the model
predicted_data = pinn_model.predict(prediction_points)

# Reshape the predicted velocity components and pressure
predicted_u = predicted_data[:, 0].reshape(velocity_u.shape)
predicted_v = predicted_data[:, 1].reshape(velocity_u.shape)
predicted_p = predicted_data[:, 2].reshape(velocity_u.shape)

# Adjust the predicted pressure to match the reference pressure level
pressure_shift = pressure_p[0, 0] - predicted_p[0, 0]
predicted_p += pressure_shift

# Stack the predicted data (u, v, p)
predicted_data = np.stack((predicted_u, predicted_v, predicted_p))

# ==============================================================================
# Saving Section
# ==============================================================================
# Save the predicted data, reference data, and model history
np.savez_compressed('pred/res_FS' + test_name, pred=predicted_data, ref=reference_data, 
                    x=coordinates_x, y=coordinates_y, hist=history, err=l2norm_err, ct=training_time)

# Save the trained model
model.save('models/model_FS' + test_name + '.h5')

# Log the completion of the save process
print("INFO: Prediction and model have been saved!")
