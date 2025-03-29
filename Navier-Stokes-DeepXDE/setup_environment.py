# Import required libraries for setup
import os
import shutil

# Function to install necessary libraries (if not already installed)
def install_libraries():
    # Install deepxde and tensorflow
    os.system('pip install deepxde tensorflow')

# Set the backend for DeepXDE (TensorFlow)
def set_deepxde_backend():
    from deepxde.backend.set_default_backend import set_default_backend
    set_default_backend("tensorflow")

# Function to delete a folder if it exists
def delete_folder(folder_path):
    """Delete folder and its contents if it exists"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Delete the entire folder and its contents
        print(f"Deleted folder: {folder_path}")

# Create directories for saving models and plots
def create_folders():
    """Create necessary directories if they don't exist"""
    # Folders for saving models and plots
    os.makedirs("Navier Stokes Models", exist_ok=True)
    os.makedirs("Navier Stokes Plots", exist_ok=True)

# Main function to execute setup tasks
def setup():
    install_libraries()  # Install necessary libraries
    set_deepxde_backend()  # Set the DeepXDE backend to TensorFlow
    create_folders()  # Create folders for models and plots

if __name__ == "__main__":
    setup()