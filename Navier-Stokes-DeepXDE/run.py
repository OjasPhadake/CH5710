# Import the necessary setup and training files
from setup_environment import setup
from geometry_and_boundary_conditions import geom, bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v
from pde_and_data import create_data
from train_and_evaluate import random_search_and_train

# Main function to execute the entire pipeline
def main():
    # Step 1: Setup environment
    setup()

    # Step 2: Create PDE data
    data = create_data(geom)

    # Step 3: Train the model with random search
    random_search_and_train(data)

if __name__ == "__main__":
    main()
