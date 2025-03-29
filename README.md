# CH5710_COURSE_PROJECT

###  :file_folder: Burger's Equation for unsteady state case

```
Burgers-Equation-PINN/
│
├── data/                                # Directory for data handling modules
│   ├── data_loader.py                   # Loads reference data for the equation
│   ├── geometry.py                      # Generates collocation and boundary points
│   └── visualization.py                 # Functions for visualization of solutions and geometry
│
├── model/                               # Directory for model architecture and PDE residual
│   ├── pinn.py                          # Defines the PINN architecture
│   └── residual.py                      # Computes the PDE residual for Burgers' equation
│
├── training/                            # Directory for training and optimization
│   ├── conditions.py                    # Defines initial and boundary conditions
│   ├── optimizers.py                    # Handles optimizer selection
│   └── trainer.py                       # Implements the training loop
│
├── main.py                              # Main script to execute the pipeline
├── Burgers.npz                          # Dataset for reference solution
└── README.md                            # Project documentation
```

###  :file_folder: 2D Navier–Stokes equations for Flow between Infinite Parallel Plate 
```
Navier-Stokes-DeepXDE/
│
├── setup_environment.py                 # Script to install libraries and set up environment
├── geometry_and_boundary_conditions.py  # Defines the geometry and boundary conditions
├── pde_and_data.py                      # Defines the PDEs and prepares the data for training
├── train_and_evaluate.py                # Handles model training, evaluation, and saving
├── run.py                               # Main script to execute the pipeline
├── Navier Stokes Models/                # Directory to store trained models
├── Navier Stokes Plots/                 # Directory to store loss plots
└── README.md                            # Project documentation
```

###  :file_folder: 2D Navier–Stokes equations for the Falkner–Skan boundary
```
Falkner–Skan-boundary-layer/ 
│
├── data/                                # Contains reference data
│   └── Falkner_Skan_Ref_Data.npz        # Reference data for the model
│
├── lbfgs.py                             # Python script for L-BFGS method 
├── PINN_FS.py                           # Python script for PINN (Physics-Informed Neural Networks) FS 
├── postprocessing.py                    # Python script for post-processing results 
├── train.py                             # Python script for model training 
├── train_configs.py                     # Python script for training configurations 
└── README.md                            # Project documentation
```

###  :file_folder: 2D Navier–Stokes equations for the ZPG turbulent boundary layer
```
ZPG-boundary-layer/ 
│
├── data/                                # Contains reference data
│   └── ZPG.mat                          # Reference data for the model
│
├── lbfgs.py                             # Python script for L-BFGS method 
├── PINN_ZPG.py                          # Python script for PINN ZPG
├── postprocessing.py                    # Python script for post-processing results 
├── train.py                             # Python script for model training 
├── train_configs.py                     # Python script for training configurations 
└── README.md                            # Project documentation
```




