#%% Imports and Setup
# ================================
# Libraries and Configurations
# ================================
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from error import l2norm_err
from train_configs import ZPG_config

"""
Postprocessing of prediction data.

Generates:
    1. Loss vs Epoch plot
    2. Contour plots for U, V, and uv (Prediction, Reference, Error)
    3. Prints L2-norm error for U, V, uv.
"""

# ---- Set up color palette and colormap ----
cmp = sns.color_palette('RdBu_r', as_cmap=True)
plt.set_cmap(cmp)

#%% Load Data and Preprocessing
# ================================
# Load Configuration and Prediction Data
# ================================
act = ZPG_config.act
nn = ZPG_config.n_neural
nl = ZPG_config.n_layer
n_adam = ZPG_config.n_adam
data_name = f"ZPG_{nn}_{nl}_{act}_{n_adam}_2400"
data = np.load(f"pred/res_{data_name}.npz")

# ================================
# Load Reference Data
# ================================
les = loadmat('data/ZPG.mat')
Retheta = les['Retheta']
ind = np.isnan(Retheta)

# ---- Clean and filter data ----
Retheta = Retheta[~ind]
i_trn = (Retheta > 1000) & (Retheta < 7000)
i_trn = np.squeeze(i_trn)
id_yn = 200
ind = ind.reshape(-1)

# ---- Extract relevant fields ----
u = les['U'][:id_yn, ~ind][:, i_trn]
x = les['x'][:id_yn, ~ind][:, i_trn]
y = les['y'][:id_yn, ~ind][:, i_trn]

#%% Plot Loss vs Epoch
# ================================
# Loss Plot
# ================================
plt.figure(0)
plt.semilogy(data["hist"][:, 0], label="Total")
plt.semilogy(data["hist"][:, 1], label="Bc")
plt.semilogy(data["hist"][:, 2], label="Residual")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"figs/{data_name}_Loss")

#%% Contour Plots for U, V, and uv
# ================================
# Contour Plots
# ================================
pred = data["pred"]
ref = data["ref"]
names = ["U", "V", "uv"]

for i, name in enumerate(names):
    fig, axs = plt.subplots(3, figsize=(9, 10))

    # ---- Contour plots ----
    axs[0].contourf(x, y, pred[i, :, :])
    axs[1].contourf(x, y, ref[i, :, :])
    clb = axs[2].contourf(x, y, np.abs(ref[i, :, :] - pred[i, :, :]))
    
    # ---- Add colorbar ----
    cbar = plt.colorbar(clb, orientation="horizontal", pad=0.3)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.tick_params(labelsize=12)

    # ---- Titles ----
    axs[0].set_title(f"{name} Prediction", fontdict={"size": 18})
    axs[1].set_title("Reference", fontdict={"size": 18})
    axs[2].set_title("Error", fontdict={"size": 18})

    plt.tight_layout()
    plt.savefig(f"figs/{data_name}_{name}", dpi=300)

#%% L2-Norm Error Calculation
# ================================
# Compute and Print L2-Norm Error
# ================================
err = l2norm_err(ref, pred)
print(
    f"Error U  = {np.round(err[0], 3)}%" +
    f"\nError V  = {np.round(err[1], 3)}%" +
    f"\nError uv = {np.round(err[2], 3)}%"
)
