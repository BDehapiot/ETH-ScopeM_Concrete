#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import shift_stack, norm_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
name = f"{experiment}_Time2_crop_df4"

# Parameters
overwrite = False
df = 4 # downscale factor

#%% 

# Open data
experiment_path = data_path / experiment
stack_norm = io.imread(
    experiment_path / (name + "_norm.tif"))
obj_labels_3D = io.imread(
    experiment_path / (name + "_labels.tif"))

# Open metadata
metadata_path = experiment_path / (name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
mtx_mask = metadata["mtx_mask"]
rod_mask = metadata["rod_mask"]
mtx_EDM  = metadata["mtx_EDM"]
rod_EDM  = metadata["rod_EDM"]
centers  = metadata["centers"]

#%%

# stack_voids = stack_norm.copy()
# stack_voids[obj_labels_3D == 0] = 0
# mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)

# void_EDM, void_int = [], []
# for idx in range(1, np.max(obj_labels_3D)):
#     # void_EDM.append(np.nanmean(mtx_EDM_3D[obj_labels_3D == idx]))
#     # void_int.append(np.nanmean(stack_voids[obj_labels_3D == idx]))
#     void_EDM.append(np.nanpercentile(mtx_EDM_3D[obj_labels_3D == idx], 1))
#     void_int.append(np.nanpercentile(stack_voids[obj_labels_3D == idx], 1))
    
# plt.scatter(void_EDM, void_int)

#%%

import napari
viewer = napari.Viewer()
viewer.add_image(stack_norm)
viewer.add_image(obj_labels_3D > 0)

