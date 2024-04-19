#%% Imports -------------------------------------------------------------------

import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from skimage.measure import label

#%% Inputs --------------------------------------------------------------------

local_path = "D:/local_Concrete"
exp_name = (
    # "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    "H9_ICONX_DoS"
    )

#%% Initialize ----------------------------------------------------------------

# Open pickle file
with open(Path(local_path, f"{exp_name}_stack_data.pkl"), 'rb') as f:
    stack_data = pickle.load(f)

#%% Displays ------------------------------------------------------------------

idx = 1 
stack_rsize = stack_data[1]["stack_rsize"]
stack_roll = stack_data[1]["stack_roll"]
stack_norm = stack_data[1]["stack_norm"]
avg_proj = stack_data[1]["avg_proj"]
obj_mask_3D_1 = stack_data[1]["obj_mask_3D"]
obj_mask_3D_0 = stack_data[0]["obj_mask_3D"]

# Display stack normalization & object segmentation
viewer = napari.Viewer()
viewer.add_image(obj_mask_3D_1)
viewer.add_image(avg_proj, colormap="plasma")
viewer.add_image(np.mean(stack_rsize, axis=0), name="avg_proj (no roll)", colormap="plasma")
viewer.add_image(stack_norm, colormap="plasma")
viewer.add_image(stack_roll, colormap="plasma")
viewer.add_image(stack_rsize, colormap="plasma")
viewer.grid.enabled = True

# Display labels_objects
viewer = napari.Viewer()
viewer.add_labels(label(obj_mask_3D_1))
viewer.add_labels(label(obj_mask_3D_0))
