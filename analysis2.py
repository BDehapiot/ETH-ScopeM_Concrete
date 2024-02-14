#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Scipy
from scipy.ndimage import shift

#%% Inputs --------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

#%% Functions -----------------------------------------------------------------

def roll_image(img):
    idx = np.argwhere((img > 30000) == 1)
    y0, x0 = img.shape[0] // 2, img.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img, yx_shift, mode='wrap'), yx_shift 

#%% Execute -------------------------------------------------------------------

# Open data
stack_reg = io.imread(Path(data_path, f"{exp_name}_stack_reg.tif"))
with open(Path(data_path, f"{exp_name}_stack_data.pkl"), "rb") as f:
    stack_data = pickle.load(f)
    
# Select slices
z_mean = np.mean(np.mean(stack_reg, axis=0), axis=(1,2))
z_mean_diff = np.gradient(z_mean)
idx = np.where((np.abs(z_mean_diff) < 5) & (z_mean > np.max(z_mean) * 0.9))[0]
stack_reg = stack_reg[:, idx[0]:idx[-1], ...]
    
# Roll stacks
stack_roll = []
for stack in stack_reg:
    outputs = Parallel(n_jobs=-1)(
            delayed(roll_image)(img) 
            for img in stack
            )
    stack_roll.append(np.stack([data[0] for data in outputs]))
stack_roll = np.stack(stack_roll)

#%%

stack_norm = []
for i, stack in enumerate(stack_roll):
    
    # Extract variables
    mtx_thresh = stack_data[i]["mtx_thresh"] 
    rod_thresh = stack_data[i]["rod_thresh"] 
    
    # Create mask
    avg_proj = np.mean(stack, axis=0)
    mtx_mask = avg_proj > mtx_thresh
    rod_mask = avg_proj < rod_thresh * 0.5
    
    #
    stack_norm.append(
        stack * mtx_mask[np.newaxis,...] * rod_mask[np.newaxis,...]
        )
    
stack_norm = np.stack(stack_norm)
stack_norm_avg = np.mean(stack_norm, axis=0)

io.imsave(
    Path(data_path, f"{exp_name}_stack_norm.tif"),
    stack_norm.astype("float32"),
    check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )
    
io.imsave(
    Path(data_path, f"{exp_name}_stack_norm_avg.tif"),
    stack_norm_avg.astype("float32"),
    check_contrast=False,
    )
