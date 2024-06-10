#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Skimage
from skimage.filters import gaussian
from skimage.exposure import match_histograms

# Scipy
from scipy.ndimage import binary_fill_holes

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
# experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
# experiment = "H9_ICONX_DoS"

#%%

# Open data
experiment_path = data_path / experiment
experiment_reg_path = data_path / experiment / "REG"
stack_reg = io.imread(experiment_reg_path / (experiment + "_reg.tif"))
norm_reg = io.imread(experiment_reg_path / (experiment + "_norm_reg.tif"))

#%% 

t0 = time.time()
print(" - test : ", end='')

stack_reg_norm = stack_reg.copy()

mask = norm_reg[0, ...] > 0
for t in range(mask.shape[0]):
    mask[t, ...] = binary_fill_holes(mask[t, ...])

for t in range(1, stack_reg_norm.shape[0]):
    stack0 = stack_reg_norm[t - 1, ...]
    stack0[mask == 0] = 0
    stack1 = stack_reg_norm[t, ...]
    stack1[mask == 0] = 0
    for z in range(stack_reg_norm.shape[1]):
        img0 = stack0[z, ...].copy()
        img1 = stack1[z, ...].copy()
        img0[img1 == 0] = 0
        stack_reg_norm[t, z, ...] = match_histograms(img1, img0)
        
stack_reg_norm = stack_reg_norm.astype("float32")
mask = gaussian(mask, sigma=2)
stack_reg_norm *= mask
stack_reg_norm = stack_reg_norm.astype("uint16")

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 
        
#%%

import napari
viewer = napari.Viewer()
viewer.add_image(stack_reg_norm, contrast_limits=(0, 50000))  
