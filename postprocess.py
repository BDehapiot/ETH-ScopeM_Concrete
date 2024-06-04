#%% Imports -------------------------------------------------------------------

import cv2
import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
# experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
experiment = "H9_ICONX_DoS"

#%%

# Open data
experiment_path = data_path / experiment
experiment_reg_path = data_path / experiment / "REG"
stack_reg = io.imread(experiment_reg_path / (experiment + "_reg.tif"))
norm_reg = io.imread(experiment_reg_path / (experiment + "_norm_reg.tif"))

#%% 

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs 

def get_histmap(stack0, stack1, mask, zIdx):
    
    # Get histograms
    hist0, hist1 = [], []
    for z in zIdx:
        img0 = stack0[z, ...]
        img1 = stack1[z, ...]
        tmp_mask = mask[z, ...]
        tmp_mask = binary_fill_holes(tmp_mask)
        if np.sum(img0) > 0 and np.sum(img1) > 0: 
            h0, _ = np.histogram(
                img0[tmp_mask].flatten(), bins=256, 
                range=(0, 255), density=True
                ) 
            h1, _ = np.histogram( 
                img1[tmp_mask].flatten(), bins=256, 
                range=(0, 255), density=True
                )
            hist0.append(h0)
            hist1.append(h1)
    
    # Avg. histograms
    hist0 = np.stack(hist0).transpose()
    hist0 = np.nanmean(hist0, axis=1)
    hist1 = np.stack(hist1).transpose()
    hist1 = np.nanmean(hist1, axis=1)
    
    # Cumulative sums
    csum0 = np.cumsum(hist0)
    csum1 = np.cumsum(hist1)
    
    # Get mapping
    histmap = np.zeros(256)
    for b in range(256):
        diff = np.abs(csum0 - csum1[b])
        idx = np.argmin(diff)
        histmap[b] = idx
            
    return histmap, hist0, hist1, csum0, csum1  

# -----------------------------------------------------------------------------

mask = norm_reg[0, ...] > 0
stack_reg = (stack_reg / 255).astype("uint8")
zIdx = get_indexes(50, stack_reg.shape[1]) # Parameter

for t in range(1, stack_reg.shape[0]):

    histmap, _, _, _, _ = get_histmap(
        stack_reg[0, ...], stack_reg[t, ...], mask, zIdx)
    
    for z in range(stack_reg.shape[1]):
        
        img0 = stack_reg[0, z, ...]
        img1 = stack_reg[t, z, ...]
        
        if np.sum(img0) > 0 and np.sum(img1) > 0:
            
            stack_reg[t, z, ...] = cv2.LUT(img1, histmap)
            
stack_reg[norm_reg < 0.01] = 0

# -----------------------------------------------------------------------------

# fig, axs = plt.subplots(4, 1, figsize=(4, 12))
# axs[0].plot(hist0)
# axs[1].plot(hist1)
# axs[2].plot(csum0)
# axs[3].plot(csum1)
# plt.show()

#%%

import napari
viewer = napari.Viewer()
viewer.add_image(stack_reg, contrast_limits=(10, 200))
# viewer.add_image(stack_reg[1, ...], contrast_limits=(10, 200))

