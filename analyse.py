#%% Imports -------------------------------------------------------------------

import time
import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Skimage
from skimage.measure import label
from skimage.morphology import (
    disk, ball, binary_erosion, binary_dilation, remove_small_objects
    )

# Scipy
from scipy.ndimage import binary_fill_holes

# Functions
from functions import filt_median, shift_stack, norm_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiment = "D1_ICONX_DoS" 
# experiment = "D11_ICONX_DoS" 
# experiment = "D12_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"
name = f"{experiment}_Time1_crop_df4"

# Parameters
overwrite = False
df = 4 # downscale factor

#%% 

# Open data
experiment_path = data_path / experiment
stack_norm = io.imread(experiment_path / (name + "_norm.tif"))
void_probs = io.imread(experiment_path / (name + "_probs.tif"))

# Open metadata
metadata_path = experiment_path / (name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
rod_mask = metadata["rod_mask"]    
mtx_mask = metadata["mtx_mask"]
mtx_EDM = metadata["mtx_EDM"]
centers = metadata["centers"]

# Shift 3D data
mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)
# mtx_mask_3D = shift_stack(mtx_mask, centers, reverse=True)

#%%

# Get void mask & labels
stack_norm = filt_median(stack_norm, 2)
void_mask = void_probs > 0.5
# void_mask = binary_erosion(void_mask, footprint=ball(2))
void_mask = remove_small_objects(void_mask, min_size=64)
void_labels = label(void_mask)

# Detect peripheral voids (connected to the exterior)
tMtx_mask = binary_erosion(mtx_mask, footprint=disk(2))
tMtx_mask = binary_fill_holes(tMtx_mask)
tMtx_mask_3D = shift_stack(tMtx_mask, centers, reverse=True)
rvl_lab = (void_labels[void_labels > 0]).ravel()
rvl_mask = (tMtx_mask_3D[void_labels > 0]).ravel()
tVoid_labels = void_labels.copy()
for idx in range(1, np.max(void_labels)):
    if np.min(rvl_mask[rvl_lab == idx]) == 0:
        tVoid_labels[tVoid_labels == idx] = 0
pVoid_labels = void_labels.copy()
pVoid_labels[tVoid_labels != 0] = 0

# Detect rod voids (connected to the rod)
tRod_mask = binary_dilation(rod_mask, footprint=disk(2))
tRod_mask_3D = shift_stack(tRod_mask, centers, reverse=True)   
rvl_lab = (void_labels[void_labels > 0]).ravel()
rvl_mask = (tRod_mask_3D[void_labels > 0]).ravel()     
tVoid_labels = void_labels.copy()
for idx in range(1, np.max(void_labels)):
    if np.max(rvl_mask[rvl_lab == idx]) != 0:
        tVoid_labels[tVoid_labels == idx] = 0
rVoid_labels = void_labels.copy()
rVoid_labels[tVoid_labels != 0] = 0

# Detect internal voids
iVoid_labels = void_labels
iVoid_labels[pVoid_labels != 0] = 0
iVoid_labels[rVoid_labels != 0] = 0

# viewer = napari.Viewer()
# viewer.add_labels(iVoid_labels)
# viewer.add_labels(pVoid_labels)
# viewer.add_labels(rVoid_labels)

#%% 

# t0 = time.time()
# print(" - Analyse : ", end='')

# # Ravel data
# rvl_lab = (void_labels[void_labels > 0]).ravel()
# rvl_EDM = ( mtx_EDM_3D[void_labels > 0]).ravel()
# rvl_int = ( stack_norm[void_labels > 0]).ravel()

# # Measure
# void_EDM, void_avg, void_std, void_pcl, void_pch = [], [], [], [], []
# for idx in range(1, np.max(void_labels)):
#     void_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
#     void_avg.append(np.mean(rvl_int[rvl_lab == idx]))
#     void_std.append(np.std(rvl_int[rvl_lab == idx]))
#     void_pcl.append(np.percentile(rvl_int[rvl_lab == idx], 5))
#     void_pch.append(np.percentile(rvl_int[rvl_lab == idx], 95))
# void_EDM = np.stack(void_EDM)
# void_avg = np.stack(void_avg)
# void_std = np.stack(void_std)
# void_pcl = np.stack(void_pcl)
# void_pch = np.stack(void_pch)
   
# # ---

# # lowEDM  = np.percentile(void_EDM, 20)
# # highEDM = np.percentile(void_EDM, 80)
# # lowIdx  = np.argwhere(void_EDM < lowEDM)
# # highIdx = np.argwhere(void_EDM > highEDM)
# # lowInt  = np.mean(void_pcl[lowIdx])
# # highInt = np.mean(void_pcl[highIdx])

# t1 = time.time()
# print(f"{(t1-t0):<5.2f}s") 

#%%

# xData = void_EDM 
# yData = void_pcl
# zData = void_pch - void_pcl 

# normalize = plt.Normalize(zData.min(), zData.max())
# plt.scatter(xData, yData, c=zData, cmap='plasma', norm=normalize, s=10)
# plt.ylim([0.4, 0.9])
# plt.show()

# plt.scatter(void_EDM, void_pcl, s=2)
# plt.scatter(void_EDM, void_pch, color="red", s=2)
# plt.ylim([0.4, 1.0])

#%%