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

#%%

# Objects -----------------------------------------------------------------

t0 = time.time()
print(" - Objects : ", end='')
    
def get_object_EDM(idx, obj_labels_3D, mtx_EDM_3D):
    labels = obj_labels_3D.copy()
    labels[labels == idx] = 0
    obj_EDM_3D = distance_transform_edt(1 - labels > 0)
    obj_EDM_3D[obj_labels_3D == 0] = 0 # Don't know why
    obj_dist = np.nanmean(obj_EDM_3D[obj_labels_3D == idx])
    mtx_dist = np.nanmean(mtx_EDM_3D[obj_labels_3D == idx])
    return obj_dist, mtx_dist

# Parameters
obj_df = 16 // df # parameter

# Normalize stack
stack_norm = norm_stack(
    stack, med_proj, centers, radius=obj_df,  mask=mtx_mask)

# Object mask and labels
obj_mask_3D = (stack_norm < 0.8) & (stack_norm > 0) # parameter (0.8)
obj_mask_3D = remove_small_objects(
    obj_mask_3D, min_size=1e5 * (1 / df) ** 3) # parameter (2.5e5)
obj_mask_3D = clear_border(obj_mask_3D)
obj_labels_3D = label(obj_mask_3D)

# Get EDM measurments
idxs = np.unique(obj_labels_3D)[1:]
obj_labels_3D_low = rescale(
    obj_labels_3D, 1 / obj_df, order=0).astype(int)
mtx_EDM_3D_low = rescale(
    shift_stack(mtx_EDM, centers, reverse=True), 1 / obj_df)
outputs = Parallel(n_jobs=-1)(
        delayed(get_object_EDM)(idx, obj_labels_3D_low, mtx_EDM_3D_low) 
        for idx in idxs
        )
obj_dist = [data[0] for data in outputs] * obj_df
mtx_dist = [data[1] for data in outputs] * obj_df
# obj_dist /= np.mean(obj_dist)
# mtx_dist /= np.mean(mtx_dist)

# Get object properties
objects = []
props = regionprops(obj_labels_3D)
for i, prop in enumerate(props):
    objects.append({
        "label"    : prop.label,
        "centroid" : prop.centroid,
        "area"     : prop.area,
        "solidity" : prop.solidity,
        "obj_dist" : obj_dist[i],
        "mtx_dist" : mtx_dist[i],
        })
        
t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

#%%

# t0 = time.time()
# print(" - Void masks : ", end='')

# # Get void mask & labels
# stack_norm = filt_median(stack_norm, 2)
# void_mask = void_probs > 0.5
# void_mask = remove_small_objects(void_mask, min_size=64)
# void_labels = label(void_mask)
# void_labels_r = (void_labels[void_labels > 0]).ravel()

# # Detect outer vs. inner voids
# tmp_mask = binary_erosion(mtx_mask, footprint=disk(2))
# tmp_mask_3D = shift_stack(tmp_mask, centers, reverse=True)
# tmp_mask_3D_r = (tmp_mask_3D[void_labels > 0]).ravel()
# iVoid_labels = void_labels.copy()

# for idx in range(1, np.max(void_labels)):
#     if np.min(tmp_mask_3D_r[void_labels_r == idx]) == 0:
#         iVoid_labels[iVoid_labels == idx] = 0
        
# # Display
# viewer = napari.Viewer()
# viewer.add_labels(iVoid_labels)

# # Detect peripheral voids (connected to the exterior)
# tMtx_mask = binary_erosion(mtx_mask, footprint=disk(2))
# tMtx_mask = binary_fill_holes(tMtx_mask)
# tMtx_mask_3D = shift_stack(tMtx_mask, centers, reverse=True)
# rvl_mask = (tMtx_mask_3D[void_labels > 0]).ravel()
# tVoid_labels = void_labels.copy()
# for idx in range(1, np.max(void_labels)):
#     if np.min(rvl_mask[rvl_lab == idx]) == 0:
#         tVoid_labels[tVoid_labels == idx] = 0
# pVoid_labels = void_labels.copy()
# pVoid_labels[tVoid_labels != 0] = 0

# # Detect rod voids (connected to the rod)
# tRod_mask = binary_dilation(rod_mask, footprint=disk(2))
# tRod_mask_3D = shift_stack(tRod_mask, centers, reverse=True)   
# rvl_mask = (tRod_mask_3D[void_labels > 0]).ravel()     
# tVoid_labels = void_labels.copy()
# for idx in range(1, np.max(void_labels)):
#     if np.max(rvl_mask[rvl_lab == idx]) != 0:
#         tVoid_labels[tVoid_labels == idx] = 0
# rVoid_labels = void_labels.copy()
# rVoid_labels[tVoid_labels != 0] = 0

# # Detect internal voids
# iVoid_labels = void_labels
# iVoid_labels[pVoid_labels != 0] = 0
# iVoid_labels[rVoid_labels != 0] = 0

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

# # Display
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