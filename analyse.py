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
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import binary_fill_holes, gaussian_filter1d 
from scipy import stats

# Functions
from functions import filt_median, shift_stack, norm_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiment = "D1_ICONX_DoS" 
# experiment = "D11_ICONX_DoS" 
# experiment = "D12_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"
name = f"{experiment}_Time7_crop_df4"

# Parameters
overwrite = False
df = 4 # downscale factor

#%% 

# Open data
experiment_path = data_path / experiment
stack = io.imread(experiment_path / (name + ".tif"))
stack_norm = io.imread(experiment_path / (name + "_norm.tif"))
obj_probs = io.imread(experiment_path / (name + "_probs.tif"))

# Open metadata
metadata_path = experiment_path / (name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
centers = metadata["centers"]
med_proj = metadata["med_proj"]
mtx_mask = metadata["mtx_mask"]
mtx_EDM = metadata["mtx_EDM"]

#%%

def get_void_mask(stack_norm, mtx_EDM, df):
    
    global \
        y0, obj_EDM, obj_pcl, obj_mask, stack_norm_corr
        
    # Format data
    stack_norm = filt_median(stack_norm, 8 // df) # Parameter (8)
    obj_mask = obj_probs > 0.5
    obj_mask = remove_small_objects(obj_mask, min_size=256 // df) # Parameter (256)
    obj_mask = binary_erosion(obj_mask, footprint=ball(12 // df)) # Parameter (12)
    obj_labels = label(obj_mask)
    mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)
    
    # Measure object EDM & percentile low
    obj_EDM, obj_pcl = [], []
    rvl_lab = (obj_labels[obj_labels > 0]).ravel()
    rvl_EDM = (mtx_EDM_3D[obj_labels > 0]).ravel()
    rvl_int = (stack_norm[obj_labels > 0]).ravel()
    for idx in range(1, np.max(obj_labels)):
        obj_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
        obj_pcl.append(np.percentile(rvl_int[rvl_lab == idx], 5)) # Parameter (5)
    obj_EDM = np.stack(obj_EDM)
    obj_pcl = np.stack(obj_pcl)
    
    # Find reference points & fit parameters
    maxEDM = np.max(obj_EDM)
    x0 = maxEDM * 0.1 # Parameter (0.1)
    x1 = maxEDM - x0
    y0 = np.percentile(obj_pcl[obj_EDM < x0], 25) # Parameter (25)
    y1 = np.percentile(obj_pcl[obj_EDM > x1], 25) # Parameter (25)
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0
    
    # Find threshold (void vs. liquide)
    obj_pcl -= (a * obj_EDM + b)
    # minPcl, maxPcl = np.min(obj_pcl), np.max(obj_pcl)
    # hist, bins = np.histogram(obj_pcl, bins=100, range=(minPcl, maxPcl))   
    # hist = gaussian_filter1d(hist, sigma=2)
    # peaks, _ = find_peaks(hist, distance=33)
    # _, _, lws, rws = peak_widths(hist, peaks, rel_height=0.5)
    # idx = np.argmin(peaks)
    # thresh = rws[idx] + (rws[idx] - peaks[idx])

    # Correct stack_norm
    obj_mask = obj_probs > 0.5
    obj_mask = remove_small_objects(obj_mask, min_size=256 // df) # Parameter (256)
    obj_mask = binary_erosion(obj_mask, footprint=ball(8 // df)) # Parameter (8)
    mtx_EDM_3D = (a * mtx_EDM_3D + b) - y0
    mtx_EDM_3D[obj_mask == 0] = 0
    stack_norm_corr = stack_norm - mtx_EDM_3D
    
    return

t0 = time.time()
print(" - Test : ", end='')

get_void_mask(stack_norm, mtx_EDM, df)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

# plt.scatter(obj_EDM, obj_pcl)
# plt.show()

# plt.hist(obj_pcl, bins=100)
# plt.show()

#%%

# Find threshold (void vs. liquide)
minPcl, maxPcl = np.min(obj_pcl), np.max(obj_pcl)
hist, bins = np.histogram(obj_pcl, bins=100, range=(minPcl, maxPcl))   
hist = gaussian_filter1d(hist, sigma=2) # Parameter
peaks, _ = find_peaks(hist, distance=33) # Parameter
_, _, lws, rws = peak_widths(hist, peaks, rel_height=0.5) # Parameter
idx = np.argmin(peaks)
thresh = rws[idx] + (rws[idx] - peaks[idx])
thresh_val = bins[int(thresh) + 1] + y0
thresh_val *= 1.05 # Parameter 
print(thresh_val)

plt.plot(hist)
plt.axvline(x=peaks[idx])
plt.axvline(x=rws[idx])
plt.axvline(x=thresh, color="red")

# -----------------------------------------------------------------------------

from skimage.morphology import binary_opening

void_mask = stack_norm_corr.copy()
void_mask[obj_mask == 0] = 0
void_mask = void_mask < thresh_val
void_mask[obj_mask == 0] = 0

liqu_mask = stack_norm_corr.copy()
liqu_mask[obj_mask == 0] = 0
liqu_mask = liqu_mask > thresh_val
liqu_mask[obj_mask == 0] = 0

# liqu_mask = binary_opening(liqu_mask, footprint=ball(1))

viewer = napari.Viewer()
viewer.add_image(stack_norm_corr, opacity=0.75)
viewer.add_image(
    void_mask, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip",
    # attenuation=0.5,
    colormap="bop orange"
    )
viewer.add_image(
    liqu_mask, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip", 
    # attenuation=0.5,
    colormap="bop blue"
    )

#%%

# t0 = time.time()
# print(" - Analyse : ", end='')

# # Prepare data
# stack_norm = filt_median(stack_norm, 2)
# obj_mask = obj_probs > 0.5
# obj_mask = remove_small_objects(obj_mask, min_size=64)
# obj_mask = binary_erosion(obj_mask, footprint=ball(3))
# obj_labels = label(obj_mask)
# mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)

# # Ravel data
# rvl_lab = (obj_labels[obj_labels > 0]).ravel()
# rvl_EDM = (mtx_EDM_3D[obj_labels > 0]).ravel()
# rvl_int = (stack_norm[obj_labels > 0]).ravel()

# # Measure
# obj_EDM, obj_pcl, obj_pch = [], [], []
# for idx in range(1, np.max(obj_labels)):
#     obj_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
#     obj_pcl.append(np.percentile(rvl_int[rvl_lab == idx], 5))
#     obj_pch.append(np.percentile(rvl_int[rvl_lab == idx], 95))
# obj_EDM = np.stack(obj_EDM)
# obj_pcl = np.stack(obj_pcl)
# obj_pch = np.stack(obj_pch)

# t1 = time.time()
# print(f"{(t1-t0):<5.2f}s") 

#%% 

# xData = obj_EDM 
# yData = obj_pcl
# zData = obj_pch

# # Find reference points
# x0 = 10 # Parameter
# x1 = np.max(obj_EDM) - x0
# y0 = np.percentile(obj_pcl[obj_EDM < x0], 25) # Parameter
# y1 = np.percentile(obj_pcl[obj_EDM > x1], 25) # Parameter

# # Fit
# a = (y1 - y0) / (x1 - x0)
# b = y0 - a * x0
# xFit = np.arange(np.min(obj_EDM), np.max(obj_EDM))
# yFit = a * xFit + b
# yCorr = (a * xData + b) - y0

# # Plot
# plt.scatter(xData, yData, s=5)
# plt.plot(xFit, yFit, color='red')
# plt.show()

# # Plot
# plt.scatter(xData, yData - yCorr, s=5)
# plt.show()

# plt.hist(yData - yCorr, bins=100)
# plt.show()

#%%

# obj_mask = obj_probs > 0.5
# obj_mask = remove_small_objects(obj_mask, min_size=64)
# obj_mask = binary_erosion(obj_mask, footprint=ball(1))
# mtx_EDM_3D_corr = (a * mtx_EDM_3D + b) - y0
# mtx_EDM_3D_corr[obj_mask == 0] = 0
# stack_norm_corr = stack_norm - mtx_EDM_3D_corr

# void_mask = stack_norm_corr.copy()
# void_mask[obj_mask == 0] = 0
# void_mask = void_mask < 0.625
# void_mask[obj_mask == 0] = 0

# liqu_mask = stack_norm_corr.copy()
# liqu_mask[obj_mask == 0] = 0
# liqu_mask = liqu_mask > 0.625
# liqu_mask[obj_mask == 0] = 0

# viewer = napari.Viewer()
# viewer.add_image(stack_norm_corr, opacity=0.5)
# viewer.add_image(void_mask, blending="additive", colormap="bop orange")
# viewer.add_image(liqu_mask, blending="additive", colormap="bop blue")

