#%% Imports -------------------------------------------------------------------

import time
import napari
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Skimage
from skimage.measure import label
from skimage.morphology import ball, binary_erosion, remove_small_objects

# Scipy
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d 

# Functions
from functions import filt_median, shift_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
# experiment = "D1_ICONX_DoS" 
# experiment = "D11_ICONX_DoS" 
# experiment = "D12_ICONX_corrosion"
experiment = "H1_ICONX_DoS"
# experiment = "H9_ICONX_DoS"
name = f"{experiment}_Time3_crop_df4"

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

t0 = time.time()
print(" - Segment : ", end='')

def get_obj_mask(obj_probs, min_size, erode_radius):
    obj_mask_3D = obj_probs > 0.5
    obj_mask_3D = remove_small_objects(obj_mask_3D, min_size=min_size)
    if erode_radius > 1:
        obj_mask_3D = binary_erosion(obj_mask_3D, footprint=ball(erode_radius))
    return obj_mask_3D

# Format data
stack_norm = filt_median(stack_norm, 8 // df) # Parameter (8)
obj_mask_3D = get_obj_mask(
    obj_probs, 1.5e4 * (1 / df) ** 3, 8 // df) # Parameter (1.5e4, 8)
obj_labels_3D = label(obj_mask_3D)
mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)

# Measure object EDM & percentile low
obj_EDM, obj_pcl = [], []
rvl_lab = (obj_labels_3D[obj_labels_3D > 0]).ravel()
rvl_EDM = (mtx_EDM_3D[obj_labels_3D > 0]).ravel()
rvl_int = (stack_norm[obj_labels_3D > 0]).ravel()
for idx in range(1, np.max(obj_labels_3D)):
    obj_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
    obj_pcl.append(np.percentile(rvl_int[rvl_lab == idx], 5)) # Parameter (5)
obj_EDM = np.stack(obj_EDM)
obj_pcl = np.stack(obj_pcl)
    
# Find ref. points & fit parameters
maxEDM = np.max(obj_EDM)
x0 = maxEDM * 0.1 # Parameter (0.1)
x1 = maxEDM - x0
y0 = np.percentile(obj_pcl[obj_EDM < x0], 25) # Parameter (25)
y1 = np.percentile(obj_pcl[obj_EDM > x1], 25) # Parameter (25)
a = (y1 - y0) / (x1 - x0)
b = y0 - a * x0

# Determine threshold
obj_pcl -= (a * obj_EDM + b)
minPcl, maxPcl = np.min(obj_pcl), np.max(obj_pcl)
hist, bins = np.histogram(obj_pcl, bins=100, range=(minPcl, maxPcl))   
hist = gaussian_filter1d(hist, sigma=2) # Parameter
peaks, _ = find_peaks(hist, distance=33) # Parameter
_, _, lws, rws = peak_widths(hist, peaks, rel_height=0.5) # Parameter
idx = np.argmin(peaks)
thresh = rws[idx] + (rws[idx] - peaks[idx])
thresh_val = bins[int(thresh) + 1] + y0
thresh_val *= 1.0 # Parameter (1.0)
# print(thresh_val)
    
# plt.plot(hist)
# plt.axvline(x=peaks[idx])
# plt.axvline(x=rws[idx])
# plt.axvline(x=thresh, color="red")

# Correct stack_norm
obj_mask_3D = get_obj_mask(
    obj_probs, 1.5e4 * (1 / df) ** 3, 8 // df) # Parameter (1.5e4, 8)
mtx_EDM_3D = (a * mtx_EDM_3D + b) - y0
mtx_EDM_3D[obj_mask_3D == 0] = 0
stack_norm_corr = stack_norm - mtx_EDM_3D

# Get void & liquide masks
void_mask_3D = stack_norm_corr.copy()
void_mask_3D[obj_mask_3D == 0] = 0
void_mask_3D = void_mask_3D < thresh_val
void_mask_3D[obj_mask_3D == 0] = 0
liquid_mask_3D = stack_norm_corr.copy()
liquid_mask_3D[obj_mask_3D == 0] = 0
liquid_mask_3D = liquid_mask_3D > thresh_val
liquid_mask_3D[obj_mask_3D == 0] = 0

# # Filter masks
# liquid_mask_3D = remove_small_objects(
#     liquid_mask_3D, min_size=1.5e4 * (1 / df) ** 3) # Parameter (1.5e4)
# void_mask_3D[(liquid_mask_3D == 0) & (obj_mask_3D == 1)] = 1

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(stack, opacity=0.75)
viewer.add_image(stack_norm_corr, opacity=0.75)
viewer.add_image(
    void_mask_3D, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip",
    # attenuation=0.5,
    colormap="bop orange"
    )
viewer.add_image(
    liquid_mask_3D, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip", 
    # attenuation=0.5,
    colormap="bop blue"
    )