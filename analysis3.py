#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Skimage 
from skimage.filters import median
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_dilation, binary_erosion, remove_small_objects
    )

# Scipy
from scipy.stats import linregress
from scipy.ndimage import shift, binary_fill_holes, distance_transform_edt

# Sklearn
from sklearn.mixture import GaussianMixture 


#%% Inputs --------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    # "D1_ICONX_DoS"
    "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

rsize_factor = 4 # Image size reduction factor

#%% Functions -----------------------------------------------------------------

def roll_image(img):
    idx = np.argwhere((img > 30000) == 1)
    y0, x0 = img.shape[0] // 2, img.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img, yx_shift, mode='wrap'), yx_shift 

def norm_image(img, avg_mask, avg_proj):
    img = median(img, footprint=disk(8 // rsize_factor)) # Parameter
    img_norm = np.divide(img, avg_proj, where=avg_proj!=0)
    img_norm[avg_mask == 0] = 0
    return img_norm

#%% Execute -------------------------------------------------------------------

# Open data
hstack_reg = io.imread(Path(data_path, f"{exp_name}_hstack_reg.tif"))
with open(Path(data_path, f"{exp_name}_stack_data.pkl"), "rb") as f:
    stack_data = pickle.load(f)
    
# Select slices
z_mean = np.mean(np.mean(hstack_reg, axis=0), axis=(1,2))
z_mean_diff = np.gradient(z_mean)
idx = np.where((np.abs(z_mean_diff) < 5) & (z_mean > np.max(z_mean) * 0.9))[0]
hstack_reg = hstack_reg[:, idx[0]:idx[-1], ...]
    
# Roll stacks
print("Roll:", end='')
t0 = time.time()
hstack_roll = []
for stack in hstack_reg:
    outputs = Parallel(n_jobs=-1)(
        delayed(roll_image)(img) 
        for img in stack
        )
    hstack_roll.append(np.stack([data[0] for data in outputs]))
hstack_roll = np.stack(hstack_roll)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# Get masks
print("Masks:", end='')
t0 = time.time()
avg_projs, mtx_masks, rod_masks = [], [], []
for i, stack in enumerate(hstack_roll):
    avg_proj = np.mean(stack, axis=0)
    mtx_mask = avg_proj > stack_data[i]["mtx_thresh"] * 1.00 # Parameter
    rod_mask = avg_proj > stack_data[i]["rod_thresh"] * 1.00 # Parameter
    rod_mask = binary_fill_holes(rod_mask)
    avg_projs.append(avg_proj)
    mtx_masks.append(mtx_mask)
    rod_masks.append(rod_mask)   
avg_projs = np.stack(avg_projs)
mtx_masks = np.stack(mtx_masks)
rod_masks = np.stack(rod_masks)
avg_mask = np.min(mtx_masks, axis=0) ^ np.max(rod_masks, axis=0)
avg_mask = binary_erosion(avg_mask, footprint=disk(8 // rsize_factor)) # Parameter
t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# Get normalized stacks
print("Normalize #1:", end='')
t0 = time.time()
hstack_norm = []
for i, stack in enumerate(hstack_roll):
    stack_norm = Parallel(n_jobs=-1)(
        delayed(norm_image)(img, avg_mask, avg_projs[i]) 
        for img in stack 
        )
    hstack_norm.append(np.stack(stack_norm))
hstack_norm = np.stack(hstack_norm)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# Get object mask (averaged)
print("Object mask:", end='')
t0 = time.time()
norm_avg = np.min(hstack_norm, axis=0)
obj_mask = (norm_avg > 0.8) & (norm_avg > 0) # Parameter
obj_mask = np.invert(obj_mask)
obj_mask[norm_avg == 0] = 0
obj_mask = remove_small_objects(
    obj_mask, min_size=2048 * (1 / rsize_factor) ** 3) # Parameter
t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# Filter object mask
print("Filter object mask:", end='')
# tmp_mask = binary_fill_holes(avg_mask)
tmp_mask = binary_erosion(avg_mask)
tmp_mask = np.invert(tmp_mask)
tmp_mask = np.tile(tmp_mask[np.newaxis,...], (obj_mask.shape[0], 1, 1))
obj_labels = label(obj_mask)
props = regionprops(obj_labels, intensity_image=tmp_mask)
for prop in props:
    if prop.intensity_max > 0:
        obj_labels[obj_labels == prop.label] = 0
obj_mask_filt = obj_labels > 0
t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

print("Normalize #2:", end='')
t0 = time.time()

# Parameter
dStep = 12

# Get edm (outer surface)
edm = distance_transform_edt(binary_fill_holes(avg_mask))
edm[avg_mask == 0] = 0
edm = np.tile(edm[np.newaxis, :, :], (obj_mask.shape[0], 1, 1))

# Process
hstack_nnorm = []
dMax = np.max(edm)
dRange = np.arange(0, dMax + 1, dMax / (dStep + 1))
for stack_norm in hstack_norm:
    
    dLow = []
    for i in range(1, len(dRange) - 1):
        
        # Get distance mask
        d0 = dRange[i - 1]
        d1 = dRange[i + 1]
        dMask = (edm > d0) & (edm <= d1)
        dMask[obj_mask == 0] = 0  
        
        # Measure low quantile
        val = stack_norm[dMask == 1]
        dLow.append(np.quantile(val, 0.001)) # Parameter
        
    # Fit (linear)
    slope, intercept, r_value, p_value, std_err = linregress(dRange[1:-1], dLow)
    x = np.linspace(1, int(np.ceil(dMax)), int(np.ceil(dMax)))
    y = slope * x + intercept
    # plt.plot(x, y, 'gray')
    # plt.plot(dRange[1:-1], dLow, 'k')
    
    # Map values and normalize
    lookup_table = dict(zip(x, y))
    nnorm = np.vectorize(lookup_table.get)(edm.astype("int"))
    nnorm = np.array(nnorm, dtype=float)
    stack_nnorm = stack_norm / nnorm[np.newaxis,...]
    # stack_nnorm[obj_mask == 0] = 0
    
    # Append
    hstack_nnorm.append(stack_nnorm.squeeze())
hstack_nnorm = np.stack(hstack_nnorm)

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

# -----------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(hstack_norm)
# viewer.add_image(obj_outl, blending="additive")

#%%

val = []
for stack_nnorm in hstack_nnorm:
    val.append(stack_nnorm[obj_mask == 1].reshape(-1, 1))
val = np.concatenate(val)
gmm = GaussianMixture(n_components=2, random_state=42).fit(val)
x = np.linspace(val.min(), val.max(), 1000).reshape(-1, 1)
resp = gmm.predict_proba(x)
logprob = gmm.score_samples(x)
pdf = np.exp(logprob)
pdfs = resp * pdf[:, np.newaxis]
thresh = x[np.argmin(np.abs(resp[:,0] - 0.5))] * 1.0

plt.plot(x, pdf)

# -----------------------------------------------------------------------------

void_mask = ((hstack_nnorm < thresh) & (hstack_nnorm > 0) & (obj_mask_filt > 0))
lqud_mask = ((hstack_nnorm > thresh) & (obj_mask_filt > 0))

# -----------------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(hstack_norm)
viewer.add_image(hstack_nnorm)
viewer.add_image(
    void_mask, rendering="attenuated_mip",
    blending="additive", opacity=0.5, 
    colormap="bop orange"
    )
viewer.add_image(
    lqud_mask, rendering="attenuated_mip",
    blending="additive", opacity=0.5, 
    colormap="bop blue"
    )

#%%

