#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.segmentation import clear_border
from skimage.transform import downscale_local_mean, rescale
from skimage.morphology import disk, binary_dilation, remove_small_objects

# Scipy
from scipy.linalg import lstsq
from skimage.filters import gaussian, median
from skimage.measure import label, regionprops
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import (
    shift, gaussian_filter1d, binary_fill_holes, distance_transform_edt, 
    affine_transform,
    )

#%% Parameters ----------------------------------------------------------------

rsize_factor = 8 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Paths ---------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
stack_names = ["D1_ICONX_DoS", "D11_ICONX_DoS", "D12_ICONX_corrosion", "H9_ICONX_DoS"]
stack_name = stack_names[0]

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)
                
#%% Functions -----------------------------------------------------------------                

def resize_image(img_path):
    return downscale_local_mean(io.imread(img_path), rsize_factor)

# -----------------------------------------------------------------------------

def process_stack(stack_path, stack_data):
    
    # Initialize
    print(f"\n{stack_path.stem}")
    print( "===================")
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Resize stack
    print("  Resize   :", end='')
    t0 = time.time()
    stack_rsize = Parallel(n_jobs=-1)(
            delayed(resize_image)(img_path) 
            for img_path in img_paths
            )
    stack_rsize = np.stack(stack_rsize)
    stack_rsize = downscale_local_mean(stack_rsize, (rsize_factor, 1, 1))
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s")
    
    # Select slices
    z_mean = np.mean(stack_rsize, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
    stack_rsize = stack_rsize[z0:z1, ...]  
        
    # Print variables
    print( "  -----------------")
    print(f"  zSlices  : {z0}-{z1}")
        
    # Outputs
    stack_data.append({
        "stack_path"    : stack_path,
        "stack_rsize"   : stack_rsize,
        })
    
#%% Execute -------------------------------------------------------------------

stack_data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        process_stack(stack_path, stack_data)
        
#%%

# Functions -------------------------------------------------------------------

def affine_registration(coords_ref, coords_reg):
    
    if coords_ref.shape[0] < coords_ref.shape[1]:
        coords_ref = coords_ref.T
        coords_reg = coords_reg.T
    (n, dim) = coords_ref.shape
    
    # Translations & transform matrices
    p, res, rnk, s = lstsq(
        np.hstack((coords_ref, np.ones([n, 1]))), coords_reg)
    t, T = p[-1].T, p[:-1].T
    
    # Merge matrices
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = T
    transform_matrix[:3, 3] = t
    
    return transform_matrix

# -----------------------------------------------------------------------------

stack = stack_data[0]["stack_rsize"]

centroids = []
for z, img in enumerate(stack):
    idx = np.argwhere(
        (gaussian(img, sigma=16 // rsize_factor) > 30000) == 1)   
    y, x = np.mean(idx, axis=0)
    centroids.append((z, y, x))
centroids = np.stack(centroids)


rod_vector = centroids[0] - centroids[-1]
ref_vector = [rod_vector[0], 0, 0]

rotation_axis = np.cross(rod_vector, ref_vector)

# stack = shift(stack, zyx_shift, mode='wrap')



# def roll_image(img_rsize):
#     idx = np.argwhere((img_rsize > 30000) == 1)
#     y0, x0 = img_rsize.shape[0] // 2, img_rsize.shape[1] // 2
#     y1, x1 = np.mean(idx, axis=0)
#     yx_shift = [y0 - y1, x0 - x1]
#     return shift(img_rsize, yx_shift, mode='wrap'), yx_shift 

# coords_ref, coords_reg = [], []
# for z, img in enumerate(stack_rsize):
#     idx = np.argwhere(
#         (gaussian(img, sigma=16 // rsize_factor) > 30000) == 1)
#     y_ref, x_ref = img.shape[0] / 2, img.shape[1] / 2
#     y_reg, x_reg = np.mean(idx, axis=0)
#     coords_ref.append((z, y_ref, x_ref))
#     coords_reg.append((z, y_reg, x_reg))
# coords_ref = np.stack(coords_ref)
# coords_reg = np.stack(coords_reg)

# -----------------------------------------------------------------------------

# transform_matrix = affine_registration(coords_ref, coords_reg)
# stack_rsize_reg = affine_transform(stack_rsize, transform_matrix)
# print(transform_matrix)

# -----------------------------------------------------------------------------

# Display
# import napari
# viewer = napari.Viewer()
# viewer.add_image(stack)
# viewer.add_image(stack_rsize2)
