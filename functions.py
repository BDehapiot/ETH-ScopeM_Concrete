#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.transform import downscale_local_mean

#%% Functions (common) --------------------------------------------------------

def nearest_divisibles(value, levels):
    divisor = 2 ** levels
    lowDiv = value - (value % divisor)
    if lowDiv == value:
        highDiv = value + divisor
    else:
        highDiv = lowDiv + divisor
    return lowDiv, highDiv

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs 

def median_filt(arr, radius):
    def _median_filt(img):
        img = median(img, footprint=disk(radius))
        return img
    if arr.ndim == 2:
        arr = _median_filt(arr)
    if arr.ndim == 3:
        arr = Parallel(n_jobs=-1)(delayed(_median_filt)(img) for img in arr)
    return np.stack(arr)

#%% Functions (procedures) ----------------------------------------------------

def import_stack(stack_path, save_path):

    # Import ------------------------------------------------------------------

    stack_name = stack_path.name

    print(stack_name)
    print(" - Import : ", end='')
    t0 = time.time()
    
    stack1 = []
    img_paths = list(stack_path.glob("**/*.tif"))
    for img_path in img_paths:
        stack1.append(io.imread(img_path))
    stack1 = np.stack(stack1)
    
    # Select slices
    z_mean = np.mean(stack1, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
     
    # Crop to the nearest divisibles (zyx)
    z0 = nearest_divisibles(z0, 4)[1] 
    z1 = nearest_divisibles(z1, 4)[0] 
    nYdiv = nearest_divisibles(stack1.shape[1], 4)[0]
    nXdiv = nearest_divisibles(stack1.shape[2], 4)[0]
    y0 = (stack1.shape[1] - nYdiv) // 2 
    y1 = y0 + nYdiv
    x0 = (stack1.shape[2] - nXdiv) // 2 
    x1 = x0 + nXdiv
    stack1 = stack1[z0:z1, y0:y1, x0:x1] 

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Downscale ---------------------------------------------------------------
    
    print(" - Downscale : ", end='')
    t0 = time.time()

    stack2 = downscale_local_mean(stack1, 2)
    stack4 = downscale_local_mean(stack2, 2)
    stack8 = downscale_local_mean(stack4, 2)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Save --------------------------------------------------------------------
    
    print(" - Save : ", end='')
    t0 = time.time()
    
    # Data
    io.imsave(
        Path(save_path, stack_name + "_crop_d1.tif"), stack1, check_contrast=False)
    io.imsave(
        Path(save_path, stack_name + "_crop_d2.tif"), stack2, check_contrast=False)
    io.imsave(
        Path(save_path, stack_name + "_crop_d4.tif"), stack4, check_contrast=False)
    io.imsave(
        Path(save_path, stack_name + "_crop_d8.tif"), stack8, check_contrast=False)
    
    # Metadata
    metadata = {
        "name"  : stack_name,
        "shape" : stack1.shape,
        "z0" : z0, "z1" : z1,
        "y0" : y0, "y1" : y1,
        "x0" : x0, "x1" : x1,
        }
    
    with open(Path(save_path, stack_name + "_metadata.pkl"), 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

#%%

# def preprocess(arr, min_pct, max_pct, radius):
#     arr = normalize_gcn(arr)
#     arr = normalize_pct(arr, min_pct, max_pct)
#     if radius > 1:
#         arr = (arr * 255).astype("uint8")
#         arr = median_filt(arr, radius=radius)
#         arr = arr.astype("float32") / 255
#     return arr

# def import_stack(stack_path, downscale_factor):

#     print(f"Import - {stack_path.name} : ", end='')
#     t0 = time.time()

#     img_paths = list(stack_path.glob("**/*.tif"))

#     # format images
#     def import_images(img_path, downscale_factor):
#         img = io.imread(img_path)
#         if downscale_factor > 1:
#             img = downscale_local_mean(img, downscale_factor)
#         return img
#     stack = Parallel(n_jobs=-1)(
#             delayed(import_images)(img_path, downscale_factor) 
#             for img_path in img_paths
#             )
#     stack = np.stack(stack)
#     if downscale_factor > 1:
#         stack = downscale_local_mean(stack, (downscale_factor, 1, 1))
    
#     # Select slices
#     z_mean = np.mean(stack, axis=(1,2)) 
#     z_mean_diff = np.gradient(z_mean)
#     z0 = np.nonzero(z_mean_diff)[0][0] + 1
#     z1 = np.where(
#         (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
#     stack = stack[z0:z1, ...]  
        
#     t1 = time.time()
#     print(f"{(t1-t0):<5.2f}s")
    
#     return stack

# def preprocess_stack(stack_path, downscale_factor):

#     print(f"Preprocess - {stack_path.name} : ", end='')
#     t0 = time.time()

#     img_paths = list(stack_path.glob("**/*.tif"))

#     # Open images
#     def open_images(img_path, downscale_factor):
#         img = io.imread(img_path)
#         img = downscale_local_mean(img, downscale_factor)
#         return img
#     stack = Parallel(n_jobs=-1)(
#             delayed(open_images)(img_path, downscale_factor) 
#             for img_path in img_paths
#             )
#     stack = np.stack(stack)
#     stack = downscale_local_mean(stack, (downscale_factor, 1, 1))
    
#     # Select slices
#     z_mean = np.mean(stack, axis=(1,2)) 
#     z_mean_diff = np.gradient(z_mean)
#     z0 = np.nonzero(z_mean_diff)[0][0] + 1
#     z1 = np.where(
#         (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
#     stack = stack[z0:z1, ...]  
    
#     # Normalize & convert to uint8
#     stack = normalize_gcn(stack)
#     stack = normalize_pct(stack, 0.01, 99.99)
#     stack = (stack * 255).astype("uint8")
    
#     # Reslice
#     rslice = np.swapaxes(stack, 0, 1)

#     t1 = time.time()
#     print(f" {(t1-t0):<5.2f}s")
    
#     return stack, rslice