#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import disk
from skimage.filters.rank import median
from skimage.transform import downscale_local_mean

#%% Functions (common) --------------------------------------------------------

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs 

idxs = get_indexes(100, 34812)

def normalize_gcn(arr):
    arr = arr.astype("float32")
    arr -= np.mean(arr)
    arr /= np.std(arr)     
    return arr

def normalize_pct(arr, min_pct, max_pct):
    arr = arr.astype("float32")
    pMin = np.percentile(arr, min_pct);
    pMax = np.percentile(arr, max_pct);
    np.clip(arr, pMin, pMax, out=arr)
    arr -= pMin
    arr /= (pMax - pMin)   
    return arr

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

def import_stack(stack_path):

    print(f"Import - {stack_path.name} : ", end='')
    t0 = time.time()

    img_paths = list(stack_path.glob("**/*.tif"))

    # Import images
    def import_images(img_path):
        img = io.imread(img_path)
        return img
    
    stack = Parallel(n_jobs=-1)(
            delayed(import_images)(img_path) 
            for img_path in img_paths
            )
    stack = np.stack(stack)
    
    # Select slices
    z_mean = np.mean(stack, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
    stack = stack[z0:z1, ...]  
    
    # Append metadata
    metadata = {
        "name" : stack_path.name,
        "nZ" : stack.shape[0], 
        "nY" : stack.shape[1], 
        "nX" : stack.shape[2],
        "z0" : z0, "z1" : z1,
        }
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    return stack, metadata

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