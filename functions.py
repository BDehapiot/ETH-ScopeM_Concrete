#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed

# Skimage
from skimage.filters import median
from skimage.morphology import disk

# Scipy
from scipy.ndimage import shift

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

#%% Function(s) ---------------------------------------------------------------

def filt_median(arr, radius):
    
    def _filt_median(img):
        return median(img, footprint=disk(radius))
    
    if arr.ndim == 2:
        arr = _filt_median(arr)
    
    if arr.ndim == 3:
        arr = Parallel(n_jobs=-1)(
            delayed(_filt_median)(img) 
            for img in arr
            )
        arr = np.stack(arr)
    
    return arr

# -----------------------------------------------------------------------------

def shift_stack(stack, centers, reverse=False):
    
    def shift_img(img, center):
        if reverse:         
            center = [- center[0], - center[1]]
        if img.dtype == bool:
            img = img.astype("uint8")
        return shift(img, center)
    
    # Shift 1 image / 1 center
    if stack.ndim == 2 and len(centers) == 1:
        stack = shift_img(stack, centers)

    # Shift 1 image / n centers
    elif stack.ndim == 2 and len(centers) > 1:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(stack, center) 
            for center in centers
            )
        stack = np.stack(stack)
        
    # Shift n images / n centers
    elif stack.ndim == 3:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(img, center) 
            for img, center in zip(stack, centers)
            )
        stack = np.stack(stack)
        
    return stack

# -----------------------------------------------------------------------------

def norm_stack(stack, med_proj, centers, radius=1, mask=None):
    
    # def filt_median(img):
    #     return median(img, footprint=disk(radius))
        
    # if radius > 1:
    #     stack = Parallel(n_jobs=-1)(
    #         delayed(filt_median)(img) 
    #         for img in stack
    #         )
    
    if radius > 1:
        stack = filt_median(stack, radius)

    med_proj = shift_stack(med_proj, centers, reverse=True)        
    stack = np.divide(stack, med_proj, where=med_proj != 0)

    if mask is not None:
        mask = shift_stack(mask, centers, reverse=True)
        stack *= mask
        
    return stack

# -----------------------------------------------------------------------------

def predict(stack, model_path, subset=1000):
    
    # Define model
    model = sm.Unet(
        'resnet34', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(model_path)
    size = int(model_path.name[22:26])
    overlap = size // 4

    # Define sub indexes
    nZ = stack.shape[0]
    z0s = np.arange(0, nZ, subset)
    z1s = z0s + subset
    z1s[z1s > nZ] = nZ
    
    # Normalize stack
    stack = norm_gcn(stack, mask=stack != 0)
    stack = norm_pct(stack, 0.01, 99.99, mask=stack != 0)
    
    # Predict
    probs = []
    for z0, z1 in zip(z0s, z1s):
        tmpStack = stack[z0:z1, ...]
        patches = extract_patches(tmpStack, size, overlap)
        patches = np.stack(patches)
        tmpProbs = model.predict(patches).squeeze()
        tmpProbs = merge_patches(tmpProbs, tmpStack.shape, overlap)
        probs.append(tmpProbs)
    probs = np.concatenate(probs, axis=0)
        
    return probs
