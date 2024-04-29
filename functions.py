#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Scipy
from scipy.ndimage import shift

#%% Function(s) ---------------------------------------------------------------

def normalize_stack(stack, med_proj, yx_shift, mask=None):
    
    def _normalize_stack(img, med_proj, yx_shift, mask=mask):
        yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
        med_proj = shift(med_proj, yx_shift)
        img_norm = np.divide(img, med_proj, where=med_proj!=0)
        if mask is not None:
            mask = shift(mask.astype("uint8"), yx_shift)
            img_norm *= mask
        return img_norm
    
    if stack.ndim == 2:
        stack_norm = _normalize_stack(stack, med_proj, yx_shift, mask=mask)
    
    if stack.ndim == 3:
        stack_norm = Parallel(n_jobs=-1)(
            delayed(_normalize_stack)(
                img, med_proj, yx_shift[z], mask=mask
                ) 
            for z, img in enumerate(stack)
            )
        stack_norm = np.stack(stack_norm)
    
    return stack_norm

def mask_stack(stack, mask, yx_shift):
    
    def _mask_array(img, mask, yx_shift):
        yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
        mask = shift(mask.astype("uint8"), yx_shift)
        img_mask = img * mask
        return img_mask
    
    if stack.ndim == 2:
        stack_mask = _mask_array(stack, mask, yx_shift)
    
    if stack.ndim == 3:
        stack_mask = Parallel(n_jobs=-1)(
            delayed(_mask_array)(
                img, mask, yx_shift[z],
                ) 
            for z, img in enumerate(stack)
            )
        stack_mask = np.stack(stack_mask)
    
    return stack_mask
