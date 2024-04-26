#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import disk
from skimage.filters.rank import median

# Scipy
from scipy.ndimage import shift

#%% Function(s) ---------------------------------------------------------------

#%% 

# Parameters
t = 3
df = 4

# Paths
data_path = Path("D:/local_Concrete/data")
stack_path = data_path / "D1_ICONX_DoS" / f"D1_ICONX_DoS_Time{t}_crop_df{df}.tif"
metadata_path = data_path / "D1_ICONX_DoS" / f"D1_ICONX_DoS_Time{t}_metadata_oo.pkl"

# Open 
stack = io.imread(stack_path)
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
idx = metadata["dfs"].index(df)
med_proj = metadata["med_projs"][idx]
mtx_mask = metadata["mtx_masks"][idx]
yx_shift = metadata["yx_shifts"][idx]

# -----------------------------------------------------------------------------

def normalize_array(arr, med_proj, yx_shift, mask=None):
    
    def _normalize_array(arr, med_proj, yx_shift, mask=mask):
        yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
        med_proj = shift(med_proj, yx_shift)
        arr_norm = np.divide(arr, med_proj, where=med_proj!=0)
        if mask is not None:
            mask = shift(mask.astype("uint8"), yx_shift)
            arr_norm *= mask
        return arr_norm
    
    if arr.ndim == 2:
        arr_norm = _normalize_array(arr, med_proj, yx_shift, mask=mask)
    
    if arr.ndim == 3:
        arr_norm = Parallel(n_jobs=-1)(
            delayed(_normalize_array)(
                img, med_proj, yx_shift[z], mask=mask
                ) 
            for z, img in enumerate(stack)
            )
        arr_norm = np.stack(arr_norm)
    
    return arr_norm

def mask_array(arr, mask, yx_shift):
    
    def _mask_array(arr, mask, yx_shift):
        yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
        mask = shift(mask.astype("uint8"), yx_shift)
        arr_mask = arr * mask
        return arr_mask
    
    if arr.ndim == 2:
        arr_mask = _mask_array(arr, mask, yx_shift)
    
    if arr.ndim == 3:
        arr_mask = Parallel(n_jobs=-1)(
            delayed(_mask_array)(
                img, mask, yx_shift[z],
                ) 
            for z, img in enumerate(stack)
            )
        arr_mask = np.stack(arr_mask)
    
    return arr_mask

# -----------------------------------------------------------------------------

stack_norm = normalize_array(stack, med_proj, yx_shift, mask=mtx_mask)
# stack_norm = mask_array(stack, mtx_mask, yx_shift)

#%%

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(stack_norm)


