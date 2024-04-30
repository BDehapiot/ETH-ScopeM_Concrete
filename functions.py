#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.filters import median
from skimage.morphology import disk, ball

# Scipy
from scipy.ndimage import shift

#%% Function(s) ---------------------------------------------------------------

def shift_stack(stack, centroids, reverse=False):
    
    def shift_img(img, centroid):
        if reverse:         
            centroid = [- centroid[0], - centroid[1]]
        if img.dtype == bool:
            img = img.astype("uint8")
        return shift(img, centroid)
    
    # Shift 1 image / 1 centroid
    if stack.ndim == 2 and len(centroids) == 1:
        stack = shift_img(stack, centroids)

    # Shift 1 image / n centroids
    elif stack.ndim == 2 and len(centroids) > 1:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(stack, centroid) 
            for centroid in centroids
            )
        stack = np.stack(stack)
        
    # Shift n images / n centroids
    elif stack.ndim == 3:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(img, centroid) 
            for img, centroid in zip(stack, centroids)
            )
        stack = np.stack(stack)
        
    return stack

def norm_stack(stack, med_proj, centroids, radius=1, mask=None):
    
    def filt_median(img):
        return median(img, footprint=disk(radius))
        
    if radius > 1:
        stack = Parallel(n_jobs=-1)(
            delayed(filt_median)(img) 
            for img in stack
            )

    med_proj = shift_stack(med_proj, centroids, reverse=True)        
    stack = np.divide(stack, med_proj, where=med_proj != 0)

    if mask is not None:
        mask = shift_stack(mask, centroids, reverse=True)
        stack *= mask
        
    return stack
    
#     return stack_norm
    
    #     def _normalize_stack(img, med_proj, centroids, mask=mask):
    #         centroids = [centroids[0] * -1, centroids[1] * -1]
    #         med_proj = shift(med_proj, centroids)
    #         img_norm = np.divide(img, med_proj, where=med_proj!=0)
    #         if radius > 1:
    #             img_norm = median(img_norm, footprint=disk(radius))
    #         if mask is not None:
    #             mask = shift(mask.astype("uint8"), centroids)
    #             img_norm *= mask
    #         return img_norm
        
    #     if stack.ndim == 2:
    #         stack_norm = _normalize_stack(stack, med_proj, centroids, mask=mask)
        
    #     if stack.ndim == 3:
    #         stack_norm = Parallel(n_jobs=-1)(
    #             delayed(_normalize_stack)(
    #                 img, med_proj, centroids[z], mask=mask
    #                 ) 
    #             for z, img in enumerate(stack)
    #             )
    #         stack_norm = np.stack(stack_norm)
        
    #     return stack_norm


#%%

# t = 0; df = 4
# data_path = Path("D:/local_Concrete/data")
# stack_path = data_path / "D1_ICONX_DoS" / f"D1_ICONX_DoS_Time{t}_crop_df{df}.tif"
# metadata_path = data_path / "D1_ICONX_DoS" / f"D1_ICONX_DoS_Time{t}_metadata_oo.pkl"
# stack = io.imread(stack_path)
# with open(metadata_path, 'rb') as file:
#     metadata = pickle.load(file)
    
# idx = metadata["dfs"].index(df)
# med_proj = metadata["med_projs"][idx]
# mtx_mask = metadata["mtx_masks"][idx]
# centroids = metadata["centroidss"][idx]
    
# t0 = time.time()
# print(" - Test :", end='')

# stack_norm = normalize_stack(stack, med_proj, centroids, mask=mtx_mask)

# t1 = time.time()
# print(f" {(t1-t0):<5.2f}s") 

# -----------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(stack_norm)
# viewer.add_image(stack)
# viewer.add_image(mtx_mask_3D * 255, blending="additive", colormap="bop orange", opacity=0.1)

#%%

# def normalize_stack(stack, med_proj, centroids, radius=1, mask=None):
    
#     def _normalize_stack(img, med_proj, centroids, mask=mask):
#         centroids = [centroids[0] * -1, centroids[1] * -1]
#         med_proj = shift(med_proj, centroids)
#         img_norm = np.divide(img, med_proj, where=med_proj!=0)
#         if radius > 1:
#             img_norm = median(img_norm, footprint=disk(radius))
#         if mask is not None:
#             mask = shift(mask.astype("uint8"), centroids)
#             img_norm *= mask
#         return img_norm
    
#     if stack.ndim == 2:
#         stack_norm = _normalize_stack(stack, med_proj, centroids, mask=mask)
    
#     if stack.ndim == 3:
#         stack_norm = Parallel(n_jobs=-1)(
#             delayed(_normalize_stack)(
#                 img, med_proj, centroids[z], mask=mask
#                 ) 
#             for z, img in enumerate(stack)
#             )
#         stack_norm = np.stack(stack_norm)
    
#     return stack_norm

# def mask_stack(stack, mask, centroids):
    
#     def _mask_array(img, mask, centroids):
#         centroids = [centroids[0] * -1, centroids[1] * -1]
#         mask = shift(mask.astype("uint8"), centroids)
#         img_mask = img * mask
#         return img_mask
    
#     if stack.ndim == 2:
#         stack_mask = _mask_array(stack, mask, centroids)
    
#     if stack.ndim == 3:
#         stack_mask = Parallel(n_jobs=-1)(
#             delayed(_mask_array)(
#                 img, mask, centroids[z],
#                 ) 
#             for z, img in enumerate(stack)
#             )
#         stack_mask = np.stack(stack_mask)
    
#     return stack_mask


    
