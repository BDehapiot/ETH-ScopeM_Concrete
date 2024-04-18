#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.filters import median
from joblib import Parallel, delayed
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.signal import find_peaks, peak_prominences
from skimage.transform import rescale, downscale_local_mean
from skimage.morphology import (
    disk, ball, binary_dilation, binary_erosion, 
    remove_small_holes, remove_small_objects,
    )
from scipy.ndimage import (
    gaussian_filter1d, binary_fill_holes, distance_transform_edt, 
    maximum_filter,
    )

#%% Parameters ----------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
# stack_name = "D1_ICONX_DoS"
# stack_name = "D11_ICONX_DoS"
stack_name = "H9_ICONX_DoS"

rsize_factor = 8 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Initialize ----------------------------------------------------------------

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

def roll_image(img_rsize):
    idx = np.argwhere((img_rsize > 30000) == 1)
    y0, x0 = img_rsize.shape[0] // 2, img_rsize.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img_rsize, yx_shift, mode='wrap'), yx_shift 

def get_2Dmasks(stack_roll):
    
    # Intensity distribution
    avg_proj = np.mean(stack_roll, axis=0)
    hist, bins = np.histogram(
        avg_proj.flatten(), bins=1024, range=(0, 65535))    
    hist = gaussian_filter1d(hist, sigma=2)
    pks, _ = find_peaks(hist, distance=30)
    proms = peak_prominences(hist, pks)[0]
    sorted_pks = pks[np.argsort(proms)[::-1]]
    select_pks = sorted_pks[:3]
    
    # Get masks
    mtx_thresh = bins[select_pks[1]] - (
        (bins[select_pks[1]] - bins[select_pks[0]]) / 2)
    rod_thresh = bins[select_pks[2]] - (
        (bins[select_pks[2]] - bins[select_pks[1]]) / 2)
    mtx_thresh *= mtx_thresh_coeff
    rod_thresh *= rod_thresh_coeff
    mtx_mask = avg_proj >= mtx_thresh
    rod_mask = avg_proj >= rod_thresh
    rod_mask = binary_fill_holes(rod_mask)
    rod_mask = binary_dilation(rod_mask, footprint=disk(3))
    mtx_mask = mtx_mask ^ rod_mask
    
    return avg_proj, mtx_thresh, rod_thresh, mtx_mask, rod_mask

def get_3Dmasks(
        img_rsize, yx_shift, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM):
    
    yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
    
    # Shift 2D masks
    avg_proj = shift(avg_proj, yx_shift)
    rod_mask = shift(rod_mask.astype("uint8"), yx_shift)
    mtx_mask = shift(mtx_mask.astype("uint8"), yx_shift)
    rod_EDM = shift(rod_EDM, yx_shift)
    mtx_EDM = shift(mtx_EDM, yx_shift)
    
    # Normalize img
    img_norm = np.divide(img_rsize, avg_proj, where=avg_proj!=0)
    img_norm = median(img_norm, footprint=disk(5)) # parameter
    img_norm *= mtx_mask
        
    return img_norm, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM

def get_object_EDM(idx, obj_labels_3D, mtx_mask_3D):
    
    # Measure object EDM avg
    labels = obj_labels_3D.copy()
    labels[labels == idx] = 0
    obj_EDM_3D = distance_transform_edt(1 - labels > 0)
    obj_EDM_3D[mtx_mask_3D == 0] = 0
    obj_EDM_avg = np.mean(obj_EDM_3D[obj_labels_3D == idx])
    
    return obj_EDM_avg

def get_object_properties(stack_norm, mtx_mask_3D, mtx_EDM_3D):
    
    # Get object mask and labels
    obj_mask_3D = (stack_norm < 0.8) & (stack_norm > 0) # parameter
    obj_mask_3D = remove_small_objects(obj_mask_3D, min_size=512) # parameter
    obj_mask_3D = clear_border(obj_mask_3D)
    obj_labels_3D = label(obj_mask_3D)
    
    # Get object properties
    obj_props = []
    mtx_EDM_3D /= np.max(mtx_EDM_3D)
    props = regionprops(obj_labels_3D, intensity_image=mtx_EDM_3D)
    for prop in props:
        obj_props.append((
            prop.label,
            prop.centroid,
            prop.area,
            prop.intensity_mean,
            prop.solidity,
            )) 
        
    # Get object EDM
    idxs = np.unique(obj_labels_3D)[1:]
    obj_EDM_avg = Parallel(n_jobs=-1)(
            delayed(get_object_EDM)(idx, obj_labels_3D, mtx_mask_3D) 
            for idx in idxs
            )
    
    # Merge properties
    obj_props = [
        data + (obj_EDM_avg[i],) for i, data in enumerate(obj_props)]
    
    return obj_mask_3D, obj_labels_3D, obj_props

# -----------------------------------------------------------------------------

def process_stack(stack_path, stack_data):
    
    # Initialize
    print(f"\n{stack_path.stem}")
    print( "  ---------")
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Resize stack
    print("  Resize  :", end='')
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
    
    # Roll stack
    print("  Roll    :", end='')
    t0 = time.time()
    outputs = Parallel(n_jobs=-1)(
            delayed(roll_image)(img_rsize) 
            for img_rsize in stack_rsize
            )
    stack_roll = np.stack([data[0] for data in outputs])
    yx_shifts = [data[1] for data in outputs]
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Get 2D masks
    print("  2Dmasks :", end='')
    t0 = time.time()
    (
     avg_proj,
     mtx_thresh,
     rod_thresh,
     mtx_mask,
     rod_mask
     ) = get_2Dmasks(stack_roll)
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 

    # Get EDM
    mtx_EDM = distance_transform_edt(mtx_mask | rod_mask)
    rod_EDM = distance_transform_edt(~rod_mask)
    
    # Rescale data
    if stack_data:
        rscale_factor = np.sqrt(
            np.sum(stack_data[0]["rod_mask"]) / np.sum(rod_mask))
        print("  Rescale :", end='')
        t0 = time.time()
        stack_rsize = rescale(stack_rsize, rscale_factor)
        avg_proj = rescale(avg_proj, rscale_factor)
        mtx_mask = rescale(mtx_mask, rscale_factor, order=0)
        rod_mask = rescale(rod_mask, rscale_factor, order=0)
        mtx_EDM = rescale(mtx_EDM, rscale_factor)
        rod_EDM = rescale(rod_EDM, rscale_factor)
        t1 = time.time()
        print(f" {(t1-t0):<5.2f}s") 
        
    # Get 3D masks
    print("  3Dmasks :", end='')
    t0 = time.time()
    outputs = Parallel(n_jobs=-1)(
            delayed(get_3Dmasks)(
                img_rsize, 
                yx_shift, 
                avg_proj, 
                rod_mask, 
                mtx_mask, 
                rod_EDM, 
                mtx_EDM
                ) 
            for img_rsize, yx_shift in zip(stack_rsize, yx_shifts)
            )    
    stack_norm  = np.stack([data[0] for data in outputs])
    avg_proj_3D = np.stack([data[1] for data in outputs])
    rod_mask_3D = np.stack([data[2] for data in outputs])
    mtx_mask_3D = np.stack([data[3] for data in outputs])
    rod_EDM_3D  = np.stack([data[4] for data in outputs])
    mtx_EDM_3D  = np.stack([data[5] for data in outputs])    
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Get object properties
    print("  Objects :", end='')
    t0 = time.time()
    obj_mask_3D, obj_labels_3D, obj_props = get_object_properties(
        stack_norm, mtx_mask_3D, mtx_EDM_3D)
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
         
    # Print variables
    print( "  ---------")
    print(f"  zSlices    : {z0}-{z1}")
    print(f"  mtx_thresh : {int(mtx_thresh):<5d}")
    print(f"  rod_thresh : {int(rod_thresh):<5d}")
    if stack_data:
        print(f"  rscaleF    : {rscale_factor:<.3f}")

    # Outputs
    stack_data.append({
        "stack_path"    : stack_path,
        "stack_rsize"   : stack_rsize,
        "stack_roll"    : stack_roll,
        "stack_norm"    : stack_norm,
        "yx_shifts"     : yx_shifts,
        "avg_proj"      : avg_proj,
        "mtx_thresh"    : mtx_thresh,
        "rod_thresh"    : rod_thresh,
        "mtx_mask"      : mtx_mask,
        "rod_mask"      : rod_mask,
        "mtx_EDM"       : mtx_EDM,
        "rod_EDM"       : rod_EDM,
        "avg_proj_3D"   : avg_proj_3D,
        "rod_mask_3D"   : rod_mask_3D,
        "mtx_mask_3D"   : mtx_mask_3D,
        "rod_EDM_3D"    : rod_EDM_3D,
        "mtx_EDM_3D"    : mtx_EDM_3D,
        "obj_mask_3D"   : obj_mask_3D,
        "obj_labels_3D" : obj_labels_3D,
        "obj_props"     : obj_props,
        })

#%% Execute -------------------------------------------------------------------

# Execute
stack_data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        process_stack(stack_path, stack_data)
            
#%%

idx = 1
obj_props_ref = stack_data[0]["obj_props"]
obj_props_reg = stack_data[idx]["obj_props"]
stack_rsize_ref = stack_data[0]["stack_norm"]
stack_rsize_reg = stack_data[idx]["stack_norm"]

# -----------------------------------------------------------------------------

obj_tests = []
for obj_prop_ref in obj_props_ref:
    obj_test = []
    for obj_prop_reg in obj_props_reg:
        tmp = [
            obj_prop_ref[2] / obj_prop_reg[2],
            obj_prop_ref[3] / obj_prop_reg[3],
            obj_prop_ref[4] / obj_prop_reg[4],
            obj_prop_ref[5] / obj_prop_reg[5],
            ]
        obj_test_avg = np.abs(1 - np.mean(tmp))
        obj_test_std = np.std(tmp)
        obj_test.append((obj_test_avg + obj_test_std) / 2)
    obj_tests.append(obj_test)      

obj_coords_ref, obj_coords_reg = [], []
for idx_ref, obj_test in enumerate(obj_tests):
    idx_reg = np.argmin(obj_test)
    score = np.min(obj_test)
    if score < 0.03: # parameter
        obj_coords_ref.append(obj_props_ref[idx_ref][1])
        obj_coords_reg.append(obj_props_reg[idx_reg][1])
obj_coords_ref = np.stack(obj_coords_ref)
obj_coords_reg = np.stack(obj_coords_reg)
        
# -----------------------------------------------------------------------------

from scipy import ndimage
from scipy.linalg import lstsq

def affine_registration(ref, reg):
    
    if ref.shape[0] < ref.shape[1]:
        ref = ref.T
        reg = reg.T
    (n, dim) = ref.shape
    
    # Compute least squares
    p, res, rnk, s = lstsq(np.hstack((ref, np.ones([n, 1]))), reg)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    
    # Merge translation and transform matrix
    affine_transform = np.eye(4)
    affine_transform[:3, :3] = T
    affine_transform[:3, 3] = t
    
    return affine_transform

# Compute and apply affine registration
affine_transform = affine_registration(obj_coords_ref, obj_coords_reg)
stack_rsize_reg = ndimage.affine_transform(stack_rsize_reg, affine_transform)

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(stack_rsize_reg)
viewer.add_image(stack_rsize_ref)

# io.imsave(
#     Path(data_path, f"stack_rsize_reg.tif"),
#     stack_rsize_reg.astype("float32"), check_contrast=False,
#     )
# io.imsave(
#     Path(data_path, f"stack_rsize_ref.tif"),
#     stack_rsize_ref.astype("float32"), check_contrast=False,
#     )

#%% Save ----------------------------------------------------------------------

for data in stack_data:
    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rsize.tif"),
        data["stack_rsize"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_roll.tif"),
        data["stack_roll"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_norm.tif"),
        data["stack_norm"].astype("float32"), check_contrast=False,
        )
    
    # -------------------------------------------------------------------------
    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_avg_proj.tif"),
        data["avg_proj"].astype("float32"), check_contrast=False,
        )    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rod_mask.tif"),
        data["rod_mask"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mtx_mask.tif"),
        data["mtx_mask"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rod_EDM.tif"),
        data["rod_EDM"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mtx_EDM.tif"),
        data["mtx_EDM"].astype("float32"), check_contrast=False,
        )
    
    # -------------------------------------------------------------------------
   
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_avg_proj_3D.tif"),
        data["avg_proj_3D"].astype("float32"), check_contrast=False,
        )    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rod_mask_3D.tif"),
        data["rod_mask_3D"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mtx_mask_3D.tif"),
        data["mtx_mask_3D"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rod_EDM_3D.tif"),
        data["rod_EDM_3D"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mtx_EDM_3D.tif"),
        data["mtx_EDM_3D"].astype("float32"), check_contrast=False,
        )

#%%

# def get_object_EDM(idx, obj_labels_3D, mtx_mask_3D):
    
#     # Measure object EDM avg
#     labels = obj_labels_3D.copy()
#     labels[labels == idx] = 0
#     obj_EDM_3D = distance_transform_edt(1 - labels > 0)
#     obj_EDM_3D[mtx_mask_3D == 0] = 0
#     obj_EDM_avg = np.mean(obj_EDM_3D[obj_labels_3D == idx])
    
#     return obj_EDM_avg

# def get_object_properties(stack_norm, mtx_mask_3D, mtx_EDM_3D):
    
#     # Get object mask and labels
#     obj_mask_3D = (stack_norm < 0.8) & (stack_norm > 0) # parameter
#     obj_mask_3D = remove_small_objects(obj_mask_3D, min_size=256) # parameter
#     obj_mask_3D = clear_border(obj_mask_3D)
#     obj_labels_3D = label(obj_mask_3D)
    
#     # Get object properties
#     obj_props = []
#     mtx_EDM_3D /= np.max(mtx_EDM_3D)
#     props = regionprops(obj_labels_3D, intensity_image=mtx_EDM_3D)
#     for prop in props:
#         obj_props.append((
#             prop.label,
#             prop.centroid,
#             prop.area,
#             prop.intensity_mean,
#             prop.solidity,
#             )) 
        
#     # Get object EDM
#     idxs = np.unique(obj_labels_3D)[1:]
#     obj_EDM_avg = Parallel(n_jobs=-1)(
#             delayed(get_object_EDM)(idx, obj_labels_3D, mtx_mask_3D) 
#             for idx in idxs
#             )
    
#     # Merge properties
#     obj_props = [
#         data + (obj_EDM_avg[i],) for i, data in enumerate(obj_props)]
    
#     return obj_mask_3D, obj_labels_3D, obj_props

# stack_norm = stack_data[0]["stack_norm"]
# mtx_mask_3D = stack_data[0]["mtx_mask_3D"]
# mtx_EDM_3D = stack_data[0]["mtx_EDM_3D"]

# obj_mask_3D, obj_labels_3D, obj_props = get_object_properties(
#     stack_norm, mtx_mask_3D, mtx_EDM_3D)