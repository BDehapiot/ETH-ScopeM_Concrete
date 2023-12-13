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
stack_name = "D11_ICONX_DoS"
# stack_name = "H9_ICONX_DoS"

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

def roll_image(img):
    idx = np.argwhere((img > 30000) == 1)
    y0, x0 = img.shape[0] // 2, img.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img, yx_shift, mode='wrap'), yx_shift 

def get_2Dmasks(stack):
    
    # Intensity distribution
    avg_proj = np.mean(stack, axis=0)
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

def get_3Dmasks(img, yx_shift, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM):
    
    yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
    
    # Shift 2D masks
    avg_proj = shift(avg_proj, yx_shift)
    rod_mask = shift(rod_mask.astype("uint8"), yx_shift)
    mtx_mask = shift(mtx_mask.astype("uint8"), yx_shift)
    rod_EDM = shift(rod_EDM, yx_shift)
    mtx_EDM = shift(mtx_EDM, yx_shift)
    
    # Normalize img
    img_norm = np.divide(img, avg_proj, where=avg_proj!=0)
    img_norm = median(img_norm, footprint=disk(5)) # test
    img_norm *= mtx_mask
    
    return img_norm, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM

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
            delayed(roll_image)(img) 
            for img in stack_rsize
            )
    stack_roll = np.stack([data[0] for data in outputs])
    yx_shifts = [data[1] for data in outputs]
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Get 2D masks
    print("  2Dmasks :", end='')
    t0 = time.time()
    avg_proj, mtx_thresh, rod_thresh, mtx_mask, rod_mask = get_2Dmasks(stack_roll)
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
                img, yx_shift, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM
                ) 
            for img, yx_shift in zip(stack_rsize, yx_shifts)
            )
    
    stack_norm  = np.stack([data[0] for data in outputs])
    avg_proj_3D = np.stack([data[1] for data in outputs])
    rod_mask_3D = np.stack([data[2] for data in outputs])
    mtx_mask_3D = np.stack([data[3] for data in outputs])
    rod_EDM_3D  = np.stack([data[4] for data in outputs])
    mtx_EDM_3D  = np.stack([data[5] for data in outputs])

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
        "stack_path"   : stack_path,
        "stack_rsize"  : stack_rsize,
        "stack_roll"   : stack_roll,
        "stack_norm"   : stack_norm,
        "yx_shifts"    : yx_shifts,
        "avg_proj"     : avg_proj,
        "mtx_thresh"   : mtx_thresh,
        "rod_thresh"   : rod_thresh,
        "mtx_mask"     : mtx_mask,
        "rod_mask"     : rod_mask,
        "mtx_EDM"      : mtx_EDM,
        "rod_EDM"      : rod_EDM,
        "avg_proj_3D"  : avg_proj_3D,
        "rod_mask_3D"  : rod_mask_3D,
        "mtx_mask_3D"  : mtx_mask_3D,
        "rod_EDM_3D"   : rod_EDM_3D,
        "mtx_EDM_3D"   : mtx_EDM_3D,
        })

#%%

# Execute
stack_data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        process_stack(stack_path, stack_data)
        
# Save
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

idxA, idxB = 0, 2
stack_norm1 = stack_data[idxA]["stack_norm"]
stack_norm2 = stack_data[idxB]["stack_norm"]
mtx_mask1_3D = stack_data[idxA]["mtx_mask_3D"]
mtx_mask2_3D = stack_data[idxB]["mtx_mask_3D"]
mtx_EDM1_3D = stack_data[idxA]["mtx_EDM_3D"]
mtx_EDM2_3D = stack_data[idxB]["mtx_EDM_3D"]
mtx_EDM1_3D /= np.max(mtx_EDM1_3D)
mtx_EDM2_3D /= np.max(mtx_EDM2_3D)

# Binarize & labels
obj_mask1 = (stack_norm1 < 0.8) & (stack_norm1 > 0)
obj_mask1 = remove_small_objects(obj_mask1, min_size=512) #
obj_mask1 = clear_border(obj_mask1)
obj_mask2 = (stack_norm2 < 0.8) & (stack_norm2 > 0)
obj_mask2 = remove_small_objects(obj_mask2, min_size=512) #
obj_mask2 = clear_border(obj_mask2)
labels1 = label(obj_mask1)
labels2 = label(obj_mask2)

# -----------------------------------------------------------------------------

# Object EDMs
def get_object_EDM(idx, labels, mtx_mask_3D):
    tmp_labels = labels.copy()
    tmp_labels[tmp_labels == idx] = 0
    obj_EDM = distance_transform_edt(1 - tmp_labels > 0)
    obj_EDM[mtx_mask_3D == 0] = 0
    return obj_EDM

idx1 = np.unique(labels1)[1:]
obj_EDM1 = Parallel(n_jobs=-1)(
        delayed(get_object_EDM)(idx, labels1, mtx_mask1_3D) 
        for idx in idx1
        )
obj_EDM1 = np.stack(obj_EDM1)

idx2 = np.unique(labels2)[1:]
obj_EDM2 = Parallel(n_jobs=-1)(
        delayed(get_object_EDM)(idx, labels2, mtx_mask2_3D) 
        for idx in idx2
        )
obj_EDM2 = np.stack(obj_EDM2)

obj_EDM1_avg = []
for idx in idx1:
    obj_EDM1_avg.append(np.mean(obj_EDM1[idx - 1][labels1 == idx]))
    
obj_EDM2_avg = []
for idx in idx2:
    obj_EDM2_avg.append(np.mean(obj_EDM2[idx - 1][labels2 == idx]))

# import napari
# viewer = napari.Viewer()
# viewer.add_image(obj_EDM1)
# viewer.add_image(obj_EDM2)

# -----------------------------------------------------------------------------

# Get properties
def get_properties(array, intensity_image):
    properties = []
    labels = label(array)
    props = regionprops(labels, intensity_image=intensity_image)
    for prop in props:
        properties.append((
            prop.label,
            prop.centroid,
            prop.area,
            prop.intensity_mean,
            prop.solidity,
            )) 
    return properties

properties1 = get_properties(obj_mask1, mtx_EDM1_3D)
properties2 = get_properties(obj_mask2, mtx_EDM2_3D)
properties1 = [data + (obj_EDM1_avg[i],) for i, data in enumerate(properties1)]
properties2 = [data + (obj_EDM2_avg[i],) for i, data in enumerate(properties2)]

# -----------------------------------------------------------------------------

test1 = np.stack([data[2:] for data in properties1])
test2 = np.stack([data[2:] for data in properties2])

tests = []
for idx1, prop1 in enumerate(properties1):
    test = []
    for idx2, prop2 in enumerate(properties2):
        tmp = [
            prop1[2] / prop2[2],
            prop1[3] / prop2[3],
            prop1[4] / prop2[4],
            prop1[5] / prop2[5],
            ]
        test_avg = np.abs(1 - np.mean(tmp))
        test_std = np.std(tmp)
        test.append((test_avg + test_std) / 2)
    tests.append(test)      

landmarks1, landmarks2 = [], []
labels1_match, labels2_match = np.zeros_like(labels1), np.zeros_like(labels2)
for idx1, test in enumerate(tests):
    idx2 = np.argmin(test)
    score = np.min(test)
    print(
        f"obj-{idx1:03d} / obj-{idx2:03d} / "
        f"score = {score:.3f}"
        )
    if score < 0.04:
        labels1_match[labels1 == idx1 + 1] = idx1 + 1
        labels2_match[labels2 == idx2 + 1] = idx1 + 1
        landmarks1.append(properties1[idx1][1])
        landmarks2.append(properties2[idx2][1])
    
landmarks1 = np.stack(landmarks1)
landmarks2 = np.stack(landmarks2)
        
# Display
import napari
viewer = napari.Viewer()
viewer.add_labels(labels1_match)
viewer.add_labels(labels2_match)

#%%

from scipy.linalg import lstsq

# Inputs:
# - P, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# - Q, a (n,dim) [or (dim,n)] matrix, a point cloud of n points in dim dimension.
# P and Q must be of the same shape.
# This function returns :
# - Pt, the P point cloud, transformed to fit to Q
# - (T,t) the affine transform

def affine_registration(P, Q):
    transposed = False
    if P.shape[0] < P.shape[1]:
        transposed = True
        P = P.T
        Q = Q.T
    (n, dim) = P.shape
    # Compute least squares
    p, res, rnk, s = lstsq(np.hstack((P, np.ones([n, 1]))), Q)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    # Compute transformed pointcloud
    Pt = P@T.T + t
    if transposed: Pt = Pt.T
    return Pt, (T, t)

Pt, (T, t) = affine_registration(landmarks1, landmarks2)

#%%

from scipy import ndimage

stack_rsize1 = stack_data[idxA]["stack_rsize"]
stack_rsize2 = stack_data[idxB]["stack_rsize"]

# Create a 4x4 affine transformation matrix
affine_transform = np.eye(4)
affine_transform[:3, :3] = T
affine_transform[:3, 3] = t

# Apply the transformation
transformed_image = ndimage.affine_transform(stack_rsize2, affine_transform)

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(transformed_image)
viewer.add_image(stack_rsize1)