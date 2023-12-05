#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.filters import median
from joblib import Parallel, delayed
from skimage.transform import rescale
from skimage.transform import downscale_local_mean
from scipy.signal import find_peaks, peak_prominences
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
stack_name = "D11_ICONX_DoS"

rsize_factor = 8 # Image size reduction factor
mThresh_coeff = 1.0 # adjust matrix threshold
rThresh_coeff = 1.0 # adjust rod threshold

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
    yxShift = [y0 - y1, x0 - x1]
    return shift(img, yxShift, mode='wrap'), yxShift 

def get_2Dmasks(stack):
    
    # Intensity distribution
    avgProj = np.mean(stack, axis=0)
    hist, bins = np.histogram(
        avgProj.flatten(), bins=1024, range=(0, 65535))    
    hist = gaussian_filter1d(hist, sigma=2)
    pks, _ = find_peaks(hist, distance=30)
    proms = peak_prominences(hist, pks)[0]
    sorted_pks = pks[np.argsort(proms)[::-1]]
    select_pks = sorted_pks[:3]
    
    # Get masks
    mThresh = bins[select_pks[1]] - (
        (bins[select_pks[1]] - bins[select_pks[0]]) / 2)
    rThresh = bins[select_pks[2]] - (
        (bins[select_pks[2]] - bins[select_pks[1]]) / 2)
    mThresh *= mThresh_coeff
    rThresh *= rThresh_coeff
    mMask = avgProj >= mThresh
    rMask = avgProj >= rThresh
    rMask = binary_fill_holes(rMask)
    rMask = binary_dilation(rMask, footprint=disk(3))
    mMask = mMask ^ rMask
    
    return avgProj, mThresh, rThresh, mMask, rMask

def get_3Dmasks(img, yxShift, avgProj, rMask, mMask, rEDM, mEDM):
    
    yxShift = [yxShift[0] * -1, yxShift[1] * -1]
    
    # Shift 2D masks
    avgProj = shift(avgProj, yxShift)
    rMask = shift(rMask.astype("uint8"), yxShift)
    mMask = shift(mMask.astype("uint8"), yxShift)
    rEDM = shift(rEDM, yxShift)
    mEDM = shift(mEDM, yxShift)
    
    # Normalize img
    img_norm = np.divide(img, avgProj, where=avgProj!=0)
    img_norm = median(img_norm, footprint=disk(5)) # test
    img_norm *= mMask
    
    return img_norm, avgProj, rMask, mMask, rEDM, mEDM

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
    yxShifts = [data[1] for data in outputs]
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Get masks
    avgProj, mThresh, rThresh, mMask, rMask = get_2Dmasks(stack_roll)

    # Get EDM
    mEDM = distance_transform_edt(mMask | rMask)
    rEDM = distance_transform_edt(~rMask)
    
    # Rescale data
    if stack_data:
        rscale_factor = np.sqrt(
            np.sum(stack_data[0]["rMask"]) / np.sum(rMask)) # rMask or mMask 
        print("  Rescale :", end='')
        t0 = time.time()
        stack_rsize = rescale(stack_rsize, rscale_factor)
        avgProj = rescale(avgProj, rscale_factor)
        mMask = rescale(mMask, rscale_factor, order=0)
        rMask = rescale(rMask, rscale_factor, order=0)
        mEDM = rescale(mEDM, rscale_factor)
        rEDM = rescale(rEDM, rscale_factor)
        t1 = time.time()
        print(f" {(t1-t0):<5.2f}s") 
        
    # Normalize stack
    print("  Norm    :", end='')
    t0 = time.time()
    outputs = Parallel(n_jobs=-1)(
            delayed(get_3Dmasks)(img, yxShift, avgProj, rMask, mMask, rEDM, mEDM) 
            for img, yxShift in zip(stack_rsize, yxShifts)
            )
    
    stack_norm = np.stack([data[0] for data in outputs])
    avgProj_3D = np.stack([data[1] for data in outputs])
    rMask_3D   = np.stack([data[2] for data in outputs])
    mMask_3D   = np.stack([data[3] for data in outputs])
    rEDM_3D    = np.stack([data[4] for data in outputs])
    mEDM_3D    = np.stack([data[5] for data in outputs])

    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
         
    # Print variables
    print( "  ---------")
    print(f"  zSlices : {z0}-{z1}")
    print(f"  mThresh : {int(mThresh):<5d}")
    print(f"  rThresh : {int(rThresh):<5d}")
    if stack_data:
        print(f"  rscaleF : {rscale_factor:<.3f}")

    # Outputs
    stack_data.append({
        "stack_path"   : stack_path,
        "stack_rsize"  : stack_rsize,
        "stack_roll"   : stack_roll,
        "stack_norm"   : stack_norm,
        "yxShifts"     : yxShifts,
        "avgProj"      : avgProj,
        "mThresh"      : mThresh,
        "rThresh"      : rThresh,
        "mMask"        : mMask,
        "rMask"        : rMask,
        "mEDM"         : mEDM,
        "rEDM"         : rEDM,
        "avgProj_3D"   : avgProj_3D,
        "rMask_3D"     : rMask_3D,
        "mMask_3D"     : mMask_3D,
        "rEDM_3D"      : rEDM_3D,
        "mEDM_3D"      : mEDM_3D,
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
        Path(data_path, f"{data['stack_path'].stem}_avgProj.tif"),
        data["avgProj"].astype("float32"), check_contrast=False,
        )    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rMask.tif"),
        data["rMask"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mMask.tif"),
        data["mMask"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rEDM.tif"),
        data["rEDM"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mEDM.tif"),
        data["mEDM"].astype("float32"), check_contrast=False,
        )
    
    # -------------------------------------------------------------------------
   
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_avgProj_3D.tif"),
        data["avgProj_3D"].astype("float32"), check_contrast=False,
        )    
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rMask_3D.tif"),
        data["rMask_3D"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mMask_3D.tif"),
        data["mMask_3D"].astype("uint8") * 255, check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_rEDM_3D.tif"),
        data["rEDM_3D"].astype("float32"), check_contrast=False,
        )
    io.imsave(
        Path(data_path, f"{data['stack_path'].stem}_mEDM_3D.tif"),
        data["mEDM_3D"].astype("float32"), check_contrast=False,
        )
    
#%%

from skimage.measure import label, regionprops

idxA, idxB = 0, 1
array1 = stack_data[idxA]["stack_norm"]
array2 = stack_data[idxB]["stack_norm"]
mEDM1_3D = stack_data[idxA]["mEDM_3D"]
mEDM2_3D = stack_data[idxB]["mEDM_3D"]
mEDM1_3D /= np.max(mEDM1_3D)
mEDM2_3D /= np.max(mEDM2_3D)

# Binarize
array1 = (array1 < 0.8) & (array1 > 0)
array1 = remove_small_objects(array1, min_size=256) #
array2 = (array2 < 0.8) & (array2 > 0)
array2 = remove_small_objects(array2, min_size=256) #

#
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

properties1 = get_properties(array1, mEDM1_3D)
properties2 = get_properties(array2, mEDM2_3D)

test1 = np.stack([data[2:] for data in properties1])
test2 = np.stack([data[2:] for data in properties2])

# Display
# import napari
# viewer = napari.Viewer()
# # viewer.add_image(array2, colormap="green", rendering="attenuated_mip")
# viewer.add_image(array1, colormap="gray", rendering="attenuated_mip")
# viewer.add_labels(labels1)