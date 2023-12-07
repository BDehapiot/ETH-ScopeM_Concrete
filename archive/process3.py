#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from pystackreg import StackReg
from scipy.ndimage import shift
from joblib import Parallel, delayed
from skimage.transform import rescale
from skimage.transform import downscale_local_mean
from scipy.signal import find_peaks, peak_prominences
from skimage.morphology import (
    disk, ball, binary_dilation, binary_erosion, remove_small_holes,
    )
from scipy.ndimage import (
    gaussian_filter1d, binary_fill_holes, distance_transform_edt, 
    maximum_filter,
    )

#%% Parameters ----------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
stack_name = "D11_ICONX_DoS"

rsize_factor = 4 # Image size reduction factor
mThresh_coeff = 1.0 # adjust matrix threshold
rThresh_coeff = 1.0 # adjust rod threshold

minHeight = 1.8
minDist = 40 / rsize_factor
minBord = 20 / rsize_factor

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

def get_masks(stack):
    
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
    avgProj, mThresh, rThresh, mMask, rMask = get_masks(stack_roll)

    # Get EDM
    mEDM = distance_transform_edt(mMask)
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
        "yxShifts"     : yxShifts,
        "avgProj"      : avgProj,
        "mThresh"      : mThresh,
        "rThresh"      : rThresh,
        "mMask"        : mMask,
        "rMask"        : rMask,
        "mEDM"         : mEDM,
        "rEDM"         : rEDM,
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
    # io.imsave(
    #     Path(data_path, f"{data['stack_path'].stem}_roll.tif"),
    #     data["stack_roll"].astype("float32"), check_contrast=False,
    #     )
    # io.imsave(
    #     Path(data_path, f"{data['stack_path'].stem}_rMask.tif"),
    #     data["rMask"].astype("uint8") * 255, check_contrast=False,
    #     )
    # io.imsave(
    #     Path(data_path, f"{data['stack_path'].stem}_mMask.tif"),
    #     data["mMask"].astype("uint8") * 255, check_contrast=False,
    #     )
    # io.imsave(
    #     Path(data_path, f"{data['stack_path'].stem}_rEDM.tif"),
    #     data["rEDM"].astype("float32"), check_contrast=False,
    #     )
    # io.imsave(
    #     Path(data_path, f"{data['stack_path'].stem}_mEDM.tif"),
    #     data["mEDM"].astype("float32"), check_contrast=False,
    #     )
        
#%%

idxA, idxB = 0, 1
stackA = stack_data[idxA]["stack_rsize"]
stackB = stack_data[idxB]["stack_rsize"]
