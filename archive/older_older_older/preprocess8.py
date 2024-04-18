#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from pystackreg import StackReg
from scipy.ndimage import shift
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from skimage.transform import downscale_local_mean
from scipy.signal import find_peaks, peak_prominences

#%% Comments ------------------------------------------------------------------

'''

- There is stack to stack variations regarding brightness and pixel size.

'''

#%% Parameters ----------------------------------------------------------------

stack_idx = 7
rsize_factor = 4
thresh1_coeff = 1.0
thresh2_coeff = 1.0
data_path = "D:/local_Concrete/data/DIA"
stack_name = "D1_ICONX_DoS"

#%% Initialize ----------------------------------------------------------------

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)

#%% Functions -----------------------------------------------------------------

def process_stack(stack_path):
    
    # Nested functions --------------------------------------------------------
    
    def resize_image(img_path):
        return downscale_local_mean(io.imread(img_path), rsize_factor)
    
    def roll_image(img):
        idx = np.argwhere((img > 30000) == 1)
        y0, x0 = img.shape[0] // 2, img.shape[1] // 2
        y1, x1 = np.mean(idx, axis=0)
        return shift(img, shift=[y0 - y1, x0 - x1], mode='wrap')
    
    # Execute -----------------------------------------------------------------
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Initialize
    print(stack_path.stem)
            
    # Resize stack
    print("  Resize :", end='')
    t0 = time.time()
    stack = Parallel(n_jobs=-1)(
            delayed(resize_image)(img_path) 
            for img_path in img_paths
            )
    stack = np.stack(stack)
    stack = downscale_local_mean(stack, (rsize_factor, 1, 1))
    t1 = time.time()
    print(f" {(t1-t0):5.2f} s") 
    
    # Select slices
    z_mean = np.mean(stack, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
    print(f"  Select : {z0}-{z1}")
    stack = stack[z0:z1, ...]   
    
    # Roll stack
    print("  Roll   :", end='')
    t0 = time.time()
    stack = Parallel(n_jobs=-1)(
            delayed(roll_image)(img) 
            for img in stack
            )
    stack = np.stack(stack)
    t1 = time.time()
    print(f" {(t1-t0):5.2f} s") 
    
    # Pixel intensity distribution
    avgProj = np.mean(stack, axis=0)
    hist, bins = np.histogram(
        avgProj.flatten(), bins=1024, range=(0, 65535))    
    hist = gaussian_filter1d(hist, sigma=2)
    pks, _ = find_peaks(hist, distance=20)
    proms = peak_prominences(hist, pks)[0]
    sorted_pks = pks[np.argsort(proms)[::-1]]
    select_pks = sorted_pks[:3]
    
    # Get masks
    thresh1 = bins[select_pks[1]] - (
        (bins[select_pks[1]] - bins[select_pks[0]]) / 2)
    thresh2 = bins[select_pks[2]] - (
        (bins[select_pks[2]] - bins[select_pks[1]]) / 2)
    thresh1 *= thresh1_coeff
    thresh2 *= thresh2_coeff
    mask1 = avgProj >= thresh1
    mask2 = avgProj >= thresh2
    
    # # Extract zProfiles
    # zProf1, zProf2 = [], []
    # for img in stack:
    #     zProf1.append(np.mean(img[mask1]))
    #     zProf2.append(np.mean(img[mask2]))
    # zProf1 = np.stack(zProf1) / np.mean(zProf1) - 1
    # zProf2 = np.stack(zProf2) / np.mean(zProf2) - 1
    
    # Outputs
    stack_data = {
        "stack"  : stack,
        "avgProj": avgProj,
        "thresh1": thresh1,
        "thresh2": thresh2,
        "mask1"  : mask1,
        "mask2"  : mask2,
        # "zProf1" : zProf1,
        # "zProf2" : zProf2,
        }
        
    return stack_data

#%%

data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        stack_data = process_stack(stack_path)
        data.append(stack_data)
        io.imsave(
            Path(data_path, f"{stack_path.stem}_process.tif"),
            stack_data["stack"].astype("float32"), check_contrast=False,
            )

# data = []
# counter = 0
# for stack_path in stack_paths:
#     if stack_name in stack_path.name: 
#         if counter == 0:
#             stack_data = process_stack(stack_path)
#             data.append(stack_data)
#             io.imsave(
#                 Path(data_path, f"{stack_path.stem}_process.tif"),
#                 stack_data["stack"].astype("float32"), check_contrast=False,
#                 )
#             counter += 1

#%%

from skimage.transform import rescale
from skimage.morphology import disk, binary_dilation

# -----------------------------------------------------------------------------

tp1 = 0
tp2 = 3

rscale_factor = np.sqrt(np.sum(data[tp1]["mask1"]) / np.sum(data[tp2]["mask1"]))
stack = rescale(data[tp2]["stack"], rscale_factor)
 
io.imsave(
    Path(data_path, f"{stack_path.stem}_rescaled{tp2}.tif"),
    stack.astype("float32"), check_contrast=False,
    )

# stack = data[tp]["stack"]
# avgProj = data[tp]["avgProj"]
# mask1 = data[tp]["mask1"]
# mask2 = data[tp]["mask2"]
# mask2 = binary_dilation(mask2, footprint=disk(3))
# mask = mask1 ^ mask2
# mask = mask[np.newaxis, :, :]
# stack = stack * mask
# stack /= avgProj

# io.imsave(
#     Path(data_path, f"{stack_path.stem}_test{tp}.tif"),
#     stack.astype("float32"), check_contrast=False,
#     )