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
from scipy.ndimage import gaussian_filter1d
from skimage.transform import downscale_local_mean
from skimage.morphology import disk, binary_dilation
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
    
    def get_correction_plane(stack):
        nZ, nY, nX = stack.shape
        idx0 = np.argwhere((stack[ 0,...] > 30000) == 1)
        idx1 = np.argwhere((stack[-1,...] > 30000) == 1)
        y0, x0 = np.mean(idx0, axis=0)
        y1, x1 = np.mean(idx1, axis=0)
        slope = np.arctan(np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / nZ)
        orient = np.arctan2(y1 - y0, x1 - x0)
        xx, yy = np.meshgrid(np.arange(nX), np.arange(nY))
        cPlane = np.tan(slope) * ((xx - nX // 2) * np.cos(orient) + (yy - nY // 2) * np.sin(orient))
        cPlane *= -1
        return cPlane

    def correct_stack(stack, tPlane):
        
        # Pad stack
        zpad = int(np.max(tPlane)) + 1
        pad = ((zpad, zpad), (0, 0), (0, 0))
        pStack = np.pad(stack, pad, mode='constant', constant_values=np.nan)

        # Correct stack
        cStack = []
        yxIdx = np.nonzero(tPlane)
        for i, img in enumerate(pStack):
            if not np.any(np.isnan(img)):
                zIdx = np.round(tPlane[yxIdx]).astype(int) + i
                idx = tuple((zIdx, yxIdx[0], yxIdx[1]))
                cImg = np.zeros_like(img)
                cImg[yxIdx] = pStack[idx]
                cStack.append(cImg)
        cStack = np.stack(cStack)
        
        return cStack
    
    # def register_img(ref, img):
    #     tmp_ref = ref[
    #         int(ref.shape[0] * 0.25): int(ref.shape[0] * 0.75),
    #         int(ref.shape[1] * 0.25): int(ref.shape[1] * 0.75)   
    #         ]
    #     tmp_img = img[
    #         int(img.shape[0] * 0.25): int(img.shape[0] * 0.75),
    #         int(img.shape[1] * 0.25): int(img.shape[1] * 0.75)   
    #         ]
    #     sr.register(tmp_ref, tmp_img)
    #     return sr.transform(img)
    
    # Execute -----------------------------------------------------------------
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Initialize
    print(stack_path.stem)
    # sr = StackReg(StackReg.TRANSLATION)
            
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
          
    # Get correction plane (cPlane)
    cPlane = get_correction_plane(stack)
    
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
    mask2 = binary_dilation(mask2, footprint=disk(3))
    mask = mask1 ^ mask2
    
    # Correct stack
    stack = correct_stack(stack, cPlane)
    idx = np.any(np.isnan(stack), axis=(1, 2))
    stack = stack[~idx]
       
    # Outputs
    data = {
        "stack"  : stack,
        "cPlane" : cPlane,
        "avgProj": avgProj,
        "thresh1": thresh1,
        "thresh2": thresh2,
        "mask1"  : mask1,
        "mask2"  : mask2,
        "mask"  : mask,
        }
        
    return data

#%%

stack_data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        data = process_stack(stack_path)
        stack_data.append(data)
        io.imsave(
            Path(data_path, f"{stack_path.stem}_process.tif"),
            data["stack"].astype("float32"), check_contrast=False,
            )

#%%

ref_data = stack_data[0]
for i, data in enumerate(stack_data):
    
    if i > 0:
        
        refStack = stack_data[0]["stack"]
        refMask1 = stack_data[0]["mask1"]
        
        # Rescale stack
        rscFactor = np.sqrt(
            np.sum(refMask1) / np.sum(data["mask1"]))
        rscStack = rescale(data["stack"], rscFactor)
        
        # Adjust canva size
        dSize = (rscStack.shape[1] - refStack.shape[1]) // 2
        if dSize > 0:
            rscStack = rscStack[:, dSize : -dSize, dSize: -dSize] 
        elif dSize < 0:
            print("bug")
        
        # Get mask
        avgProj = np.mean(rscStack, axis=0)
        mask1 = avgProj >= data["thresh1"]
        mask2 = avgProj >= data["thresh2"]
        mask2 = binary_dilation(mask2, footprint=disk(3))
        mask = mask1 ^ mask2
        
        # Outputs
        stack_data[i]["stack"] = rscStack
        stack_data[i]["mask1"] = mask1
        stack_data[i]["mask2"] = mask2
        stack_data[i]["mask"] = mask
        
    # Extract zProfiles
    zProf1, zProf2 = [], []
    for img in stack_data[i]["stack"]:
        zProf1.append(np.mean(img[stack_data[i]["mask"]]))
        zProf2.append(np.mean(img[stack_data[i]["mask"]]))
    zProf1 = np.stack(zProf1) / np.mean(zProf1) - 1
    zProf2 = np.stack(zProf2) / np.mean(zProf2) - 1
    
    # Outputs
    stack_data[i]["zProf1"] = zProf1
    stack_data[i]["zProf2"] = zProf2
 
# io.imsave(
#     Path(data_path, f"{stack_path.stem}_rescaled{tp2}.tif"),
#     stack.astype("float32"), check_contrast=False,
#     )


#%%

data = stack_data[0]

from scipy.signal import correlate

tp1 = 0
tp2 = 3

sig1 = stack_data[tp1]["zProf1"]
sig2 = stack_data[tp2]["zProf1"]
min_length = np.minimum(sig1.shape[0], sig2.shape[0])
crop1 = sig1[0: min_length]
crop2 = sig2[0: min_length]
ccor = correlate(crop1, crop2, mode='full')
lag = np.argmax(ccor) - min_length
pad = np.full(np.abs(lag), np.nan)
if lag > 0:
    aCrop2 = np.concatenate((pad, crop2))
elif lag < 0:
    aCrop2 = crop2[lag:]


# Plotting
plt.figure(figsize=(6, 6))

plt.subplot(3, 1, 1)
plt.plot(crop1, label="crop1")
plt.plot(crop2, label="crop2")
plt.title("Original signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(ccor, label="ccor")
plt.title("Cross correlation")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(crop1, label="crop1")
plt.plot(aCrop2, label="aCrop2")
plt.title("Aligned signal")
plt.legend()


