#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from pystackreg import StackReg
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

#%% Comments ------------------------------------------------------------------

'''

- There is stack to stack variations regarding brightness and pixel size.

'''

#%% Parameters ----------------------------------------------------------------

stack_idx = 7
rsize_factor = 4
min_int = 5173
max_int = 40000
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
    
    def resize_img(img_path):
        return downscale_local_mean(io.imread(img_path), rsize_factor)
    
    def register_img(ref, img):
        tmp_ref = ref[
            int(ref.shape[0] * 0.25): int(ref.shape[0] * 0.75),
            int(ref.shape[1] * 0.25): int(ref.shape[1] * 0.75)   
            ]
        tmp_img = img[
            int(img.shape[0] * 0.25): int(img.shape[0] * 0.75),
            int(img.shape[1] * 0.25): int(img.shape[1] * 0.75)   
            ]
        sr.register(tmp_ref, tmp_img)
        return sr.transform(img)
    
    # Execute -----------------------------------------------------------------
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Initialize
    print(stack_path.stem)
    sr = StackReg(StackReg.TRANSLATION)
            
    # Resize stack
    print("  Resize :", end='')
    t0 = time.time()
    stack = Parallel(n_jobs=-1)(
            delayed(resize_img)(img_path) 
            for img_path in img_paths
            )
    stack = np.stack(stack)
    t1 = time.time()
    print(f" {(t1-t0):5.2f} s") 
    
    # Select slices
    z_mean = np.mean(stack, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.argmax(np.abs(z_mean_diff)) + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
    stack = stack[z0:z1, ...]    
    
    # # Register stack
    # print("  Register :", end='')
    # t0 = time.time()
    # stack = Parallel(n_jobs=-1)(
    #         delayed(register_img)(stack[0,...], stack[i,...]) 
    #         for i in range(len(stack))
    #         )
    # stack = np.stack(stack)
    # t1 = time.time()
    # print(f" {(t1-t0):5.2f} s") 

    return stack

#%%

stacks = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        stack = process_stack(stack_path)
        stacks.append(stack)
        io.imsave(
            Path(data_path, f"{stack_path.stem}_process.tif"),
            stack.astype("float32"), check_contrast=False,
            )
        
#%%

tp = 2

# Top/bottom tilt angle
top = stacks[tp][ 0,...] > 30000
bot = stacks[tp][-1,...] > 30000

idxTop = np.argwhere(top == 1)
idxBot = np.argwhere(bot == 1)
y0, x0 = np.mean(idxTop, axis=0)
y1, x1 = np.mean(idxBot, axis=0)

dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
angr = np.arctan2(y1 - y0, x1 - x0)
angd = np.degrees(angr)

hypo = np.sqrt(dist**2 + stacks[tp].shape[0]**2)

test = stacks[tp]

io.imsave(
    Path(data_path, f"{stack_path.stem}_top_{tp}.tif"),
    top.astype("uint8")*255, check_contrast=False,
    )
io.imsave(
    Path(data_path, f"{stack_path.stem}_bot_{tp}.tif"),
    bot.astype("uint8")*255, check_contrast=False,
    )
        
#%%

# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import find_peaks, peak_prominences

# data = []
# for stack in stacks:
    
#     # Pixel intensity distribution
#     avgProj = np.mean(stack, axis=0)
#     hist, bins = np.histogram(
#         avgProj.flatten(), bins=1024, range=(0, 65535))    
#     hist = gaussian_filter1d(hist, sigma=2)
#     pks, _ = find_peaks(hist, distance=20)
#     proms = peak_prominences(hist, pks)[0]
#     sorted_pks = pks[np.argsort(proms)[::-1]]
#     select_pks = sorted_pks[:3]
    
#     # Get masks
#     thresh1 = bins[select_pks[1]] - (
#         (bins[select_pks[1]] - bins[select_pks[0]]) / 2)
#     thresh2 = bins[select_pks[2]] - (
#         (bins[select_pks[2]] - bins[select_pks[1]]) / 2)
#     mask1 = (avgProj >= thresh1) & (avgProj <= thresh2)
#     mask2 = avgProj >= thresh2
    
#     # Extract zProfiles
#     zProf1, zProf2 = [], []
#     for img in stack:
#         zProf1.append(np.mean(img[mask1]))
#         zProf2.append(np.mean(img[mask2]))
#     zProf1 = (np.stack(zProf1) - np.mean(zProf1)) / np.std(zProf1)
#     zProf2 = (np.stack(zProf2) - np.mean(zProf2)) / np.std(zProf2)
    
#     data.append({
#         "avgProj": avgProj,
#         "zProf1" : zProf1,
#         "zProf2" : zProf2,
#         "thresh1": thresh1,
#         "thresh2": thresh2,
#         "mask1"  : mask1,
#         "mask2"  : mask2,
#         })
    
#%%
