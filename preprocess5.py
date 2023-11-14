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
    
    # Register stack
    print("  Register :", end='')
    t0 = time.time()
    stack = Parallel(n_jobs=-1)(
            delayed(register_img)(stack[0,...], stack[i,...]) 
            for i in range(len(stack))
            )
    stack = np.stack(stack)
    t1 = time.time()
    print(f" {(t1-t0):5.2f} s") 

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

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences

data = []
for stack in stacks:
    
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
    mask1 = (avgProj >= thresh1) & (avgProj <= thresh2)
    mask2 = avgProj >= thresh2
    
    # Extract zProfiles
    zProf1, zProf2 = [], []
    for img in stack:
        zProf1.append(np.mean(img[mask1]))
        zProf2.append(np.mean(img[mask2]))
    zProf1 = (np.stack(zProf1) - np.mean(zProf1)) / np.std(zProf1)
    zProf2 = (np.stack(zProf2) - np.mean(zProf2)) / np.std(zProf2)
    
    data.append({
        "avgProj": avgProj,
        "zProf1" : zProf1,
        "zProf2" : zProf2,
        "thresh1": thresh1,
        "thresh2": thresh2,
        "mask1"  : mask1,
        "mask2"  : mask2,
        })
    
#%%

idx = 2

mask = data[idx]["mask1"]
avgProj = data[idx]["avgProj"]
test = (stacks[idx] / avgProj) * mask[None, :, :]

# test = (stacks[idx] >= data[idx]["thresh1"] * 1.25) & (stacks[idx] <= data[idx]["thresh2"])

io.imsave(
    Path(data_path, f"{stack_path.stem}_mask_{idx}.tif"),
    test.astype("float32"), check_contrast=False,
    )

    
#%%

# from scipy.signal import correlate

# signal1 = data[0]["zProf1"]
# signal2 = data[1]["zProf1"]

# def sliding_correlation(signal1, signal2, window_size):
#     delays, cross_corrs = [], []
#     for i in range(len(signal1) - window_size):
#         window1 = signal1[i:i+window_size]
#         window2 = signal2[i:i+window_size]
        
#         # Compute cross-correlation
#         cross_corr = correlate(window1, window2, mode='full')
#         lag = np.argmax(cross_corr) - (window_size - 1)
        
#         if cross_corr.shape[0] == (window_size * 2) - 1:
#             delays.append(lag)
#             cross_corrs.append(cross_corr)
        
#     return delays, cross_corrs

# delays, cross_corrs = sliding_correlation(signal1, signal2, 1500)

# cross_corrs = np.stack(cross_corrs)

# # plt.plot(delays)

# # Create the heatmap
# plt.imshow(cross_corrs, aspect='auto', cmap='viridis')

# # Add color bar
# plt.colorbar()

# # Add labels as needed
# plt.xlabel('Index')
# plt.ylabel('Array Number')
# plt.title('Heatmap of 1D Arrays')

# # Show the plot
# plt.show()

# signal1_crop = signal1[signal1.shape[0] // 2 - 250 : signal1.shape[0] // 2 + 250]
# signal2_crop = signal2[signal2.shape[0] // 2 - 250 : signal2.shape[0] // 2 + 250]

# cross_corr = correlate(signal1_crop, signal2_crop, mode='full')

# # Plotting
# plt.figure(figsize=(6, 6))

# # Original signals
# plt.subplot(2, 1, 1)
# plt.plot(signal1_crop, label='Signal 1')
# plt.plot(signal2_crop, label='Signal 2')
# plt.title("Original Signals")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.legend()

# # Aligned signals
# plt.subplot(2, 1, 2)
# plt.plot(cross_corr, label='cross_corr')
# # plt.plot(aligned_signal2, label='Aligned Signal 2')
# plt.title("Aligned Signals using DTW")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.legend()

# plt.tight_layout()
# plt.show()


#%%

# io.imsave(
#     Path(data_path, f"{stack_path.stem}_avgProjs_reg.tif"),
#     avgProjs_reg, check_contrast=False,
#     )

