#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import disk, binary_dilation

# Scipy
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import shift, gaussian_filter1d, binary_fill_holes

#%% Inputs --------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    # "D1_ICONX_DoS"
    "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

# Parameters
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Functions -----------------------------------------------------------------

def roll_image(img_rsize):
    idx = np.argwhere((img_rsize > 30000) == 1)
    y0, x0 = img_rsize.shape[0] // 2, img_rsize.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img_rsize, yx_shift, mode='wrap'), yx_shift 

# -----------------------------------------------------------------------------

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

#%% Process -------------------------------------------------------------------

# Open data
stack_reg = io.imread(Path(data_path, f"{exp_name}_stack_reg.tif"))

# Select slices
z_mean = np.mean(np.mean(stack_reg, axis=0), axis=(1,2))
z_mean_diff = np.gradient(z_mean)
idx = np.where(
    (np.abs(z_mean_diff) < 5) &
    (z_mean > np.max(z_mean) * 0.9)
    )[0]
stack_reg = stack_reg[:, idx[0]:idx[-1], ...]

# Roll stack
print("  Roll       :", end='')
t0 = time.time()
stack_roll = []
for stack in stack_reg:
    outputs = Parallel(n_jobs=-1)(
            delayed(roll_image)(img) 
            for img in stack
            )
    stack_roll.append(np.stack([data[0] for data in outputs]))
stack_roll = np.stack(stack_roll)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# -----------------------------------------------------------------------------

avg_proj, mtx_thresh, rod_thresh, mtx_mask, rod_mask = get_2Dmasks(
    np.mean(stack_roll, axis=0)
    )

# -----------------------------------------------------------------------------

io.imsave(
    Path(data_path, f"{exp_name}_avg_proj.tif"),
    avg_proj.astype("float32"),
    check_contrast=False,
    # imagej=True,
    # metadata={'axes': 'TYX'},
    # photometric='minisblack',
    # planarconfig='contig',
    )

# stack_norm = []
# for stack in stack_roll:
#     avg_proj = np.mean(stack, axis=0)
#     norm = []
#     for img in stack:
#         img_norm = np.divide(img, avg_proj, where=avg_proj!=0)
#         norm.append(img_norm)
#     norm = np.stack(norm)
#     stack_norm.append(norm)
# stack_norm = np.stack(stack_norm)

# stack_norm = np.mean(stack_norm, axis=0)

# -----------------------------------------------------------------------------

# # Save 
# io.imsave(
#     Path(data_path, f"{exp_name}_stack_reg_mod.tif"),
#     stack_norm.astype("float32"),
#     check_contrast=False,
#     # imagej=True,
#     # metadata={'axes': 'TZYX'},
#     # photometric='minisblack',
#     # planarconfig='contig',
#     )


