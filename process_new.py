#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from scipy.ndimage import shift
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

#%% Parameters ----------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
stack_name = "D11_ICONX_DoS"

rsize_factor = 4 # Image size reduction factor
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

def shift_image(img, yxShift):
    return shift(img, yxShift, mode='wrap')

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
    
    # Shift stack
    print("  Shift   :", end='')
    t0 = time.time()
    mean_proj = np.mean(stack_rsize, axis=0)
    idx = np.argwhere((mean_proj > 30000) == 1)
    y0, x0 = mean_proj.shape[0] // 2, mean_proj.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yxShift = [y0 - y1, x0 - x1]
    stack_shift = Parallel(n_jobs=-1)(
            delayed(shift_image)(img, yxShift) 
            for img in stack_rsize
            )
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Print variables
    print( "  ---------")
    print(f"  zSlices : {z0}-{z1}")
    
    # Outputs
    stack_data.append({
        "stack_path"   : stack_path,
        "stack_rsize"  : stack_rsize,
        "stack_shift"  : stack_shift,
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
        Path(data_path, f"{data['stack_path'].stem}_shift.tif"),
        data["stack_shift"].astype("float32"), check_contrast=False,
        )
    
#%%
