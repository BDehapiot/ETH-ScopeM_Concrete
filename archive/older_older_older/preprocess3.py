#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from pystackreg import StackReg
from scipy.signal import find_peaks
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
    
    def open_img(img_path):
        img = downscale_local_mean(io.imread(img_path), rsize_factor)
        # img = np.clip(img, min_int, max_int)
        # img = ((img - min_int) / (max_int - min_int) * 255).astype("uint8")
        return img
    
    def register_img(reference, img):
        return sr.register_transform(reference, img)
    
    # Execute -----------------------------------------------------------------
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Open stack
    print(stack_path.stem)
    print("  Open     :", end='')
    t0 = time.time()
    stack = Parallel(n_jobs=-1)(
            delayed(open_img)(img_path) 
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
    sr = StackReg(StackReg.TRANSLATION)
    stack = Parallel(n_jobs=-1)(
            delayed(register_img)(stack[0,...], stack[i,...]) 
            for i in range(stack.shape[0])
            )
    stack = np.stack(stack)
    # stack[stack < 0] = 0
    # stack[stack > 255] = 255
    # stack = stack.astype("uint8")
    t1 = time.time()
    print(f" {(t1-t0):5.2f} s") 

    return stack

#%%

for stack_path in stack_paths:
    if stack_name in stack_path.name:      
        stack = process_stack(stack_path)
        io.imsave(
            Path(data_path, f"{stack_path.stem}_process.tif"),
            stack.astype("float32"), check_contrast=False,
            )
    
