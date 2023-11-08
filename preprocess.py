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

avg_proj = []
for stack in stacks:
    avg_proj.append(np.mean(stack, axis=0).astype("uint16"))
    
#%%

plt.hist(avg, kwargs)
