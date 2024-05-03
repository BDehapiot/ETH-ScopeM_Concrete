#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

#%% Parameters ----------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"

#%% Initialize ----------------------------------------------------------------

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)

#%% Open ----------------------------------------------------------------------

stack_idx = 0

t0 = time.time()
print("Open") 

# Open stack
def open_stack(stack_idx, stack_paths):
    stack = []
    for path in stack_paths[stack_idx].iterdir():
        if path.suffix == ".tif":
            stack.append(io.imread(path))
    return stack

stack = open_stack(stack_idx, stack_paths)
stack = np.stack(stack)

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  

#%% Process ------------------------------------------------------------------- 

from skimage.transform import downscale_local_mean

# -----------------------------------------------------------------------------

t0 = time.time()
print("Select slices") 

z_mean = np.mean(stack, axis=(1,2)) 
idx = z_mean > np.max(z_mean) * 0.9
z0 = np.argmax(idx) 
z1 = len(idx) - np.argmax(idx[::-1]) 
stack = stack[z0:z1, ...]

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  

# -----------------------------------------------------------------------------

t0 = time.time()
print("Resize") 

stack = downscale_local_mean(stack, (1, 2, 2)).astype("uint16")

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  

# -----------------------------------------------------------------------------
io.imsave(
    Path(data_path, f"{stack_paths[stack_idx].stem}_resize.tif"),
    stack, check_contrast=False,
    )


    