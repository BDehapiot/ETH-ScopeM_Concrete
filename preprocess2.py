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

#%% Parameters ----------------------------------------------------------------

stack_idx = 13
rsize_factor = 4
max_diff = 0.33
min_int = 5173
max_int = 40000
data_path = "D:/local_Concrete/data/DIA"

#%% Initialize ----------------------------------------------------------------

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)

#%% Open & resize -------------------------------------------------------------

t0 = time.time()
print("Open & resize") 

# Open & resize stack
def open_stack(stack_idx, stack_paths):
    
    # def open_img(img_path):
    #     return downscale_local_mean(
    #         io.imread(img_path), rsize_factor).astype("uint16")
    
    def open_img(img_path):
        img = downscale_local_mean(io.imread(img_path), rsize_factor)
        img = np.clip(img, min_int, max_int)
        img = ((img - min_int) / (max_int - min_int) * 255).astype("uint8")
        return img
    
    img_paths = []
    for path in stack_paths[stack_idx].iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    stack = Parallel(n_jobs=-1)(
            delayed(open_img)(img_path) 
            for img_path in img_paths
            )

    return stack

stack = open_stack(stack_idx, stack_paths)
stack = np.stack(stack)

t1 = time.time()
print(f"  {(t1-t0):5.6f} s") 

#%% Select slices ------------------------------------------------------------- 

t0 = time.time()
print("Select slices") 

z_mean = np.mean(stack, axis=(1,2)) 
z_mean_diff = np.gradient(z_mean)
z0 = np.argmax(np.abs(z_mean_diff)) + 1
z1 = np.where(
    (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
stack = stack[z0:z1, ...]

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  

#%%



t0 = time.time()
print("Register") 

def register_stack(stack):
    
    sr = StackReg(StackReg.TRANSLATION)
    
    def register_img(reference, img):
        return sr.register_transform(reference, img)

    stack_reg = Parallel(n_jobs=-1)(
            delayed(register_img)(stack[0,...], stack[i,...]) 
            for i in range(stack.shape[0])
            )
    
    return stack_reg

stack_reg = register_stack(stack)
stack_reg = np.stack(stack_reg)

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  


#%% Save ---------------------------------------------------------------------- 

t0 = time.time()
print("Save") 

io.imsave(
    Path(data_path, f"{stack_paths[stack_idx].stem}_resize.tif"),
    stack, check_contrast=False,
    )

io.imsave(
    Path(data_path, f"{stack_paths[stack_idx].stem}_resize_reg.tif"),
    stack_reg.astype("float32"), check_contrast=False,
    )

t1 = time.time()
print(f"  {(t1-t0):5.6f} s")  


    