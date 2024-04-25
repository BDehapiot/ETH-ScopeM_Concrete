#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.transform import downscale_local_mean

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
raw_path = Path(data_path, "0-raw")
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H9_ICONX_DoS",
    ]

# Parameters
dfs = [1, 2, 4] # downscale factor range

#%% Function(s) ---------------------------------------------------------------

def format_stack(path, save_path, dfs):
    
    # Initialize --------------------------------------------------------------

    name = path.name    

    # Read --------------------------------------------------------------------
    
    t0 = time.time()
    print(name)
    print(" - Read : ", end='')
    
    stack = []
    img_paths = list(path.glob("**/*.tif"))
    for img_path in img_paths:
        stack.append(io.imread(img_path))
    stack = np.stack(stack)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Format ------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Format : ", end='')

    def nearest_divisibles(value, levels):
        divisor = 2 ** levels
        lowDiv = value - (value % divisor)
        if lowDiv == value:
            highDiv = value + divisor
        else:
            highDiv = lowDiv + divisor
        return lowDiv, highDiv

    # Select slices
    z_mean = np.mean(stack, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
     
    # Crop (zyx)   
    z0 = nearest_divisibles(z0, len(dfs))[1] 
    z1 = nearest_divisibles(z1, len(dfs))[0] 
    nYdiv = nearest_divisibles(stack.shape[1], len(dfs))[0]
    nXdiv = nearest_divisibles(stack.shape[2], len(dfs))[0]
    y0 = (stack.shape[1] - nYdiv) // 2 
    y1 = y0 + nYdiv
    x0 = (stack.shape[2] - nXdiv) // 2 
    x1 = x0 + nXdiv
    stack = stack[z0:z1, y0:y1, x0:x1] 

    # Downscale (zyx) 
    stacks = [stack]
    for i, df in enumerate(dfs):
        if df > 1:
            stacks.append(downscale_local_mean(stacks[i - 1], 2))
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Save --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Paths
    save_names = [name + f"_crop_df{df}.tif" for df in dfs]
    save_paths = [save_path / save_name for save_name in save_names]
    
    # Data
    for i, df in enumerate(dfs): 
        io.imsave(save_paths[i], stacks[i], check_contrast=False)
        
    # Metadata
    metadata = {
        "dfs"    : dfs,
        "names"  : save_names,
        "paths"  : save_paths,
        "shapes" : [stack.shape for stack in stacks],
        "info"   : {
            "z0" : z0, "z1" : z1,
            "y0" : y0, "y1" : y1,
            "x0" : x0, "x1" : x1,
            }
        }
    
    with open(save_path / name + "_metadata.pkl", 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        save_path = Path(data_path, experiment)
        save_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            format_stack(path, save_path, dfs)
            # if not path.with_name(path.name + "_crop_df1.tif").is_file():
                