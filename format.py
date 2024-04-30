#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

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
overwrite = False
dfs = [1, 2, 4] # downscale factor range

#%% Function(s) ---------------------------------------------------------------

def format_stack(path, experiment_path, dfs):
    
    global metadata
    
    # Initialize --------------------------------------------------------------

    name = path.name    

    # Read --------------------------------------------------------------------
    
    print(f"(format) {name}")
    t0 = time.time()
    print(" - Read : ", end='')
    
    stack = []
    img_paths = list(path.glob("**/*.tif"))
    for img_path in img_paths:
        stack.append(io.imread(img_path))
    stack = np.stack(stack)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Crop --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Crop : ", end='')

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
    zCrop0 = np.nonzero(z_mean_diff)[0][0] + 1
    zCrop1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
     
    # Crop (zyx)   
    zCrop0 = nearest_divisibles(zCrop0, len(dfs))[1] 
    zCrop1 = nearest_divisibles(zCrop1, len(dfs))[0] 
    nYdiv = nearest_divisibles(stack.shape[1], len(dfs))[0]
    nXdiv = nearest_divisibles(stack.shape[2], len(dfs))[0]
    yCrop0 = (stack.shape[1] - nYdiv) // 2 
    yCrop1 = yCrop0 + nYdiv
    xCrop0 = (stack.shape[2] - nXdiv) // 2 
    xCrop1 = xCrop0 + nXdiv
    stack = stack[zCrop0:zCrop1, yCrop0:yCrop1, xCrop0:xCrop1] 

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

    # Downscale ---------------------------------------------------------------

    t0 = time.time()
    print(" - Downscale : ", end='')

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
    save_paths = [experiment_path / save_name for save_name in save_names]
    
    # Data
    for i, df in enumerate(dfs): 
        io.imsave(save_paths[i], stacks[i], check_contrast=False)
        
    # Metadata
    metadata_path = experiment_path / (name + "_metadata_o.pkl") 
    metadata = {
        "dfs"      : dfs,
        "names"    : save_names,
        "paths"    : save_paths,
        "shapes"   : [stack.shape for stack in stacks],
        "crop"     : (zCrop0, zCrop1, yCrop0, yCrop1, xCrop0, xCrop1),
        }
    
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

#%% Execute -------------------------------------------------------------------

t = 0

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            
            if f"Time{t}" in path.name:
            
                test_path = experiment_path / (path.name + "_crop_df1.tif")
                if not test_path.is_file():
                    format_stack(path, experiment_path, dfs)   
                elif overwrite:
                    format_stack(path, experiment_path, dfs)  