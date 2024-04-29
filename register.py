#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

from functions import normalize_stack, mask_stack 

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H9_ICONX_DoS",
    ]

# Parameters
overwrite = True
df = 4 # downscale factor for preprocessing

#%% Function(s) ---------------------------------------------------------------

def process_stack(path, experiment_path, df):
    
    global metadata, stack_norm # dev
    
    # Initialize --------------------------------------------------------------
        
    name = path.name.replace(f"_crop_df{df}.tif", "")
    
    # Read --------------------------------------------------------------------

    print(f"(process) {name}")
    t0 = time.time()
    print(" - Read :", end='')
    
    # Data
    stack = io.imread(path)
        
    # Metadata
    metadata_path = experiment_path / (name + "_metadata_oo.pkl") 
    with open(metadata_path, 'rb') as file:
        metadata = pickle.load(file)
    
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # -------------------------------------------------------------------------
    
    # Normalize stack
    df_idx = metadata["dfs"].index(df)
    stack_norm = normalize_stack(
        stack, 
        metadata["med_projs"][df_idx], 
        metadata["yx_shifts"][df_idx], 
        mask=metadata["mtx_masks"][df_idx],
        )
    
    
    
    # # Get object mask and labels
    # obj_mask_3D = (stack_norm < 0.8) & (stack_norm > 0) # parameter
    # obj_mask_3D = remove_small_objects(
    #     obj_mask_3D, min_size=2.5e5 * (1 / rsize_factor) ** 3) # parameter
    # obj_mask_3D = clear_border(obj_mask_3D)
    # obj_labels_3D = label(obj_mask_3D)


#%% Execute -------------------------------------------------------------------

t = 0   

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        for path in experiment_path.glob(f"*_crop_df{df}*"):
            if f"Time{t}" in path.name:            
                process_stack(path, experiment_path, df)
            
#%%

import napari
viewer = napari.Viewer()
viewer.add_image(stack_norm)
