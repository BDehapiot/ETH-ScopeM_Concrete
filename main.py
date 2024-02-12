#%% Imports -------------------------------------------------------------------

from pathlib import Path
from functions import process_stacks, register_stacks

#%% Parameters ----------------------------------------------------------------

rsize_factor = 8 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Paths ---------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    # "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)
                
#%% Execute -------------------------------------------------------------------

stack_data = []
for stack_path in stack_paths:
    if exp_name in stack_path.name: 
        process_stacks(
            stack_path, 
            stack_data,
            rsize_factor,
            mtx_thresh_coeff,
            rod_thresh_coeff,
            )
        
#%% Experiment ----------------------------------------------------------------

import time
from skimage import io

stack_names = "\n - ".join([data["stack_path"].name for data in stack_data])

print("Register stacks :", end='')
t0 = time.time()
stack_reg, transform_matrices = register_stacks(stack_data)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# io.imsave(
#     Path(data_path, "stack_reg.tif"),
#     stack_reg.astype("float32"),
#     check_contrast=False,
#     imagej=True,
#     metadata={'axes': 'TZYX'},
#     photometric='minisblack',
#     planarconfig='contig',
#     )

#%% Experiment ----------------------------------------------------------------

stack = io.imread(stack_data[1]["stack_path"])
