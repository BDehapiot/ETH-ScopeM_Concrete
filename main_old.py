#%% Imports -------------------------------------------------------------------

import pickle
from skimage import io
from pathlib import Path
from functions_old import process_stacks, register_stacks

#%% Parameters ----------------------------------------------------------------

rsize_factor = 4 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Paths ---------------------------------------------------------------------

raw_path = "D:/local_Concrete/data/raw"
data_path = "D:/local_Concrete/data"
exp_name = (
    # "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    "H9_ICONX_DoS"
    )

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)
                
#%% Execute -------------------------------------------------------------------

# Process stacks
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

# Register stacks     
hstack_reg = register_stacks(stack_data)

# Save 
io.imsave(
    Path(data_path, f"{exp_name}_hstack_reg.tif"),
    hstack_reg.astype("float32"),
    check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )

with open(Path(data_path, f"{exp_name}_stack_data.pkl"), 'wb') as f:
    pickle.dump(stack_data, f)