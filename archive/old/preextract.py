#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
from functions import preprocess_stack

#%% Inputs --------------------------------------------------------------------

# Path
local_path  = Path('D:/local_Concrete/data')

# Parameters
downscale_factor = 2

#%% Preextract ----------------------------------------------------------------

stack_paths = []
for stack_path in local_path.iterdir():
    if stack_path.is_dir():
        
        stack, rslice = preprocess_stack(stack_path, downscale_factor)
        
        io.imsave(
            Path(local_path, f"{stack_path.name}_d{downscale_factor}.tif"),
            stack, check_contrast=False
            )
        io.imsave(
            Path(local_path, f"{stack_path.name}_rslice_d{downscale_factor}.tif"),
            rslice, check_contrast=False
            ) 