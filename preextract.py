#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import import_stack

#%% Inputs --------------------------------------------------------------------

# Path
experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"
local_path = Path('D:/local_Concrete/data')
processed_path = Path(local_path, experiment + "_processed")
processed_path.mkdir(parents=True, exist_ok=True)

# Parameters
downscale_factor = 4

#%% Initialize ----------------------------------------------------------------

stack_paths = []
for path in local_path.iterdir():
    if path.is_dir() :
        if experiment in path.name:
            if "processed" not in path.name:
                stack_paths.append(path)
        
#%% Preprocess ----------------------------------------------------------------

idx = 0
stack, metadata = import_stack(stack_paths[idx])
io.imsave(
    Path(processed_path, metadata["name"] + ".tif"),
    stack, check_contrast=False,
    )

test_path = Path(processed_path, metadata["name"] + ".tif")
print(test_path.resolve())