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
data_path = Path('D:/local_Concrete/data')
data_raw_path = Path(data_path, "0-raw")
data_exp_path = Path(data_path, experiment)
data_exp_path.mkdir(parents=True, exist_ok=True)

#%% Initialize ----------------------------------------------------------------

stack_paths = []
for path in data_raw_path.iterdir():
    if path.is_dir() and experiment in path.name:
        stack_paths.append(path)
        
#%% Preprocess ----------------------------------------------------------------

idx = 0

# -----------------------------------------------------------------------------

stack = import_stack(stack_paths[idx], data_exp_path)

# test_path = Path(processed_path, metadata["name"] + ".tif")
# print(test_path.resolve())