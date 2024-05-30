#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
from skimage import io
from pathlib import Path

# Functions
from functions import preprocess

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
model_path = Path.cwd() / f"model-weights_void_p0256_d{df}.h5"
experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
# experiment = "H9_ICONX_DoS"
stack_name = f"{experiment}_Time2_crop_df{df}"

#%% Initialize ----------------------------------------------------------------

# Open data
experiment_path = data_path / experiment
stack = io.imread(experiment_path / (stack_name + ".tif"))

# Open metadata
metadata_path = experiment_path / (stack_name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)

#%% Preprocess ----------------------------------------------------------------

t0 = time.time()
print(" - Preprocess : ", end='')

centers, med_proj, mtx_mask, rod_mask, mtx_EDM, rod_EDM, stack_norm \
    = preprocess(stack)

t1 = time.time()
print(f" - Predict : {(t1-t0):<5.2f}s") 

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(med_proj)
viewer.add_image(mtx_mask)
viewer.add_image(rod_mask)
viewer.add_image(mtx_EDM)
viewer.add_image(rod_EDM)
viewer.add_image(stack_norm)
