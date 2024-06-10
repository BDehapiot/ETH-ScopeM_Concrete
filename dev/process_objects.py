#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
from skimage import io
from pathlib import Path

# Functions
from functions import objects

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
model_path = Path.cwd() / f"model-weights_void_p0256_d{df}.h5"
# experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
experiment = "H1_ICONX_DoS"
# experiment = "H9_ICONX_DoS"
stack_name = f"{experiment}_Time3_crop_df{df}"

#%% Initialize ----------------------------------------------------------------

# Open data
experiment_path = data_path / experiment
stack = io.imread(experiment_path / (stack_name + ".tif"))
stack_norm = io.imread(experiment_path / (stack_name + "_norm.tif"))
obj_probs = io.imread(experiment_path / (stack_name + "_probs.tif"))
air_mask_3D = io.imread(experiment_path / (stack_name + "_air_mask.tif"))
liquid_mask_3D = io.imread(experiment_path / (stack_name + "_liquid_mask.tif"))

# Open metadata
metadata_path = experiment_path / (stack_name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
mtx_mask = metadata["mtx_mask"]
rod_mask = metadata["rod_mask"]
centers = metadata["centers"]
mtx_EDM = metadata["mtx_EDM"]

#%% Objects -------------------------------------------------------------------

t0 = time.time()
print(" - Objects : ", end='')

obj_labels, obj_data = objects(
        obj_probs, 
        mtx_mask, rod_mask, 
        air_mask_3D, liquid_mask_3D, 
        mtx_EDM, centers,
        df) 

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_labels(obj_labels)