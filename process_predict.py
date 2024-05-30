#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
from skimage import io
from pathlib import Path

# Functions
from functions import predict

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
stack_norm = io.imread(experiment_path / (stack_name + "_norm.tif"))

# Open metadata
metadata_path = experiment_path / (stack_name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)

#%% Predict -------------------------------------------------------------------

t0 = time.time()

obj_probs = predict(stack_norm, model_path, subset=1000)

t1 = time.time()
print(f" - Predict : {(t1-t0):<5.2f}s") 

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(stack_norm)
viewer.add_image(obj_probs)