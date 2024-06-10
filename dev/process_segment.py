#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
from skimage import io
from pathlib import Path

# Functions
from functions import segment

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

# Open metadata
metadata_path = experiment_path / (stack_name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
centers = metadata["centers"]
mtx_EDM = metadata["mtx_EDM"]

#%% Segment -------------------------------------------------------------------

t0 = time.time()
print(" - Segment : ", end='')

stack_norm_corr, air_mask_3D, liquid_mask_3D = segment(
    stack_norm, obj_probs, mtx_EDM, centers, df)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s") 

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(stack, opacity=0.75)
viewer.add_image(stack_norm_corr, opacity=0.75)
viewer.add_image(
    air_mask_3D, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip",
    # attenuation=0.5,
    colormap="bop orange"
    )
viewer.add_image(
    liquid_mask_3D, 
    opacity=0.5,
    blending="additive", 
    rendering="attenuated_mip", 
    # attenuation=0.5,
    colormap="bop blue"
    )