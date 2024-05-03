#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from functions import import_stack, preprocess
from bdtools import extract_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Concrete/data/raw')
model_name = "model-weights_matrix_p0512_d2.h5"
# model_name = "model-weights_matrix_p0256_d4.h5"
experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"

# Prepare
downscale_factor = int(model_name[28])
size = int(model_name[22:26])
overlap = size // 4

# Predict
sub_size = 100

#%% Initialize ----------------------------------------------------------------

stack_paths = []
for path in local_path.iterdir():
    if path.is_dir() and experiment in path.name:
        stack_paths.append(path)

#%%

stack_idx = 0
stack_path = stack_paths[stack_idx]
stack = import_stack(stack_path, downscale_factor)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack)

#%% Predict -------------------------------------------------------------------

def predict(stack, model_name, sub_size):
    
    # Define model
    model = sm.Unet(
        'resnet18', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(Path(Path.cwd(), model_name))

    # Define sub indexes
    nZ = stack.shape[0]
    z0s = np.arange(0, nZ, sub_size)
    z1s = z0s + sub_size
    z1s[z1s > nZ] = nZ
    
    # Preprocess stack
    stack = preprocess(stack, 0.01, 99.99, 10 // downscale_factor)
    
    # Predict
    probs = []
    for z0, z1 in zip(z0s, z1s):
        tmpStack = stack[z0:z1, ...]
        patches = extract_patches(tmpStack, size, overlap)
        patches = np.stack(patches)
        tmpProbs = model.predict(patches).squeeze()
        tmpProbs = merge_patches(tmpProbs, tmpStack.shape, overlap)
        probs.append(tmpProbs)
    probs = np.concatenate(probs, axis=0)
        
    return probs

probs = predict(stack, model_name, sub_size)

# Display 
viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(probs)
