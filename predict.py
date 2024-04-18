#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from skimage.transform import rescale 
from bdtools import extract_patches, merge_patches
from functions import median_filt

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Concrete/data')
model_name = "model-weights_0512.h5"
stack_paths = list(local_path.glob("*.tif*"))
stack_idx = 0

# Patches
size = int(model_name[14:18])
overlap = size // 8 # overlap between patches

# Parameters
subSize = 100

#%% Preprocess ----------------------------------------------------------------

# Open & preprocess image
stack = io.imread(stack_paths[stack_idx])
rslice = np.swapaxes(stack, 0, 1)
stack = median_filt(stack, radius=5).astype("float32")
rslice = median_filt(rslice, radius=5).astype("float32")

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(rslice) 

#%% Predict -------------------------------------------------------------------

def predict(stack, model_name, subSize):
    
    # Define model
    model = sm.Unet(
        'resnet18', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Load weights
    model_path = Path(Path.cwd(), model_name.replace("weights_", "weights_matrix_"))
    model.load_weights(model_path)

    # Define sub indexes
    nZ = stack.shape[0]
    z0s = np.arange(0, nZ, subSize)
    z1s = z0s + subSize
    z1s[z1s > nZ] = nZ
    
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

probs_stack = predict(stack, model_name, subSize)
probs_rslice = predict(rslice, model_name, subSize)
probs_rslice = np.swapaxes(probs_rslice, 0, 1)

# Display 
viewer = napari.Viewer()
viewer.add_image(stack, contrast_limits=(0, 255))
viewer.add_image(probs_stack, contrast_limits=(0, 1), opacity=0.33, colormap="bop orange")
viewer.add_image(probs_rslice, contrast_limits=(0, 1), opacity=0.33, colormap="bop orange")

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(rslice, contrast_limits=(0, 255))
# viewer.add_image(probs_rslice, contrast_limits=(0, 1), opacity=0.33, colormap="bop orange")

#%% Predict -------------------------------------------------------------------

# # Define model
# model = sm.Unet(
#     'resnet18', # ResNet 18, 34, 50, 101 or 152
#     input_shape=(None, None, 1), 
#     classes=1, 
#     activation='sigmoid', 
#     encoder_weights=None,
#     )

# # Load weights
# model_path = Path(Path.cwd(), model_name.replace("weights_", "weights_matrix_"))
# model.load_weights(model_path)

# # Predict
# probs = []
# for patch in patches:
#     probs.append(model.predict(patch[np.newaxis, :]))

# probs = model.predict(patches).squeeze()
# probs = merge_patches(probs, stack.shape, overlap)


#%% Display -------------------------------------------------------------------