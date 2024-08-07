#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

# Functions 
from functions import filt_median

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
# model_name = "model-weights_matrix_p0256_d4.h5"
model_name = "model-weights_liquid_p0256_d4.h5"
model_path = Path.cwd() / model_name
stack_name = "D1_ICONX_DoS_Time1_crop_df4_norm.tif"
stack_path = list(data_path.glob(f"**/*{stack_name}"))[0]

# Parameters
subset = 1000

#%% Functions -----------------------------------------------------------------

def predict(stack, model_path, subset=1000):
    
    # Define model
    model = sm.Unet(
        'resnet34', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(model_path)
    size = int(model_path.name[20:24]) # void
    # size = int(model_path.name[22:26]) # matrix
    overlap = size // 4

    # Define sub indexes
    nZ = stack.shape[0]
    z0s = np.arange(0, nZ, subset)
    z1s = z0s + subset
    z1s[z1s > nZ] = nZ
    
    # Prepare images
    stack = norm_gcn(stack, mask=stack != 0)
    stack = norm_pct(stack, 0.01, 99.99, mask=stack != 0)
    
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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Open 
    stack = io.imread(stack_path)
    
    # Predict
    probs = predict(stack, model_path, subset)
    
    # Prepare images
    stack = norm_gcn(stack, mask=stack != 0)
    stack = norm_pct(stack, 0.01, 99.99, mask=stack != 0)

    # Display 
    viewer = napari.Viewer()
    viewer.add_image(stack)
    viewer.add_image(probs, colormap="bop orange")