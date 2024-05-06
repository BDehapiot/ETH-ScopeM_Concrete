#%% Imports -------------------------------------------------------------------

import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches
from functions import shift_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
model_name = "model-weights_matrix_p0256_d4.h5"
experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"
stack_name = f"{experiment}_Time2_crop_df4"

# Parameters
df = int(model_name[28])
size = int(model_name[22:26])
overlap = size // 4

# Predict
sub_size = 100

#%% Initialize ----------------------------------------------------------------

# Open data
experiment_path = data_path / experiment
stack = io.imread(experiment_path / (stack_name + ".tif"))

# Open metadata
metadata_path = experiment_path / (stack_name + "_metadata.pkl") 
with open(metadata_path, 'rb') as file:
    metadata = pickle.load(file)
mtx_mask = metadata["mtx_mask"]
rod_mask = metadata["rod_mask"]
mtx_EDM  = metadata["mtx_EDM"]
rod_EDM  = metadata["rod_EDM"]
centers  = metadata["centers"]

# Shift matrix mask
mtx_mask_3D = shift_stack(mtx_mask, centers, reverse=True)

# Normalize stack
stack[mtx_mask_3D == 0] = 0
stack = norm_gcn(stack, mask=stack != 0)
stack = norm_pct(stack, pct_low=0.01, pct_high=99.99, mask=stack != 0)

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(stack)
# viewer.add_image(mtx_mask_3D)

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
    
    # # Preprocess stack
    # stack = norm_gcn(stack, mask=mask)
    # stack = norm_pct(stack, pct_low=0.01, pct_high=99.99, mask=mask)
    
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
