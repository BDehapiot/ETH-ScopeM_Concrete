#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
model_name = "model-weights_matrix_p0256_d4.h5"
stack_name = "D1_ICONX_DoS_Time3_crop_df4_norm.tif"
stack_path = list(data_path.glob(f"**/*{stack_name}"))[0]

# Parameters
df = int(model_name[28])
size = int(model_name[22:26])
overlap = size // 4
sub_size = 100

#%% Functions -----------------------------------------------------------------

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
    
    # Normalize stack
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
    probs = predict(stack, model_name, sub_size)

    # Display 
    viewer = napari.Viewer()
    viewer.add_image(stack)
    viewer.add_image(probs, colormap="bop orange")