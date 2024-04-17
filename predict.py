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
from functions import normalize_gcn, normalize_pct

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path('D:/local_Concrete/data')
model_name = "model-weights_0512.h5"
img_paths = list(local_path.glob("**/*.tif"))
img_idx = 20000

# Patches
size = int(model_name[14:18])
overlap = size // 4 # overlap between patches

#%% Preprocess ----------------------------------------------------------------

# Open & preprocess image
img = io.imread(img_paths[img_idx]).astype("float32")
img = normalize_gcn(img)
img = normalize_pct(img, 0.01, 99.99)

# Extract patches
patches = extract_patches(img, size, overlap)
patches = np.stack(patches)

# Display 
viewer = napari.Viewer()
# viewer.add_image(img, contrast_limits=(0.1, 1))
viewer.add_image(patches, contrast_limits=(0.1, 1))

#%% Predict -------------------------------------------------------------------

# Define model
model = sm.Unet(
    'resnet18', # ResNet 18, 34, 50, 101 or 152
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )

# Load weights & predict (matrix)
model_path = Path(Path.cwd(), model_name.replace("weights_", "weights_matrix_"))
model.load_weights(model_path)
probs = model.predict(patches).squeeze()
probs = merge_patches(probs, img.shape, overlap)

# Display 
viewer = napari.Viewer()
viewer.add_image(img, contrast_limits=(0, 1))
viewer.add_image(probs, contrast_limits=(0, 1), opacity=0.33, colormap="bop orange")
    
#%% Display -------------------------------------------------------------------