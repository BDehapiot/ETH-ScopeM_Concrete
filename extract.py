#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import get_indexes

#%% Inputs --------------------------------------------------------------------

# Path
local_path = Path('D:/local_Concrete/data/raw')
train_path = Path(Path.cwd(), 'data', 'train')
img_paths = list(local_path.glob("**/*.tif"))

# Selection
nImg = 100 # number of images

#%% Extract -------------------------------------------------------------------

idxs = get_indexes(nImg, len(img_paths))
for i, idx in enumerate(idxs):
    img = io.imread(img_paths[idx])
    if np.std(img) > 1:        
        io.imsave(
            Path(train_path, f"image_{i:03d}.tif"),
            img, check_contrast=False,
            )