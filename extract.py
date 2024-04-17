#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import normalize_gcn, normalize_pct

#%% Inputs --------------------------------------------------------------------

# Path
local_path = Path('D:/local_Concrete/data')
train_path = Path(Path.cwd(), 'data', 'train')
img_paths = list(local_path.glob("**/*.tif"))

# Selection
nImg = 100

#%% Extract -------------------------------------------------------------------

# Create random indexes
idxs = np.random.choice(
    range(0, len(img_paths)), size=nImg, replace=False)

# Extract & normalize images
for i, idx in enumerate(idxs):   
    img = io.imread(img_paths[idx])
    if np.std(img) > 0:
        img = normalize_gcn(img)
        img = normalize_pct(img, 0.01, 99.99)
        io.imsave(
            Path(train_path, f"img_{i:03d}.tif"),
            img.astype("float32"), check_contrast=False,
            )