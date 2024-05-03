#%% Imports -------------------------------------------------------------------

import tifffile
from skimage import io
from pathlib import Path

from functions import get_indexes, median_filt

#%% Inputs --------------------------------------------------------------------

# Path
local_path = Path('D:/local_Concrete/data')
train_path = Path(Path.cwd(), 'data', 'train')
stack_paths = list(local_path.glob("*.tif*"))

# Selection
nImg = 4 # number of image per stack

#%% Extract -------------------------------------------------------------------

for stack_path in stack_paths:
    
    with tifffile.TiffFile(stack_path) as tif:
        
        # Get stack shape
        nY, nX = tif.pages[0].shape
        nZ = len(tif.pages)
        
        # Get indexes
        idxs = get_indexes(nImg, nZ)
        
        for idx in idxs:
            img = tif.pages[idx].asarray()
            img = median_filt(img, radius=5)
            io.imsave(
                Path(train_path, f"{stack_path.name}".replace(".tif", f"_{idx:03d}.tif")),
                img, check_contrast=False,
                )