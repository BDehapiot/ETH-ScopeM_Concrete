#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
train_path = Path(Path.cwd(), 'data', 'train')
experiments = [
    "D1_ICONX_DoS",
    "D11_ICONX_DoS",
    "D12_ICONX_corrosion", 
    "H9_ICONX_DoS",
    ]

# Parameters
df = 4
nImg = 4 # number of extracted images per stack

#%% Function(s) ---------------------------------------------------------------

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs   

def extract_images(path, nImg):
    
    with tifffile.TiffFile(path) as tif:
    
        # Get image indexes
        nZ = len(tif.pages)
        idxs = get_indexes(nImg, nZ)
        
        # Extract & save images
        for idx in idxs:
            img = tif.pages[idx].asarray()
            io.imsave(
                train_path / f"{path.stem}_{idx:03d}.tif",
                img, check_contrast=False,
                )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        for path in experiment_path.glob(f"*_crop_df{df}_norm.tif*"):
            extract_images(path, nImg)