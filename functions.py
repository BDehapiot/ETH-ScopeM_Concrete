#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import disk
from skimage.filters.rank import median

#%% Function(s) ---------------------------------------------------------------

def nearest_divisibles(value, levels):
    divisor = 2 ** levels
    lowDiv = value - (value % divisor)
    if lowDiv == value:
        highDiv = value + divisor
    else:
        highDiv = lowDiv + divisor
    return lowDiv, highDiv

def get_indexes(nIdx, maxIdx):
    if maxIdx <= nIdx:
        idxs = np.arange(0, maxIdx)
    else:
        idxs = np.linspace(maxIdx / (nIdx + 1), maxIdx, nIdx, endpoint=False)
    idxs = np.round(idxs).astype(int)
    return idxs 

def median_filt(arr, radius):
    def _median_filt(img):
        img = median(img, footprint=disk(radius))
        return img
    if arr.ndim == 2:
        arr = _median_filt(arr)
    if arr.ndim == 3:
        arr = Parallel(n_jobs=-1)(delayed(_median_filt)(img) for img in arr)
    return np.stack(arr)