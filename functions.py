#%% Imports -------------------------------------------------------------------

import numpy as np

#%% Functions -----------------------------------------------------------------

def normalize_gcn(img):
    img = img - np.mean(img)
    img = img / np.std(img)     
    return img

def normalize_pct(img, min_pct, max_pct):
    pMin = np.percentile(img, min_pct); # print(pMin)
    pMax = np.percentile(img, max_pct); # print(pMax)
    if pMax == 0:
        pMax = np.max(img) # Debug
    np.clip(img, pMin, pMax, out=img)
    img -= pMin
    img /= (pMax - pMin)   
    return img