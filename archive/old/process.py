#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_dilation, binary_erosion, remove_small_objects
    )

# Scipy
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import (
    shift, gaussian_filter1d, binary_fill_holes, distance_transform_edt
    )

# Functions 
from functions import normalize_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H9_ICONX_DoS",
    ]

# Parameters
overwrite = True
df = 4 # downscale factor for preprocessing

#%% Function(s) ---------------------------------------------------------------

def preprocess_stack(path, experiment_path, df):
    
    global metadata, new_metadata, stack, stack_norm, obj_mask, obj_labels # dev
    
    # Initialize --------------------------------------------------------------
        
    name = path.name.replace(f"_crop_df{df}.tif", "")
    
    # Read --------------------------------------------------------------------

    print(f"(preprocess) {name}")
    t0 = time.time()
    print(" - Read :", end='')
    
    # Data
    stack = io.imread(path)
        
    # Metadata
    metadata_path = experiment_path / (name + "_metadata_o.pkl") 
    with open(metadata_path, 'rb') as file:
        metadata = pickle.load(file)
    
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Roll --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Roll :", end='')
    
    def roll_image(img):
        idx = np.argwhere((img > 30000) == 1)
        y0, x0 = img.shape[0] // 2, img.shape[1] // 2
        y1, x1 = np.mean(idx, axis=0)
        yx_shift = [y0 - y1, x0 - x1]
        return shift(img, yx_shift, mode='wrap'), yx_shift
    
    outputs = Parallel(n_jobs=-1)(
        delayed(roll_image)(img) 
        for img in stack
        )
    stack_roll = np.stack([data[0] for data in outputs])
    yx_shift = [data[1] for data in outputs]
    
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Mask --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Mask : ", end='')

    # Median projection
    med_proj = np.median(stack_roll, axis=0)

    # Intensity distribution
    hist, bins = np.histogram(
        med_proj.flatten(), bins=1024, range=(0, 65535))    
    hist = gaussian_filter1d(hist, sigma=2)
    pks, _ = find_peaks(hist, distance=30)
    proms = peak_prominences(hist, pks)[0]
    sorted_pks = pks[np.argsort(proms)[::-1]]
    select_pks = sorted_pks[:3]
    
    # Get masks
    mtx_thresh = bins[select_pks[1]] - (
        (bins[select_pks[1]] - bins[select_pks[0]]) / 2)
    rod_thresh = bins[select_pks[2]] - (
        (bins[select_pks[2]] - bins[select_pks[1]]) / 2)
    mtx_mask = med_proj >= mtx_thresh
    rod_mask = med_proj >= rod_thresh
    rod_mask = binary_fill_holes(rod_mask)
    rod_mask = binary_dilation(rod_mask, footprint=disk(1)) # Parameter
    mtx_mask = mtx_mask ^ rod_mask
    mtx_mask = binary_erosion(mtx_mask, footprint=disk(1)) # Parameter

    # Get EDMs
    mtx_EDM = distance_transform_edt(mtx_mask | rod_mask)
    rod_EDM = distance_transform_edt(~rod_mask)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Objects -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Objects : ", end='')
    
    # Normalize stack
    stack_norm = normalize_stack(
        stack, med_proj, yx_shift, radius=df, mask=mtx_mask)

    # Object mask and labels
    obj_mask = (stack_norm < 0.8) & (stack_norm > 0) # parameter
    obj_mask = remove_small_objects(
        obj_mask, min_size=2.5e5 * (1 / df) ** 3) # parameter
    obj_mask = clear_border(obj_mask)
    obj_labels = label(obj_mask)
    
    # Get object properties
    obj_props = []
    # mtx_EDM_3D /= np.max(mtx_EDM_3D)
    props = regionprops(obj_labels, intensity_image=mtx_EDM)
    for prop in props:
        obj_props.append((
            prop.label,
            prop.centroid,
            prop.area,
            prop.intensity_mean,
            prop.solidity,
            )) 
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Rescale -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Rescale : ", end='')
     
    def rescale_mask(mask, rf):
        mask = rescale(mask.astype(float), rf)
        mask = gaussian(mask, sigma=rf * 2) > 0.5
        return mask
    
    def rescale_shift(yx_shift, rf):
        nZ = len(yx_shift)
        z_old = np.arange(nZ)
        z_new = np.linspace(0, nZ - 1, int(nZ * rf))
        y_old = np.stack([data[0] for data in yx_shift])
        f_linear = interp1d(z_old, y_old, kind='linear')
        y_new = f_linear(z_new)
        x_old = np.stack([data[1] for data in yx_shift])
        f_linear = interp1d(z_old, x_old, kind='linear')
        x_new = f_linear(z_new)
        
        yx_new = []
        for y, x in zip(y_new, x_new):
            yx_new.append((y * rf, x * rf))
            
        return yx_new
    
    med_projs, mtx_masks, rod_masks, yx_shifts = [], [], [], []
    for f in metadata["dfs"]:
        rf = df / f
        if rf != 1:
            med_projs.append(rescale(med_proj, rf))
            mtx_masks.append(rescale_mask(mtx_mask, rf))
            rod_masks.append(rescale_mask(rod_mask, rf)) 
            yx_shifts.append(rescale_shift(yx_shift, rf))
        else:
            med_projs.append(med_proj)
            mtx_masks.append(mtx_mask)
            rod_masks.append(rod_mask)
            yx_shifts.append(yx_shift)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Save --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Metadata
    new_metadata_path = experiment_path / (name + "_metadata_oo.pkl") 
    new_metadata = metadata.copy()
    new_metadata["med_projs"] = med_projs
    new_metadata["mtx_masks"] = mtx_masks
    new_metadata["rod_masks"] = rod_masks
    new_metadata["yx_shifts"] = yx_shifts
    
    with open(new_metadata_path, 'wb') as file:
        pickle.dump(new_metadata, file)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
#%% Execute -------------------------------------------------------------------

t = 2

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        for path in experiment_path.glob(f"*_crop_df{df}*"):
            
            if f"Time{t}" in path.name:
            
                name = path.name.replace(f"_crop_df{df}.tif", "")
                test_path = experiment_path / (name + "_metadata_oo.pkl")
                if not test_path.is_file():
                    preprocess_stack(path, experiment_path, df)
                elif overwrite:
                    preprocess_stack(path, experiment_path, df)
                
#%%

# idx = 0
# mask_opacity = 0.5

import napari
viewer = napari.Viewer()
# viewer.add_image(stack)
viewer.add_image(stack_norm)
viewer.add_labels(obj_labels)

# import napari
# viewer = napari.Viewer()
# viewer.add_image(new_metadata["med_projs"][idx])
# viewer.add_image(new_metadata["mtx_masks"][idx], blending="additive", colormap="bop orange", opacity=mask_opacity)
# viewer.add_image(new_metadata["rod_masks"][idx], blending="additive", colormap="bop blue", opacity=mask_opacity)
