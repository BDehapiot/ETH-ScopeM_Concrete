#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.filters import gaussian
from skimage.transform import downscale_local_mean
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
from functions import shift_stack, norm_stack

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
raw_path = Path(data_path, "0-raw")
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H9_ICONX_DoS",
    ]

# Parameters
overwrite = False
df = 8 # downscale factor

#%% Function(s) ---------------------------------------------------------------

def format_stack(path, experiment_path, df):
    
    global stack, metadata, centroids, stack_shift, stack_norm, obj_mask, obj_labels, props
    
    # Initialize --------------------------------------------------------------

    name = path.name    

    # Read --------------------------------------------------------------------
    
    print(f"(format) {name}")
    t0 = time.time()
    print(" - Read : ", end='')
    
    stack = []
    img_paths = list(path.glob("**/*.tif"))
    for img_path in img_paths:
        stack.append(io.imread(img_path))
    stack = np.stack(stack)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Crop --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Crop : ", end='')

    def nearest_divisibles(value, df):
        divisor = int(2 ** np.log2(df))
        lowDiv = value - (value % divisor)
        if lowDiv == value:
            highDiv = value + divisor
        else:
            highDiv = lowDiv + divisor
        return lowDiv, highDiv
    
    # Select slices
    z_mean = np.mean(stack, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
     
    # Crop (zyx)   
    z0 = nearest_divisibles(z0, df)[1] 
    z1 = nearest_divisibles(z1, df)[0] 
    nYdiv = nearest_divisibles(stack.shape[1], df)[0]
    nXdiv = nearest_divisibles(stack.shape[2], df)[0]
    y0 = (stack.shape[1] - nYdiv) // 2 
    y1 = y0 + nYdiv
    x0 = (stack.shape[2] - nXdiv) // 2 
    x1 = x0 + nXdiv
    stack = stack[z0:z1, y0:y1, x0:x1] 

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

    # Downscale ---------------------------------------------------------------

    t0 = time.time()
    print(" - Downscale : ", end='')
       
    stack = downscale_local_mean(stack, df)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Shift -------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Shift : ", end='')

    def get_centroid(img):
        idx = np.argwhere((img > 30e3) == 1) # parameter
        y0, x0 = img.shape[0] // 2, img.shape[1] // 2
        y1, x1 = np.mean(idx, axis=0)
        return [y0 - y1, x0 - x1]
    
    centroids = Parallel(n_jobs=-1)(
            delayed(get_centroid)(img) 
            for img in stack
            )
    
    stack_shift = shift_stack(stack, centroids)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Mask --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Mask : ", end='')

    # Median projection
    med_proj = np.median(stack_shift, axis=0)

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
    stack_norm = norm_stack(
        stack, med_proj, centroids,
        radius= 16 // df, # Parameter
        mask=mtx_mask
        )
    
    # Object mask and labels
    obj_mask = (stack_norm < 0.8) & (stack_norm > 0) # parameter
    obj_mask = remove_small_objects(
        obj_mask, min_size=2.5e5 * (1 / df) ** 3) # parameter
    obj_mask = clear_border(obj_mask)
    obj_labels = label(obj_mask)
    
    # Get object properties
    obj_props = []
    mtx_EDM_3D = shift_stack(mtx_EDM, centroids, reverse=True)
    mtx_EDM_3D /= np.max(mtx_EDM_3D)
    props = regionprops(obj_labels, intensity_image=mtx_EDM_3D)
    # for prop in props:
    #     obj_props.append((
    #         prop.label,
    #         prop.centroid,
    #         prop.area,
    #         prop.intensity_mean,
    #         prop.solidity,
    #         )) 
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Save --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Data
    save_name = name + f"_crop_df{df}.tif" 
    save_path = experiment_path / save_name
    io.imsave(save_path, stack, check_contrast=False)
        
    # Metadata
    metadata_path = experiment_path / (name + "_metadata_o.pkl") 
    metadata = {
        "df"        : df,
        "name"      : save_name,
        "path"      : save_path,
        "shape"     : stack.shape,
        "crop"      : (z0, z1, y0, y1, x0, x1),
        "centroids" : centroids,
        }
    
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

#%% Execute -------------------------------------------------------------------

t = 0

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            
            if f"Time{t}" in path.name:
            
                test_path = experiment_path / (path.name + "_crop_df1.tif")
                if not test_path.is_file():
                    format_stack(path, experiment_path, df)   
                elif overwrite:
                    format_stack(path, experiment_path, df)  

#%% Display -------------------------------------------------------------------

obj_props = []
for prop in props:
    obj_props.append((
        prop.label,
        prop.centroid,
        prop.area,
        prop.intensity_mean,
        prop.solidity,
        )) 

import napari
viewer = napari.Viewer()
viewer.add_image(stack_norm)
viewer.add_labels(obj_labels)