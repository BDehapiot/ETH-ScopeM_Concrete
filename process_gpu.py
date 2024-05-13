#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.transform import downscale_local_mean, rescale
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_dilation, binary_erosion, remove_small_objects
    )

# Scipy
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import (
    gaussian_filter1d, binary_fill_holes, distance_transform_edt
    )

# Functions
from functions import shift_stack, norm_stack, predict

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
raw_path = Path(data_path, "0-raw")
model_path = Path.cwd() / f"model-weights_matrix_p0256_d{df}.h5"
experiments = [
    # "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    "H9_ICONX_DoS",
    ]

#%% Function(s) ---------------------------------------------------------------

def process_stack(path, experiment_path, df):
    
    global \
        stack, metadata, centers,\
        stack_shift, stack_norm,\
        mtx_mask, rod_mask,\
        mtx_EDM, rod_EDM,\
        mtx_EDM_3D, mtx_EDM_3D_low,\
        obj_df, obj_probs, obj_mask_3D, obj_labels_3D, obj_labels_3D_low,\
        mtx_EDM_avg, obj_EDM_avg,\
        out_mask, out_mask_3D_low,\
        obj_cat,\
        objects

    # Initialize --------------------------------------------------------------

    name = path.name    

    # Read --------------------------------------------------------------------
    
    print(f"(process) {name}")
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
        
    # Process -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Process : ", end='')
    
    def get_centers(img):
        idx = np.argwhere((img > 30e3) == 1) # parameter
        y0, x0 = img.shape[0] // 2, img.shape[1] // 2
        y1, x1 = np.mean(idx, axis=0)
        return [y0 - y1, x0 - x1]
    
    # Get centers
    centers = Parallel(n_jobs=-1)(
            delayed(get_centers)(img) 
            for img in stack
            )
    
    # Shift & median projection
    stack_shift = shift_stack(stack, centers)
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
    # mtx_mask = binary_erosion(mtx_mask, footprint=disk(1)) # Parameter
    
    # Get EDMs
    mtx_EDM = distance_transform_edt(binary_fill_holes(mtx_mask | rod_mask))
    rod_EDM = distance_transform_edt(~rod_mask)

    # Normalize stack
    stack_norm = norm_stack(stack, med_proj, centers, mask=mtx_mask)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Predict -----------------------------------------------------------------
    
    t0 = time.time()

    # Predict objects
    obj_probs = predict(stack_norm, model_path, subset=1000)
    
    t1 = time.time()
    print(f" - Predict : {(t1-t0):<5.2f}s") 
    
    # Objects -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Objects : ", end='')
        
    def get_object_EDM(idx, obj_labels_3D, mtx_EDM_3D):
        labels = obj_labels_3D.copy()
        labels[labels == idx] = 0
        obj_EDM_3D = distance_transform_edt(1 - labels > 0)
        obj_EDM_3D[obj_labels_3D == 0] = 0 # Don't know why
        obj_dist = np.nanmean(obj_EDM_3D[obj_labels_3D == idx])
        mtx_dist = np.nanmean(mtx_EDM_3D[obj_labels_3D == idx])
        return obj_dist, mtx_dist
    
    # Parameters
    obj_df = 16 // df # parameter
        
    # Object mask and labels
    obj_mask_3D = (obj_probs > 0.5) # parameter (0.5)
    obj_mask_3D = remove_small_objects(
        obj_mask_3D, min_size=1e5 * (1 / df) ** 3) # parameter (1.5e5)
    obj_mask_3D = clear_border(obj_mask_3D)
    obj_labels_3D = label(obj_mask_3D)
    
    # Get EDM measurments
    idxs = np.unique(obj_labels_3D)[1:]
    obj_labels_3D_low = rescale(
        obj_labels_3D, 1 / obj_df, order=0).astype(int)
    mtx_EDM_3D_low = rescale(
        shift_stack(mtx_EDM, centers, reverse=True), 1 / obj_df)
    outputs = Parallel(n_jobs=-1)(
            delayed(get_object_EDM)(idx, obj_labels_3D_low, mtx_EDM_3D_low) 
            for idx in idxs
            )
    obj_dist = [data[0] for data in outputs] * obj_df
    mtx_dist = [data[1] for data in outputs] * obj_df
    # obj_dist /= np.mean(obj_dist)
    # mtx_dist /= np.mean(mtx_dist)
    
    #%% -------------------------------------------------------------------------
    
    obj_df = 16 // df # parameter
    
    t0 = time.time()
    print(" - Test : ", end='')
    
    # Get outer vs. inner objects
    idxs = np.unique(obj_labels_3D)[1:]
    out_mask = binary_erosion(mtx_mask, footprint=disk(obj_df))
    # out_mask = binary_fill_holes(out_mask)
    inn_mask = binary_dilation(rod_mask, footprint=disk(obj_df))
    
    out_mask_3D_low = rescale(
        shift_stack(out_mask, centers, reverse=True), 1 / obj_df, order=0)
    inn_mask_3D_low = rescale(
        shift_stack(inn_mask, centers, reverse=True), 1 / obj_df, order=0)
    cat_mask_3D_low = 1 - out_mask_3D_low + inn_mask_3D_low

    obj_cat = []
    for idx in idxs:
        obj_cat.append(cat_mask_3D_low[obj_labels_3D_low == idx])
        
    # isOut = []
    # for idx in idxs:
    #     if np.min(out_mask_3D_low[obj_labels_3D_low == idx]) == 0:
    #         isOut.append(1)
    #         obj_labels_3D_low[obj_labels_3D_low == idx] = 1
    #     else:
    #         isOut.append(0) 
    #         obj_labels_3D_low[obj_labels_3D_low == idx] = 2
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    import napari
    viewer = napari.Viewer()
    # viewer.add_image(out_mask_3D_low, colormap="green", blending="additive", contrast_limits=(0, 2))
    # viewer.add_image(inn_mask_3D_low, colormap="magenta", blending="additive", contrast_limits=(0, 2))
    viewer.add_image(cat_mask_3D_low)
    # viewer.add_labels(obj_labels_3D_low)
    # viewer.add_image(out_mask1, colormap="green", blending="additive")
    # viewer.add_image(out_mask2, colormap="magenta", blending="additive")
    
    #%% -------------------------------------------------------------------------
    
    # Get object properties
    objects = []
    props = regionprops(obj_labels_3D)
    for i, prop in enumerate(props):
        objects.append({
            "label"    : prop.label,
            "centroid" : prop.centroid,
            "area"     : prop.area,
            "solidity" : prop.solidity,
            "obj_dist" : obj_dist[i],
            "mtx_dist" : mtx_dist[i],
            })
            
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Save --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Data
    io.imsave(
        experiment_path / (name + f"_crop_df{df}.tif"), 
        stack.astype("uint16"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_norm.tif"), 
        stack_norm.astype("float32"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_probs.tif"), 
        obj_probs.astype("float32"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_labels.tif"), 
        obj_labels_3D.astype("uint16"), check_contrast=False
        )
        
    # Metadata
    metadata_path = experiment_path / (name + f"_crop_df{df}_metadata.pkl") 
    metadata = {
        "df"       : df,
        "crops"    : (z0, z1, y0, y1, x0, x1),
        "centers"  : centers,
        "med_proj" : med_proj,
        "mtx_mask" : mtx_mask,
        "rod_mask" : rod_mask,
        "mtx_EDM"  : mtx_EDM,
        "rod_EDM"  : rod_EDM,
        "objects"  : objects,
        }
    
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            if "Time0" in path.name:
                test_path = experiment_path / (path.name + f"_crop_df{df}.tif")
                if not test_path.is_file():
                    process_stack(path, experiment_path, df)   
                elif overwrite:
                    process_stack(path, experiment_path, df)  

#%% Display -------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(out_mask_3D_low)
viewer.add_labels(obj_labels_3D_low)
# viewer.add_image(mtx_EDM_3D_low)
# viewer.add_image(obj_mask_3D)
# viewer.add_labels(obj_labels_3D)