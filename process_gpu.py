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
    disk, ball, binary_dilation, binary_erosion, remove_small_objects
    )

# Scipy
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import (
    gaussian_filter1d, binary_fill_holes, distance_transform_edt
    )

# Functions
from functions import filt_median, shift_stack, norm_stack, predict

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
raw_path = Path(data_path, "0-raw")
model_path = Path.cwd() / f"model-weights_matrix_p0256_d{df}.h5"
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H9_ICONX_DoS",
    ]

#%% Function(s) ---------------------------------------------------------------

def process_stack(path, experiment_path, df):
    
    global \
        stack, metadata, centers,\
        stack_shift, stack_norm,\
        mtx_mask, rod_mask,\
        mtx_EDM, rod_EDM,\
        mtx_EDM_3D, mtx_EDM_3D_low,\
        void_mask_3D, liquid_mask_3D,\
        obj_df, obj_probs, obj_mask_3D, obj_labels_3D, obj_labels_3D_low,\
        obj_dist, mtx_dist, void_area, liquid_area,\
        mtx_EDM_avg, obj_EDM_avg,\
        out_mask, out_mask_3D_low,\
        obj_cat,\
        objects

    # Initialize --------------------------------------------------------------

    name = path.name    

    #%% Read ------------------------------------------------------------------
    
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
    
    #%% Crop ------------------------------------------------------------------
    
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

    #%% Downscale -------------------------------------------------------------

    t0 = time.time()
    print(" - Downscale : ", end='')
       
    stack = downscale_local_mean(stack, df)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
        
    #%% Preprocess ------------------------------------------------------------
    
    t0 = time.time()
    print(" - Preprocess : ", end='')
    
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
    
    #%% Predict ---------------------------------------------------------------
    
    t0 = time.time()

    # Predict objects
    obj_probs = predict(stack_norm, model_path, subset=1000)
    
    t1 = time.time()
    print(f" - Predict : {(t1-t0):<5.2f}s") 
    
    #%% Segment ---------------------------------------------------------------
    
    t0 = time.time()
    print(" - Segment : ", end='')

    def get_obj_mask(obj_probs, min_size, erode_radius):
        obj_mask_3D = obj_probs > 0.5
        obj_mask_3D = remove_small_objects(obj_mask_3D, min_size=min_size)
        if erode_radius > 1:
            obj_mask_3D = binary_erosion(obj_mask_3D, footprint=ball(erode_radius))
        return obj_mask_3D
    
    # Format data
    stack_norm = filt_median(stack_norm, 8 // df) # Parameter (8)
    obj_mask_3D = get_obj_mask(
        obj_probs, 1.5e4 * (1 / df) ** 3, 8 // df) # Parameter (1.5e4, 8)
    obj_labels_3D = label(obj_mask_3D)
    mtx_EDM_3D = shift_stack(mtx_EDM, centers, reverse=True)
    
    # Measure object EDM & percentile low
    obj_EDM, obj_pcl = [], []
    rvl_lab = (obj_labels_3D[obj_labels_3D > 0]).ravel()
    rvl_EDM = (mtx_EDM_3D[obj_labels_3D > 0]).ravel()
    rvl_int = (stack_norm[obj_labels_3D > 0]).ravel()
    for idx in range(1, np.max(obj_labels_3D)):
        obj_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
        obj_pcl.append(np.percentile(rvl_int[rvl_lab == idx], 5)) # Parameter (5)
    obj_EDM = np.stack(obj_EDM)
    obj_pcl = np.stack(obj_pcl)
        
    # Find ref. points & fit parameters
    maxEDM = np.max(obj_EDM)
    x0 = maxEDM * 0.1 # Parameter (0.1)
    x1 = maxEDM - x0
    y0 = np.percentile(obj_pcl[obj_EDM < x0], 25) # Parameter (25)
    y1 = np.percentile(obj_pcl[obj_EDM > x1], 25) # Parameter (25)
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0
    
    # Determine threshold
    obj_pcl -= (a * obj_EDM + b)
    minPcl, maxPcl = np.min(obj_pcl), np.max(obj_pcl)
    hist, bins = np.histogram(obj_pcl, bins=100, range=(minPcl, maxPcl))   
    hist = gaussian_filter1d(hist, sigma=2) # Parameter
    peaks, _ = find_peaks(hist, distance=33) # Parameter
    _, _, lws, rws = peak_widths(hist, peaks, rel_height=0.5) # Parameter
    idx = np.argmin(peaks)
    thresh = rws[idx] + (rws[idx] - peaks[idx])
    thresh_val = bins[int(thresh) + 1] + y0
    thresh_val *= 1.0 # Parameter (1.0)
    # print(thresh_val)
        
    # plt.plot(hist)
    # plt.axvline(x=peaks[idx])
    # plt.axvline(x=rws[idx])
    # plt.axvline(x=thresh, color="red")

    # Correct stack_norm
    obj_mask_3D = get_obj_mask(
        obj_probs, 1.5e4 * (1 / df) ** 3, 4 // df) # Parameter (1.5e4, 4)
    mtx_EDM_3D = (a * mtx_EDM_3D + b) - y0
    mtx_EDM_3D[obj_mask_3D == 0] = 0
    stack_norm_corr = stack_norm - mtx_EDM_3D
    
    # Get void & liquide masks
    void_mask_3D = stack_norm_corr.copy()
    void_mask_3D[obj_mask_3D == 0] = 0
    void_mask_3D = void_mask_3D < thresh_val
    void_mask_3D[obj_mask_3D == 0] = 0
    liquid_mask_3D = stack_norm_corr.copy()
    liquid_mask_3D[obj_mask_3D == 0] = 0
    liquid_mask_3D = liquid_mask_3D > thresh_val
    liquid_mask_3D[obj_mask_3D == 0] = 0
    
    # Filter masks
    liquid_mask_3D = remove_small_objects(liquid_mask_3D, min_size=256 // df)
    void_mask_3D[(liquid_mask_3D == 0) & (obj_mask_3D == 1)] = 1
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    #%% Objects ---------------------------------------------------------------
    
    t0 = time.time()
    print(" - Objects : ", end='')
        
    def get_object_measurments(
            idx, obj_labels_3D, 
            void_mask_3D, liquid_mask_3D,
            mtx_EDM_3D, cat_mask_3D
            ):
        labels = obj_labels_3D.copy()
        labels[labels == idx] = 0
        void_area = np.sum(void_mask_3D[obj_labels_3D == idx])
        liquid_area = np.sum(liquid_mask_3D[obj_labels_3D == idx])
        obj_EDM_3D = distance_transform_edt(1 - labels > 0)
        obj_EDM_3D[obj_labels_3D == 0] = 0 # Don't know why
        obj_dist = np.nanmean(obj_EDM_3D[obj_labels_3D == idx])
        mtx_dist = np.nanmean(mtx_EDM_3D[obj_labels_3D == idx])
        category = np.max(cat_mask_3D_low[obj_labels_3D == idx])
        return void_area, liquid_area, obj_dist, mtx_dist, category
    
    # Parameters
    obj_df = 16 // df # parameter
    
    # Object mask and labels
    obj_mask_3D = get_obj_mask(
        obj_probs, 1.5e5 * (1 / df) ** 3, 4 // df) # Parameter (1.5e5, 4)
    obj_mask_3D = clear_border(obj_mask_3D)
    obj_labels_3D = label(obj_mask_3D)

    # Category masks
    cat_mask1 = binary_erosion(mtx_mask, footprint=disk(obj_df))
    cat_mask2 = binary_dilation(rod_mask, footprint=disk(obj_df))
    
    # Downscale data
    obj_labels_3D_low = rescale(
        obj_labels_3D, 1 / obj_df, order=0).astype(int)
    void_mask_3D_low = rescale(void_mask_3D, 1 / obj_df, order=0)
    liquid_mask_3D_low = rescale(liquid_mask_3D, 1 / obj_df, order=0)
    mtx_EDM_3D_low = rescale(
        shift_stack(mtx_EDM, centers, reverse=True), 1 / obj_df)
    cat_mask1_3D_low = rescale(
        shift_stack(cat_mask1, centers, reverse=True), 1 / obj_df, order=0)
    cat_mask2_3D_low = rescale(
        shift_stack(cat_mask2, centers, reverse=True), 1 / obj_df, order=0)
    cat_mask_3D_low = 1 - cat_mask1_3D_low + cat_mask2_3D_low

    # Get object measurments
    idxs = np.unique(obj_labels_3D)[1:]
    outputs = Parallel(n_jobs=-1)(
            delayed(get_object_measurments)(
                idx, obj_labels_3D_low, 
                void_mask_3D_low, liquid_mask_3D_low, 
                mtx_EDM_3D_low, cat_mask_3D_low
                ) 
            for idx in idxs
            )
    void_area   = [data[0] for data in outputs]
    liquid_area = [data[1] for data in outputs] 
    obj_dist    = [data[2] for data in outputs]
    mtx_dist    = [data[3] for data in outputs]
    category    = [data[4] for data in outputs]   

    # Get object properties
    objects = []
    props = regionprops(obj_labels_3D)
    for i, prop in enumerate(props):
        objects.append({
            "label"       : prop.label,
            "centroid"    : prop.centroid,
            "area"        : prop.area,
            "solidity"    : prop.solidity,
            "void_area"   : void_area[i] * (obj_df ** 3),
            "liquid_area" : liquid_area[i] * (obj_df ** 3),
            "obj_dist"    : obj_dist[i] * obj_df,
            "mtx_dist"    : mtx_dist[i] * obj_df,
            "category"    : category[i],
            })
            
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    #%% Save ------------------------------------------------------------------
    
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
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_void_mask.tif"), 
        void_mask_3D.astype("uint8") * 255, check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_liquid_mask.tif"), 
        liquid_mask_3D.astype("uint8") * 255, check_contrast=False
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
    print(f"{(t1-t0):<5.2f}s\n")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            # if "Time0" in path.name:
            test_path = experiment_path / (path.name + f"_crop_df{df}.tif")
            if not test_path.is_file():
                process_stack(path, experiment_path, df)   
            elif overwrite:
                process_stack(path, experiment_path, df)  

#%% Display -------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(out_mask_3D_low)
# viewer.add_labels(obj_labels_3D_low)
# viewer.add_image(mtx_EDM_3D_low)
# viewer.add_image(obj_mask_3D)
# viewer.add_labels(obj_labels_3D)