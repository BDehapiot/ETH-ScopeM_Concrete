#%% Imports -------------------------------------------------------------------

import numpy as np
import segmentation_models as sm
from joblib import Parallel, delayed

# Skimage
from skimage.filters import median
from skimage.transform import rescale
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, ball, binary_dilation, binary_erosion, remove_small_objects
    )

# Scipy
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.ndimage import (
    shift, gaussian_filter1d, binary_fill_holes, distance_transform_edt
    )

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches, merge_patches

#%% Function(s) general -------------------------------------------------------

def filt_median(arr, radius):
    
    def _filt_median(img):
        return median(img, footprint=disk(radius))
    
    if arr.ndim == 2:
        arr = _filt_median(arr)
    
    if arr.ndim == 3:
        arr = Parallel(n_jobs=-1)(
            delayed(_filt_median)(img) 
            for img in arr
            )
        arr = np.stack(arr)
    
    return arr

# -----------------------------------------------------------------------------

def shift_stack(stack, centers, reverse=False):
    
    def shift_img(img, center):
        if reverse:         
            center = [- center[0], - center[1]]
        if img.dtype == bool:
            img = img.astype("uint8")
        return shift(img, center)
    
    # Shift 1 image / 1 center
    if stack.ndim == 2 and len(centers) == 1:
        stack = shift_img(stack, centers)

    # Shift 1 image / n centers
    elif stack.ndim == 2 and len(centers) > 1:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(stack, center) 
            for center in centers
            )
        stack = np.stack(stack)
        
    # Shift n images / n centers
    elif stack.ndim == 3:
        stack = Parallel(n_jobs=-1)(
            delayed(shift_img)(img, center) 
            for img, center in zip(stack, centers)
            )
        stack = np.stack(stack)
        
    return stack

# -----------------------------------------------------------------------------

def norm_stack(stack, med_proj, centers, radius=1, mask=None):
       
    if radius > 1:
        stack = filt_median(stack, radius)

    med_proj = shift_stack(med_proj, centers, reverse=True)        
    stack = np.divide(stack, med_proj, where=med_proj != 0)

    if mask is not None:
        mask = shift_stack(mask, centers, reverse=True)
        stack *= mask
        
    return stack

# -----------------------------------------------------------------------------

def get_obj_mask(probs, min_size, erode_radius):
    
    obj_mask = probs > 0.5
    obj_mask = remove_small_objects(obj_mask, min_size=min_size)
    
    if erode_radius > 1:
        obj_mask = binary_erosion(obj_mask, footprint=ball(erode_radius))
        
    return obj_mask

#%% Function(s) crop ----------------------------------------------------------

def crop(stack, df):
    
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
    crops = (z0, z1, y0, y1, x0, x1)
    
    return stack, crops

#%% Function(s) predict -------------------------------------------------------

def predict(stack, model_path, subset=1000):
    
    # Define model
    model = sm.Unet(
        'resnet34', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(model_path)
    idx = model_path.name.find("_p")
    size = int(model_path.name[idx + 2:idx + 6])
    overlap = size // 4

    # Define sub indexes
    nZ = stack.shape[0]
    z0s = np.arange(0, nZ, subset)
    z1s = z0s + subset
    z1s[z1s > nZ] = nZ
    
    # Normalize stack
    stack = norm_gcn(stack, mask=stack != 0)
    stack = norm_pct(stack, 0.01, 99.99, mask=stack != 0)
    
    # Predict
    probs = []
    for z0, z1 in zip(z0s, z1s):
        tmpStack = stack[z0:z1, ...]
        patches = extract_patches(tmpStack, size, overlap)
        patches = np.stack(patches)
        tmpProbs = model.predict(patches).squeeze()
        tmpProbs = merge_patches(tmpProbs, tmpStack.shape, overlap)
        probs.append(tmpProbs)
    probs = np.concatenate(probs, axis=0)
        
    return probs

#%% Function(s) preprocess ----------------------------------------------------

def preprocess(stack):

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

    return centers, med_proj, mtx_mask, rod_mask, mtx_EDM, rod_EDM, stack_norm

#%% Function(s) segment -------------------------------------------------------

def segment(norm, probs, EDM, centers, df):

    # Format data
    norm = filt_median(norm, 8 // df) # Parameter (8)
    obj_mask = get_obj_mask(
        probs, 1.5e4 * (1 / df) ** 3, 8 // df) # Parameter (1.5e4, 8)
    obj_labels = label(obj_mask)
    EDM_3D = shift_stack(EDM, centers, reverse=True)
    
    # Measure object EDM & percentile low
    obj_EDM, obj_pcl = [], []
    rvl_lab = (obj_labels[obj_labels > 0]).ravel()
    rvl_EDM = (EDM_3D[obj_labels > 0]).ravel()
    rvl_int = (norm[obj_labels > 0]).ravel()
    for idx in range(1, np.max(obj_labels)):
        obj_EDM.append(np.mean(rvl_EDM[rvl_lab == idx]))
        obj_pcl.append(
            np.percentile(rvl_int[rvl_lab == idx], 5)) # Parameter (5)
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
    obj_mask = get_obj_mask(
        probs, 1.5e4 * (1 / df) ** 3, 8 // df) # Parameter (1.5e4, 8)
    EDM_3D = (a * EDM_3D + b) - y0
    EDM_3D[obj_mask == 0] = 0
    norm_corr = norm - EDM_3D
    
    # Get air & liquide masks
    air_mask = norm_corr.copy()
    air_mask[obj_mask == 0] = 0
    air_mask = air_mask < thresh_val
    air_mask[obj_mask == 0] = 0
    liquid_mask = norm_corr.copy()
    liquid_mask[obj_mask == 0] = 0
    liquid_mask = liquid_mask > thresh_val
    liquid_mask[obj_mask == 0] = 0
    
    return norm_corr, air_mask, liquid_mask

#%% Function(s) objects -------------------------------------------------------

def objects(
        probs, 
        mtx_mask, rod_mask, 
        air_mask, liquid_mask, 
        mtx_EDM, centers, df
        ):

    def measure_objects(
            idx, obj_labels, 
            air_mask, liquid_mask,
            mtx_EDM, cat_mask,
            ):
        labels = obj_labels.copy()
        labels[labels == idx] = 0
        area = np.sum(obj_labels == idx)
        air_area = np.sum(air_mask[obj_labels == idx])
        liquid_area = np.sum(liquid_mask[obj_labels == idx])
        obj_EDM = distance_transform_edt(1 - labels > 0)
        obj_EDM[obj_labels == 0] = 0 # Don't know why
        obj_dist = np.nanmean(obj_EDM[obj_labels == idx])
        mtx_dist = np.nanmean(mtx_EDM[obj_labels == idx])
        category = np.max(cat_mask_low[obj_labels == idx])
        return area, air_area, liquid_area, obj_dist, mtx_dist, category
    
    # Parameters
    obj_df = 16 // df # parameter (16)
    
    # Object mask and labels
    obj_mask = get_obj_mask(
        probs, 5e4 * (1 / df) ** 3, 4 // df) # Parameter (1.5e5, 4)
    obj_mask = clear_border(obj_mask)
    obj_labels = label(obj_mask)
    
    # mtx_EDM
    mtx_EDM = shift_stack(mtx_EDM, centers, reverse=True)

    # Category masks
    cat_mask1 = binary_erosion(mtx_mask, footprint=disk(obj_df))
    cat_mask2 = binary_dilation(rod_mask, footprint=disk(obj_df))
    
    # Downscale data
    obj_labels_low = rescale(obj_labels, 1 / obj_df, order=0).astype(int)
    air_mask_low = rescale(air_mask, 1 / obj_df, order=0)
    liquid_mask_low = rescale(liquid_mask, 1 / obj_df, order=0)
    mtx_EDM_low = rescale(mtx_EDM, 1/ obj_df)
    cat_mask1_low = rescale(
        shift_stack(cat_mask1, centers, reverse=True), 1 / obj_df, order=0)
    cat_mask2_low = rescale(
        shift_stack(cat_mask2, centers, reverse=True), 1 / obj_df, order=0)
    cat_mask_low = 1 - cat_mask1_low + cat_mask2_low

    # Get object measurments
    idxs = np.unique(obj_labels)[1:]
    outputs = Parallel(n_jobs=-1)(
            delayed(measure_objects)(
                idx, obj_labels_low, 
                air_mask_low, liquid_mask_low, 
                mtx_EDM_low, cat_mask_low
                ) 
            for idx in idxs
            )
    area        = [data[0] for data in outputs]
    air_area    = [data[1] for data in outputs]
    liquid_area = [data[2] for data in outputs] 
    obj_dist    = [data[3] for data in outputs]
    mtx_dist    = [data[4] for data in outputs]
    category    = [data[5] for data in outputs]   
    
    # Scale areas
    area = np.stack(area) * obj_df ** 3
    air_area = np.stack(air_area) * obj_df ** 3
    liquid_area = np.stack(liquid_area) * obj_df ** 3
    ratio = liquid_area / (air_area + liquid_area)
    air_area = np.round(area * (1 - ratio)).astype(int)
    liquid_area = np.round(area * ratio).astype(int)

    # Get object data
    obj_data = []
    props = regionprops(obj_labels)
    for i, prop in enumerate(props):
        
        obj_data.append({
            "label"       : prop.label,
            "ctrd_z"      : prop.centroid[0],
            "ctrd_y"      : prop.centroid[1],
            "ctrd_x"      : prop.centroid[2],
            "area"        : area[i],
            "air_area"    : air_area[i],
            "liquid_area" : liquid_area[i],
            "ratio"       : ratio[i],
            "solidity"    : prop.solidity,
            "obj_dist"    : obj_dist[i] * obj_df,
            "mtx_dist"    : mtx_dist[i] * obj_df,
            "category"    : category[i],
            })
        
    return obj_labels, obj_data