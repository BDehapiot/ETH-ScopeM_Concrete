#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Skimage
from skimage.segmentation import clear_border
from skimage.transform import downscale_local_mean, rescale
from skimage.morphology import disk, binary_dilation, remove_small_objects

# Scipy
from scipy.linalg import lstsq
from skimage.filters import median
from skimage.measure import label, regionprops
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import (
    shift, gaussian_filter1d, binary_fill_holes, distance_transform_edt, 
    affine_transform,
    )

#%% Parameters ----------------------------------------------------------------

rsize_factor = 8 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Paths ---------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
stack_names = ["D1_ICONX_DoS", "D11_ICONX_DoS", "D12_ICONX_corrosion", "H9_ICONX_DoS"]
stack_name = stack_names[2]

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)

#%% Functions -----------------------------------------------------------------

def resize_image(img_path):
    return downscale_local_mean(io.imread(img_path), rsize_factor)

def roll_image(img_rsize):
    idx = np.argwhere((img_rsize > 30000) == 1)
    y0, x0 = img_rsize.shape[0] // 2, img_rsize.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img_rsize, yx_shift, mode='wrap'), yx_shift 

def get_2Dmasks(stack_roll):
    
    # Intensity distribution
    avg_proj = np.mean(stack_roll, axis=0)
    hist, bins = np.histogram(
        avg_proj.flatten(), bins=1024, range=(0, 65535))    
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
    mtx_thresh *= mtx_thresh_coeff
    rod_thresh *= rod_thresh_coeff
    mtx_mask = avg_proj >= mtx_thresh
    rod_mask = avg_proj >= rod_thresh
    rod_mask = binary_fill_holes(rod_mask)
    rod_mask = binary_dilation(rod_mask, footprint=disk(3))
    mtx_mask = mtx_mask ^ rod_mask
    
    return avg_proj, mtx_thresh, rod_thresh, mtx_mask, rod_mask

def get_3Dmasks(
        img_rsize, yx_shift, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM):
    
    yx_shift = [yx_shift[0] * -1, yx_shift[1] * -1]
    
    # Shift 2D masks
    avg_proj = shift(avg_proj, yx_shift)
    rod_mask = shift(rod_mask.astype("uint8"), yx_shift)
    mtx_mask = shift(mtx_mask.astype("uint8"), yx_shift)
    rod_EDM = shift(rod_EDM, yx_shift)
    mtx_EDM = shift(mtx_EDM, yx_shift)
    
    # Normalize img
    img_norm = np.divide(img_rsize, avg_proj, where=avg_proj!=0)
    img_norm = median(img_norm, footprint=disk(8 // rsize_factor)) # parameter
    img_norm *= mtx_mask
        
    return img_norm, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM

def get_object_EDM(idx, obj_labels_3D, mtx_mask_3D):
    
    # Measure object EDM avg
    labels = obj_labels_3D.copy()
    labels[labels == idx] = 0
    obj_EDM_3D = distance_transform_edt(1 - labels > 0)
    obj_EDM_3D[mtx_mask_3D == 0] = 0
    obj_EDM_avg = np.mean(obj_EDM_3D[obj_labels_3D == idx])
    
    return obj_EDM_avg

def get_object_properties(stack_norm, mtx_mask_3D, mtx_EDM_3D):
    
    # Get object mask and labels
    obj_mask_3D = (stack_norm < 0.8) & (stack_norm > 0) # parameter
    obj_mask_3D = remove_small_objects(
        obj_mask_3D, min_size=2.5e5 * (1 / rsize_factor) ** 3) # parameter
    obj_mask_3D = clear_border(obj_mask_3D)
    obj_labels_3D = label(obj_mask_3D)
    
    # Get object properties
    obj_props = []
    mtx_EDM_3D /= np.max(mtx_EDM_3D)
    props = regionprops(obj_labels_3D, intensity_image=mtx_EDM_3D)
    for prop in props:
        obj_props.append((
            prop.label,
            prop.centroid,
            prop.area,
            prop.intensity_mean,
            prop.solidity,
            )) 
        
    # Get object EDM
    downscale_factor = 4 # parameter
    idxs = np.unique(obj_labels_3D)[1:]
    obj_labels_3D_low = rescale(
        obj_labels_3D, 1/downscale_factor, order=0).astype(int)
    mtx_mask_3D_low = obj_labels_3D_low > 0
    obj_EDM_avg = Parallel(n_jobs=-1)(
            delayed(get_object_EDM)(idx, obj_labels_3D_low, mtx_mask_3D_low) 
            for idx in idxs
            )
    
    # Merge properties
    obj_props = [
        data + (obj_EDM_avg[i] * downscale_factor,)
        for i, data in enumerate(obj_props)
        ]
    
    # # Get object EDM
    # idxs = np.unique(obj_labels_3D)[1:]
    # obj_EDM_avg = Parallel(n_jobs=-1)(
    #         delayed(get_object_EDM)(idx, obj_labels_3D, mtx_mask_3D) 
    #         for idx in idxs
    #         )
    
    # # Merge properties
    # obj_props = [
    #     data + (obj_EDM_avg[i],) for i, data in enumerate(obj_props)]
    
    return obj_mask_3D, obj_labels_3D, obj_props

# -----------------------------------------------------------------------------

def process_stack(stack_path, stack_data):
    
    # Initialize
    print(f"\n{stack_path.stem}")
    print( "-----------------------------------------------------------------")
    
    # Get img paths
    img_paths = []
    for path in stack_path.iterdir():
        if path.suffix == ".tif":
            img_paths.append(path)
            
    # Resize stack
    print("  Resize  :", end='')
    t0 = time.time()
    stack_rsize = Parallel(n_jobs=-1)(
            delayed(resize_image)(img_path) 
            for img_path in img_paths
            )
    stack_rsize = np.stack(stack_rsize)
    stack_rsize = downscale_local_mean(stack_rsize, (rsize_factor, 1, 1))
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s")
    
    # Select slices
    z_mean = np.mean(stack_rsize, axis=(1,2)) 
    z_mean_diff = np.gradient(z_mean)
    z0 = np.nonzero(z_mean_diff)[0][0] + 1
    z1 = np.where(
        (z_mean_diff > 0) & (z_mean > np.max(z_mean) * 0.9))[0][-1] + 1
    stack_rsize = stack_rsize[z0:z1, ...]  
    
    # Roll stack
    print("  Roll    :", end='')
    t0 = time.time()
    outputs = Parallel(n_jobs=-1)(
            delayed(roll_image)(img_rsize) 
            for img_rsize in stack_rsize
            )
    stack_roll = np.stack([data[0] for data in outputs])
    yx_shifts = [data[1] for data in outputs]
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Get 2D masks
    print("  2Dmasks :", end='')
    t0 = time.time()
    (
     avg_proj, mtx_thresh, rod_thresh,
     mtx_mask, rod_mask
     ) = get_2Dmasks(stack_roll)
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 

    # Get EDM
    mtx_EDM = distance_transform_edt(mtx_mask | rod_mask)
    rod_EDM = distance_transform_edt(~rod_mask)
    
    # Rescale data
    if stack_data:
        rscale_factor = np.sqrt(
            np.sum(stack_data[0]["rod_mask"]) / np.sum(rod_mask))
        print("  Rescale :", end='')
        t0 = time.time()
        stack_rsize = rescale(stack_rsize, rscale_factor)
        avg_proj = rescale(avg_proj, rscale_factor)
        mtx_mask = rescale(mtx_mask, rscale_factor, order=0)
        rod_mask = rescale(rod_mask, rscale_factor, order=0)
        mtx_EDM = rescale(mtx_EDM, rscale_factor)
        rod_EDM = rescale(rod_EDM, rscale_factor)
        t1 = time.time()
        print(f" {(t1-t0):<5.2f}s") 
        
    # Get 3D masks
    print("  3Dmasks :", end='')
    t0 = time.time()
    outputs = Parallel(n_jobs=-1)(
            delayed(get_3Dmasks)(
                img_rsize, yx_shift, avg_proj, 
                rod_mask, mtx_mask, 
                rod_EDM, mtx_EDM
                ) 
            for img_rsize, yx_shift in zip(stack_rsize, yx_shifts)
            )    
    stack_norm  = np.stack([data[0] for data in outputs])
    avg_proj_3D = np.stack([data[1] for data in outputs])
    rod_mask_3D = np.stack([data[2] for data in outputs])
    mtx_mask_3D = np.stack([data[3] for data in outputs])
    rod_EDM_3D  = np.stack([data[4] for data in outputs])
    mtx_EDM_3D  = np.stack([data[5] for data in outputs])    
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s")
    
    # Get object properties
    print("  Objects :", end='')
    t0 = time.time()
    obj_mask_3D, obj_labels_3D, obj_props = get_object_properties(
        stack_norm, mtx_mask_3D, mtx_EDM_3D)
    t1 = time.time()
    print(f" {(t1-t0):<5.2f}s") 
    
    # Print variables
    print( "  ---------")
    print(f"  zSlices    : {z0}-{z1}")
    print(f"  mtx_thresh : {int(mtx_thresh):<5d}")
    print(f"  rod_thresh : {int(rod_thresh):<5d}")
    if stack_data:
        print(f"  rscaleF    : {rscale_factor:<.3f}")
        
    # Outputs
    stack_data.append({
        "stack_path"    : stack_path,
        "stack_rsize"   : stack_rsize,
        "stack_roll"    : stack_roll,
        "stack_norm"    : stack_norm,
        "yx_shifts"     : yx_shifts,
        "avg_proj"      : avg_proj,
        "mtx_thresh"    : mtx_thresh,
        "rod_thresh"    : rod_thresh,
        "mtx_mask"      : mtx_mask,
        "rod_mask"      : rod_mask,
        "mtx_EDM"       : mtx_EDM,
        "rod_EDM"       : rod_EDM,
        "avg_proj_3D"   : avg_proj_3D,
        "rod_mask_3D"   : rod_mask_3D,
        "mtx_mask_3D"   : mtx_mask_3D,
        "rod_EDM_3D"    : rod_EDM_3D,
        "mtx_EDM_3D"    : mtx_EDM_3D,
        "obj_mask_3D"   : obj_mask_3D,
        "obj_labels_3D" : obj_labels_3D,
        "obj_props"     : obj_props,
        })

#%%

def get_distances(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

def affine_registration(coords_ref, coords_reg):
   
    if coords_ref.shape[0] < coords_ref.shape[1]:
        coords_ref = coords_ref.T
        coords_reg = coords_reg.T
    (n, dim) = coords_ref.shape
    
    # Compute least squares
    p, res, rnk, s = lstsq(
        np.hstack((coords_ref, np.ones([n, 1]))), coords_reg)
    # Get translations & transform matrix
    t, T = p[-1].T, p[:-1].T
    
    # Merge translations and transform matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = T
    transform_matrix[:3, 3] = t
    
    return transform_matrix

def get_transform_matrix(data_ref, data_reg):
    
    # Extract variables
    obj_props_ref = data_ref["obj_props"]
    obj_props_reg = data_reg["obj_props"]
    obj_labels_3D_ref = data_ref["obj_labels_3D"]
    obj_labels_3D_reg = data_ref["obj_labels_3D"]
    
    # Test object pairs
    tests = []
    props_ref = np.stack([data[2:] for data in obj_props_ref])
    props_reg = np.stack([data[2:] for data in obj_props_reg])
    for prop_ref in props_ref:
        test = []
        for prop_reg in props_reg:
            ratio = prop_ref / prop_reg
            ratio_avg = np.abs(1 - np.mean(ratio))
            ratio_std = np.std(ratio)
            test.append((ratio_avg + ratio_std) / 2)
        tests.append(test)
        
    # Identify matching pairs
    pairs = []
    for idx_ref, test in enumerate(tests):
        idx_reg = np.argmin(test)
        score = np.min(test)
        if score < 0.08: # parameter
            pairs.append((idx_ref, idx_reg, score))
    pairs = np.stack(pairs)

    # Keep best match only
    for unique in np.unique(pairs[:,1]):
        idxs = np.where(pairs[:,1] == unique)[0]
        if len(idxs) > 1:
            idx = np.argmin(pairs[idxs, 2])
            idxs = np.delete(idxs, idx)
            pairs = np.delete(pairs, idxs, axis=0)
            
    # Isolate pairs coordinates
    coords_ref, coords_reg = [], []
    labels_3D_ref = np.zeros_like(obj_labels_3D_ref)
    labels_3D_reg = np.zeros_like(obj_labels_3D_reg)
    for pair in pairs:
        idx_ref, idx_reg = int(pair[0]), int(pair[1])
        coords_ref.append(obj_props_ref[idx_ref][1])
        coords_reg.append(obj_props_reg[idx_reg][1])
        labels_3D_ref[obj_labels_3D_ref == idx_ref + 1] = idx_ref + 1
        labels_3D_reg[obj_labels_3D_reg == idx_reg + 1] = idx_ref + 1
    coords_ref = np.stack(coords_ref)
    coords_reg = np.stack(coords_reg)
    
    # Remove false pairs
    dist_ref = get_distances(coords_ref)
    dist_reg = get_distances(coords_reg)
    scores = np.median(np.abs(dist_ref - dist_reg), axis=0)
    outliers = np.where(scores > 2)[0]
    coords_ref = np.delete(coords_ref, outliers, axis=0)
    coords_reg = np.delete(coords_reg, outliers, axis=0)
    for outlier in outliers:
        labels_3D_ref[labels_3D_ref == pairs[outlier, 0] + 1] = 0
        labels_3D_reg[labels_3D_reg == pairs[outlier, 0] + 1] = 0
        
    # Compute transformation matrix
    transform_matrix = affine_registration(coords_ref, coords_reg)
    
    return transform_matrix

#%% Execute -------------------------------------------------------------------

# Execute
stack_data = []
for stack_path in stack_paths:
    if stack_name in stack_path.name: 
        process_stack(stack_path, stack_data)
       
# transform_matrices = []
# data_ref = stack_data[0]
# for i in range(1, len(stack_data)):
#     transform_matrices.append(
#         get_transform_matrix(stack_data[0], stack_data[i]))
       
#%%

idx = 3
obj_labels_3D_ref = stack_data[0]["obj_labels_3D"]
obj_labels_3D_reg = stack_data[idx]["obj_labels_3D"]
obj_props_ref = stack_data[0]["obj_props"]
obj_props_reg = stack_data[idx]["obj_props"]
stack_rsize_ref = stack_data[0]["stack_rsize"]
stack_rsize_reg = stack_data[idx]["stack_rsize"]

# # Display
# import napari
# viewer = napari.Viewer()
# viewer.add_labels(obj_labels_3D_ref)
# viewer.add_labels(obj_labels_3D_reg)

# -----------------------------------------------------------------------------

# Test object pairs (ref and reg)
obj_tests = []
for obj_prop_ref in obj_props_ref:
    obj_test = []
    for obj_prop_reg in obj_props_reg:
        tmp = [
            obj_prop_ref[2] / obj_prop_reg[2],
            obj_prop_ref[3] / obj_prop_reg[3],
            obj_prop_ref[4] / obj_prop_reg[4],
            obj_prop_ref[5] / obj_prop_reg[5],
            ]
        obj_test_avg = np.abs(1 - np.mean(tmp))
        obj_test_std = np.std(tmp)
        obj_test.append((obj_test_avg + obj_test_std) / 2)
    obj_tests.append(obj_test)      

# Identify best matching pairs
mtchs = []
for idx_ref, obj_test in enumerate(obj_tests):
    idx_reg = np.argmin(obj_test)
    score = np.min(obj_test)
    if score < 0.08: # parameter
        # print(f"ref {idx_ref:03d} | reg {idx_reg:03d} | {score:.3f}")
        mtchs.append((idx_ref, idx_reg, score))
mtchs = np.stack(mtchs)

# Solve multiple matches
uniques = np.unique(mtchs[:,1])
for unique in uniques:
    idxs = np.where(mtchs[:,1] == unique)[0]
    if len(idxs) > 1:
        idx = np.argmin(mtchs[idxs, 2])
        idxs = np.delete(idxs, idx)
        mtchs = np.delete(mtchs, idxs, axis=0)

# Update match parameters
mtch_coords_ref, mtch_coords_reg = [], []
mtch_labels_3D_ref = np.zeros_like(obj_labels_3D_ref)
mtch_labels_3D_reg = np.zeros_like(obj_labels_3D_reg)
for mtch in mtchs:
    idx_ref, idx_reg = int(mtch[0]), int(mtch[1])
    mtch_coords_ref.append(obj_props_ref[idx_ref][1])
    mtch_coords_reg.append(obj_props_reg[idx_reg][1])
    mtch_labels_3D_ref[obj_labels_3D_ref == idx_ref + 1] = idx_ref + 1
    mtch_labels_3D_reg[obj_labels_3D_reg == idx_reg + 1] = idx_ref + 1
mtch_coords_ref = np.stack(mtch_coords_ref)
mtch_coords_reg = np.stack(mtch_coords_reg)

# 
def get_distances(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix

mtch_dist_ref = get_distances(mtch_coords_ref)
mtch_dist_reg = get_distances(mtch_coords_reg)
mtch_dist_scr = np.median(np.abs(mtch_dist_ref - mtch_dist_reg), axis=0)
outliers = np.where(mtch_dist_scr > 2)[0]
mtch_coords_ref = np.delete(mtch_coords_ref, outliers, axis=0)
mtch_coords_reg = np.delete(mtch_coords_reg, outliers, axis=0)
for outlier in outliers:
    mtch_labels_3D_ref[mtch_labels_3D_ref == mtchs[outlier, 0] + 1] = 0
    mtch_labels_3D_reg[mtch_labels_3D_reg == mtchs[outlier, 0] + 1] = 0
    
# Display
import napari
viewer = napari.Viewer()
viewer.add_labels(mtch_labels_3D_ref)
viewer.add_labels(mtch_labels_3D_reg)

# -----------------------------------------------------------------------------

def affine_registration(coords_ref, coords_reg):
    
    if coords_ref.shape[0] < coords_ref.shape[1]:
        coords_ref = coords_ref.T
        coords_reg = coords_reg.T
    (n, dim) = coords_ref.shape
    
    # Compute least squares
    p, res, rnk, s = lstsq(
        np.hstack((coords_ref, np.ones([n, 1]))), coords_reg)
    # Get translation
    t = p[-1].T
    # Get transform matrix
    T = p[:-1].T
    
    # Merge translation and transform
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = T
    transform_matrix[:3, 3] = t
    
    return transform_matrix

# Compute and apply affine registration
transform_matrix = affine_registration(mtch_coords_ref, mtch_coords_reg)
stack_rsize_reg = affine_transform(stack_rsize_reg, transform_matrix)

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(stack_rsize_reg)
viewer.add_image(stack_rsize_ref)

#%% Save ----------------------------------------------------------------------

# for data in stack_data:
    
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_rsize.tif"),
#         data["stack_rsize"].astype("float32"), check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_roll.tif"),
#         data["stack_roll"].astype("float32"), check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_norm.tif"),
#         data["stack_norm"].astype("float32"), check_contrast=False,
#         )
    
#     # -------------------------------------------------------------------------
    
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_avg_proj.tif"),
#         data["avg_proj"].astype("float32"), check_contrast=False,
#         )    
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_rod_mask.tif"),
#         data["rod_mask"].astype("uint8") * 255, check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_mtx_mask.tif"),
#         data["mtx_mask"].astype("uint8") * 255, check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_rod_EDM.tif"),
#         data["rod_EDM"].astype("float32"), check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_mtx_EDM.tif"),
#         data["mtx_EDM"].astype("float32"), check_contrast=False,
#         )
    
#     # -------------------------------------------------------------------------
   
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_avg_proj_3D.tif"),
#         data["avg_proj_3D"].astype("float32"), check_contrast=False,
#         )    
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_rod_mask_3D.tif"),
#         data["rod_mask_3D"].astype("uint8") * 255, check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_mtx_mask_3D.tif"),
#         data["mtx_mask_3D"].astype("uint8") * 255, check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_rod_EDM_3D.tif"),
#         data["rod_EDM_3D"].astype("float32"), check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_mtx_EDM_3D.tif"),
#         data["mtx_EDM_3D"].astype("float32"), check_contrast=False,
#         )

#%%

# def resize_image(img_path):
#     return downscale_local_mean(io.imread(img_path), 1)