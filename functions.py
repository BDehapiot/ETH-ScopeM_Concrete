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

#%% Function : process_stacks -------------------------------------------------

def process_stacks(
        stack_path,
        stack_data,
        rsize_factor,
        mtx_thresh_coeff,
        rod_thresh_coeff,
        ):
    
    # Nested functions --------------------------------------------------------
    
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

    def get_3Dmasks(img_rsize, yx_shift, avg_proj, rod_mask, mtx_mask, rod_EDM, mtx_EDM):
        
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
    
    # Execute -----------------------------------------------------------------
    
    # Initialize
    print(f"\n{stack_path.stem}")
    print( "=========================")
    
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
    print( "  -----------------------")
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
    
#%% Function : register_stacks ------------------------------------------------

def register_stacks(stack_data):
    
    # Nested functions --------------------------------------------------------

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
    
    # Execute -----------------------------------------------------------------
    
    transform_matrices = []
    stack_reg_data = [stack_data[0]["stack_rsize"]]
    for i in range(1, len(stack_data)):
        transform_matrix = get_transform_matrix(stack_data[0], stack_data[i])
        transform_matrices.append(transform_matrix)
        stack_reg_data.append(
            affine_transform(stack_data[i]["stack_rsize"], transform_matrix)
            )
    
    min_z = np.min([stack.shape[0] for stack in stack_reg_data])
    min_y = np.min([stack.shape[1] for stack in stack_reg_data]) 
    min_x = np.min([stack.shape[2] for stack in stack_reg_data])

    for i in range(len(stack_reg_data)):
        stack_reg_data[i] = stack_reg_data[i][:min_z, :min_y, :min_x]
    stack_reg = np.stack(stack_reg_data)    
    
    return stack_reg, transform_matrices