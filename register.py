#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Scipy
from scipy.linalg import lstsq
from scipy.ndimage import affine_transform

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
overwrite = False
df = 4 # downscale factor

#%% Function(s) ---------------------------------------------------------------

def register_stacks(path_ref, path_reg):
    
    global \
        metadata_ref, metadata_reg,\
        stack_ref, stack_reg,\
        obj_labels_3D_ref, obj_labels_3D_reg,\
        obj_props_ref, obj_props_reg, props_ref, props_reg, rscale_factor,\
        size_ref, size_reg,\
        tests, pairs,\
        labels_3D_ref, labels_3D_reg,\
        coords_ref, coords_reg,\
        dist_ref, dist_reg,\
        scores,\
        transformed_stack
    
    # Initialize --------------------------------------------------------------

    name_ref = path_ref.stem  
    name_reg = path_reg.stem
    experiment_path = path_ref.parent
    
    # Read --------------------------------------------------------------------
    
    print(
        f"(register)\n"
        f"ref : {name_ref}\n"
        f"reg : {name_reg}"
        )
    t0 = time.time()
    print(" - Read : ", end='')
    
    # Data
    stack_ref = io.imread(experiment_path / (name_ref + ".tif"))
    stack_reg = io.imread(experiment_path / (name_reg + ".tif"))
    obj_labels_3D_ref = io.imread(experiment_path / (name_ref + "_labels.tif"))
    obj_labels_3D_reg = io.imread(experiment_path / (name_reg + "_labels.tif"))
    
    # Metadata
    metadata_ref_path = experiment_path / (name_ref + "_metadata.pkl") 
    metadata_reg_path = experiment_path / (name_reg + "_metadata.pkl") 
    with open(metadata_ref_path, 'rb') as file:
        metadata_ref = pickle.load(file)
    with open(metadata_reg_path, 'rb') as file:
        metadata_reg = pickle.load(file)
    obj_props_ref = metadata_ref["objects"]
    obj_props_reg = metadata_reg["objects"]
    rscale_factor = np.sqrt(
        np.sum(metadata_ref["rod_mask"]) / np.sum(metadata_reg["rod_mask"]))
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Match -------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Match : ", end='')
    
    def get_distances(coords):
        num_points = len(coords)
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
        return distance_matrix
    
    # Extract properties
    props_ref = np.stack([
        (data["area"], data["obj_dist"], data["mtx_dist"], data["solidity"]) 
        for data in obj_props_ref
        ])
    props_reg = np.stack([
        (data["area"], data["obj_dist"], data["mtx_dist"], data["solidity"]) 
        for data in obj_props_reg
        ])
    
    # Normalize (area, obj_dist, mtx_dist)
    for col in [0, 1, 2]:
        props_reg[:, col] *= rscale_factor
    
    # Test object pairs
    tests = []
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
        if score < 0.2: # parameter
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
        coords_ref.append(obj_props_ref[idx_ref]["centroid"])
        coords_reg.append(obj_props_reg[idx_reg]["centroid"])
        labels_3D_ref[obj_labels_3D_ref == idx_ref + 1] = idx_ref + 1
        labels_3D_reg[obj_labels_3D_reg == idx_reg + 1] = idx_ref + 1
    coords_ref = np.stack(coords_ref)
    coords_reg = np.stack(coords_reg)
    
    # Remove false pairs
    dist_ref = get_distances(coords_ref) 
    dist_reg = get_distances(coords_reg) * rscale_factor
    scores = np.median(np.abs(dist_ref - dist_reg), axis=0)
    pairs = np.column_stack((pairs, scores))
    outliers = np.where(scores > 30)[0] # parameter
    coords_ref = np.delete(coords_ref, outliers, axis=0)
    coords_reg = np.delete(coords_reg, outliers, axis=0)
    for outlier in outliers:
        labels_3D_ref[labels_3D_ref == pairs[outlier, 0] + 1] = 0
        labels_3D_reg[labels_3D_reg == pairs[outlier, 0] + 1] = 0
       
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s ({coords_ref.shape[0]} objects)") 
       
    # Register ----------------------------------------------------------------
       
    t0 = time.time()
    print(" - Register : ", end='')
    
    def get_transform_matrix(coords_ref, coords_reg):
       
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
    
    # Compute transformation matrix
    transform_matrix = get_transform_matrix(coords_ref, coords_reg)
    
    # Apply transformation
    transformed_stack = affine_transform(stack_reg, transform_matrix)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment        
        paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
                
        for i in range(1, len(paths)):
            if i == 3:
                register_stacks(paths[0], paths[i])
        
        # stack_reg = Parallel(n_jobs=-1)(
        #         delayed(register_stacks)(paths[0], paths[i]) 
        #         for i in range(1, len(paths))
        #         )


#%% Display -------------------------------------------------------------------

import napari

# viewer = napari.Viewer()
# viewer.add_labels(obj_labels_3D_ref)
# viewer.add_labels(obj_labels_3D_reg)

viewer = napari.Viewer()
viewer.add_labels(labels_3D_ref)
viewer.add_labels(labels_3D_reg)

viewer = napari.Viewer()
viewer.add_image(stack_ref)
viewer.add_image(transformed_stack)