#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Skimage
from skimage.morphology import remove_small_objects

# Scipy
from scipy.linalg import lstsq
from scipy.ndimage import affine_transform

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
experiments = [
    # "D1_ICONX_DoS",
    "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H1_ICONX_DoS",
    # "H9_ICONX_DoS",
    ]

#%% Function(s) get transform matrix ------------------------------------------

def get_transform_matrix(path_ref, path_reg):

    name_ref = path_ref.stem  
    name_reg = path_reg.stem
    experiment_path = path_ref.parent
    print(
        f"(get_transform_matrix)\n"
        f"ref : {name_ref}\n"
        f"reg : {name_reg}"
        )
    
    # Read --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Read : ", end='')
    
    # Data
    obj_labels_ref = io.imread(experiment_path / (name_ref + "_labels.tif"))
    obj_labels_reg = io.imread(experiment_path / (name_reg + "_labels.tif"))
    
    # Metadata
    metadata_ref_path = experiment_path / (name_ref + "_metadata.pkl") 
    metadata_reg_path = experiment_path / (name_reg + "_metadata.pkl") 
    with open(metadata_ref_path, 'rb') as file:
        metadata_ref = pickle.load(file)
    with open(metadata_reg_path, 'rb') as file:
        metadata_reg = pickle.load(file)
    obj_data_ref = metadata_ref["obj_data"]
    obj_data_reg = metadata_reg["obj_data"]
    rf = np.sqrt(
        np.sum(metadata_ref["rod_mask"]) / np.sum(metadata_reg["rod_mask"]))
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s ({rf:.3f} rescale factor)") 
    
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
    
    def filter_objects(
            obj_data, obj_labels, min_size=1e5 * (1 /df) ** 3): # Parameter (1.0e5)
        valid_obj_data = []
        obj_mask = obj_labels > 0
        for i, data in enumerate(obj_data):
            if data["area"] > min_size:
                valid_obj_data.append(data)
        obj_mask = remove_small_objects(obj_mask, min_size=min_size)
        obj_labels[obj_mask == 0] = 0
        return valid_obj_data, obj_labels
    
    # Filter objects
    obj_data_ref, obj_labels_ref = filter_objects(obj_data_ref, obj_labels_ref)
    obj_data_reg, obj_labels_reg = filter_objects(obj_data_reg, obj_labels_reg)
    
    # Extract properties    
    props_ref = np.stack([(
        data["area"], 
        data["obj_dist"], 
        data["mtx_dist"], 
        data["solidity"]
        ) for data in obj_data_ref
        ])
    props_reg = np.stack([(
        data["area"] * (rf ** 3), 
        data["obj_dist"] * rf, 
        data["mtx_dist"] * rf,
        data["solidity"]
        ) for data in obj_data_reg
        ])
       
    # Identify pairs
    pairs = []
    for idx_ref, prop_ref in enumerate(props_ref):
        tests = []
        for prop_reg in props_reg:
            tests.append(np.mean(np.abs(1 - prop_ref / prop_reg)))
        idx_reg = np.argmin(tests)
        pairs.append((
            idx_ref, idx_reg,
            obj_data_ref[idx_ref]["label"],
            obj_data_reg[idx_reg]["label"],
            tests[idx_reg]))
    pairs = np.stack(pairs)
    
    # Keep best matching pairs
    for unique in np.unique(pairs[:, 1]):
        idxs = np.where(pairs[:, 1] == unique)[0]
        if len(idxs) > 1:
            idx = np.argmin(pairs[idxs, 4])
            idxs = np.delete(idxs, idx)
            pairs = np.delete(pairs, idxs, axis=0)
            
    # Detect false pairs
    coords_ref, coords_reg = [], []
    for pair in pairs:
        coords_ref.append(obj_data_ref[int(pair[0])]["centroid"])
        coords_reg.append(obj_data_reg[int(pair[1])]["centroid"])
    coords_ref = np.stack(coords_ref)
    coords_reg = np.stack(coords_reg)
    dist_ref = get_distances(coords_ref) 
    dist_reg = get_distances(coords_reg) * rf
    pairs = np.column_stack((
        pairs, np.median(np.abs(dist_ref - dist_reg), axis=0)))
    pcLow1 = np.percentile(pairs[:, 4], 30) # Parameter (30)
    pcLow2 = np.percentile(pairs[:, 5], 30) # Parameter (30)
    idxs = (pairs[:, 4] < pcLow1) & (pairs[:, 5] < pcLow2)
    pairs = np.column_stack((pairs, idxs))
    coords_ref = coords_ref[idxs, :]
    coords_reg = coords_reg[idxs, :]
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s ({coords_ref.shape[0]} objects)") 
    
    # Register ----------------------------------------------------------------
       
    t0 = time.time()
    print(" - get_transform_matrix : ", end='')
    
    def _get_transform_matrix(coords_ref, coords_reg):
       
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
    transform_matrix = _get_transform_matrix(coords_ref, coords_reg)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

    # Save --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Transformation matrix
    transform_matrix_path = experiment_path / (name_reg + "_transform_matrix.pkl")
    with open(transform_matrix_path, 'wb') as file:
        pickle.dump(transform_matrix, file)
        
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Validation --------------------------------------------------------------
    
    # # Check valid pairs
    # valid_pairs = pairs[idxs, :]
    # valid_labels_ref = valid_pairs[:, 2].astype(int)
    # valid_labels_reg = valid_pairs[:, 3].astype(int)
    # obj_mask_ref = np.isin(obj_labels_ref, valid_labels_ref)
    # obj_mask_reg = np.isin(obj_labels_reg, valid_labels_reg)
    # obj_labels_ref[obj_mask_ref == 0] = 0
    # obj_labels_reg[obj_mask_reg == 0] = 0
    # for l in range(len(valid_labels_ref)):
    #     label_ref = valid_labels_ref[l]
    #     label_reg = valid_labels_reg[l]
    #     obj_labels_reg[obj_labels_reg == label_reg] = label_ref 
    
    # # Intermediate plot #1
    # fig, axs = plt.subplots(4, 1, figsize=(4, 12))
    # titles = ["area", "obj_dist", "mtx_dist", "solidity"]
    # for i in range(4):
    #     axs[i].boxplot([props_ref[:, i], props_reg[:, i]], showfliers=False)
    #     axs[i].set_title(titles[i])
    #     axs[i].set_xticklabels(['ref', 'reg'])
    #     median_ref = np.median(props_ref[:, i])
    #     median_reg = np.median(props_reg[:, i])
    #     axs[i].text(1.15, median_ref, f'{median_ref:.2f}', fontsize=10)
    #     axs[i].text(2.15, median_reg, f'{median_reg:.2f}', fontsize=10)
    # plt.show()
    
    # # Intermediate plot #2
    # fig, axs = plt.subplots(3, 1, figsize=(4, 8))
    # axs[0].hist(pairs[:, 2], bins=50)
    # axs[0].set_title("scores1")
    # axs[1].hist(pairs[:, 3], bins=50)
    # axs[1].set_title("scores2")
    # axs[2].scatter(pairs[:, 2], pairs[:, 3], s=5)
    # axs[2].set_title("scores2 = f(scores1)")
    # plt.tight_layout(pad=1)
    # plt.show()

#%% Function(s) apply transform matrix ----------------------------------------

def apply_transform_matrix(path_reg):

    name_reg = path_reg.stem
    experiment_path = path_reg.parent
    print(
        f"(apply_transform_matrix)\n"
        f"reg : {name_reg}"
        )    

    # Read --------------------------------------------------------------------
    
    

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
        for i in range(1, len(paths)):
            get_transform_matrix(paths[0], paths[i])
            apply_transform_matrix(paths[i])