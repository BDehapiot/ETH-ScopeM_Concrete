#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Scipy
from scipy.linalg import lstsq
from scipy.ndimage import affine_transform

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
# experiment = "D1_ICONX_DoS"
experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
# experiment = "H9_ICONX_DoS"

#%%

path_ref = Path(data_path, experiment, f"{experiment}_Time0_crop_df{df}")
path_reg = Path(data_path, experiment, f"{experiment}_Time3_crop_df{df}")

#%% Function(s) ---------------------------------------------------------------

# Initialize ------------------------------------------------------------------

name_ref = path_ref.stem  
name_reg = path_reg.stem
experiment_path = path_ref.parent

# Read ------------------------------------------------------------------------

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

# Match -----------------------------------------------------------------------

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
    (data["area"] , data["obj_dist"], data["mtx_dist"], data["solidity"]) 
    for data in obj_data_ref
    ])
props_reg = np.stack([
    (data["area"] * (rf ** 3), data["obj_dist"] * rf, data["mtx_dist"] * rf, data["solidity"])
    for data in obj_data_reg
    ])

# Intermediate plot #1
fig, axs = plt.subplots(4, 1, figsize=(4, 12))
titles = ["area", "obj_dist", "mtx_dist", "solidity"]
for i in range(4):
    axs[i].boxplot([props_ref[:, i], props_reg[:, i]], showfliers=False)
    axs[i].set_title(titles[i])
    axs[i].set_xticklabels(['ref', 'reg'])
    median_ref = np.median(props_ref[:, i])
    median_reg = np.median(props_reg[:, i])
    axs[i].text(1.15, median_ref, f'{median_ref:.2f}', fontsize=10)
    axs[i].text(2.15, median_reg, f'{median_reg:.2f}', fontsize=10)
plt.show()
    
# Identify pairs
pairs = []
for idx_ref, prop_ref in enumerate(props_ref):
    tests = []
    for prop_reg in props_reg:
        tests.append(np.mean(np.abs(1 - prop_ref / prop_reg)))
    idx_reg = np.argmin(tests)
    pairs.append((idx_ref, idx_reg, tests[idx_reg]))
pairs = np.stack(pairs)

# Keep best matching pairs
for unique in np.unique(pairs[:,1]):
    idxs = np.where(pairs[:,1] == unique)[0]
    if len(idxs) > 1:
        idx = np.argmin(pairs[idxs, 2])
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
dist_scores = np.median(np.abs(dist_ref - dist_reg), axis=0)
pairs = np.column_stack((pairs, dist_scores))

# Intermediate plot #2
plt.hist(pairs[:, 2], bins=50)
plt.show()
plt.hist(pairs[:, 3], bins=50)
plt.show()
plt.scatter(pairs[:, 2], pairs[:, 3])
plt.show()
# plt.hist(pairs[:, 2], bins=50)
# plt.hist(pairs[:, 3], bins=50)

# # Remove false pairs
# dist_ref = get_distances(coords_ref) 
# dist_reg = get_distances(coords_reg) * rscale_factor
# scores2 = np.median(np.abs(dist_ref - dist_reg), axis=0)
# pairs = np.column_stack((pairs, scores2))
# outliers = np.where(scores > 60)[0] # parameter (30)
# coords_ref = np.delete(coords_ref, outliers, axis=0)
# coords_reg = np.delete(coords_reg, outliers, axis=0)
# for outlier in outliers:
#     labels_ref[labels_ref == pairs[outlier, 0] + 1] = 0
#     labels_reg[labels_reg == pairs[outlier, 0] + 1] = 0

# # Isolate pairs coordinates
# coords_ref, coords_reg = [], []
# labels_ref = np.zeros_like(obj_labels_ref)
# labels_reg = np.zeros_like(obj_labels_reg)
# for pair in pairs:
#     idx_ref, idx_reg = int(pair[0]), int(pair[1])
#     coords_ref.append(obj_data_ref[idx_ref]["centroid"])
#     coords_reg.append(obj_data_reg[idx_reg]["centroid"])
#     labels_ref[obj_labels_ref == idx_ref + 1] = idx_ref + 1
#     labels_reg[obj_labels_reg == idx_reg + 1] = idx_ref + 1
# coords_ref = np.stack(coords_ref)
# coords_reg = np.stack(coords_reg)

# # Intermediate plot #2
# plt.hist(pairs[:, 2], bins=50)

t1 = time.time()
print(f"{(t1-t0):<5.2f}s ({coords_ref.shape[0]} objects)") 

#%% Function(s) ---------------------------------------------------------------


# def register_stacks(path_ref, path_reg):
    
#     global \
#         metadata_ref, metadata_reg,\
#         stack_ref, stack_reg,\
#         obj_labels_ref, obj_labels_reg,\
#         obj_props_ref, obj_props_reg, props_ref, props_reg, rscale_factor,\
#         size_ref, size_reg,\
#         tests, pairs,\
#         labels_ref, labels_reg,\
#         coords_ref, coords_reg,\
#         dist_ref, dist_reg,\
#         scores,\
#         transformed_stack
    
#     # Initialize --------------------------------------------------------------

#     name_ref = path_ref.stem  
#     name_reg = path_reg.stem
#     experiment_path = path_ref.parent
    
#     # Read --------------------------------------------------------------------
    
#     print(
#         f"(register)\n"
#         f"ref : {name_ref}\n"
#         f"reg : {name_reg}"
#         )
#     t0 = time.time()
#     print(" - Read : ", end='')
    
#     # Data
#     stack_ref = io.imread(experiment_path / (name_ref + ".tif"))
#     stack_reg = io.imread(experiment_path / (name_reg + ".tif"))
#     obj_labels_ref = io.imread(experiment_path / (name_ref + "_labels.tif"))
#     obj_labels_reg = io.imread(experiment_path / (name_reg + "_labels.tif"))
    
#     # Metadata
#     metadata_ref_path = experiment_path / (name_ref + "_metadata.pkl") 
#     metadata_reg_path = experiment_path / (name_reg + "_metadata.pkl") 
#     with open(metadata_ref_path, 'rb') as file:
#         metadata_ref = pickle.load(file)
#     with open(metadata_reg_path, 'rb') as file:
#         metadata_reg = pickle.load(file)
#     obj_data_ref = metadata_ref["obj_data"]
#     obj_data_reg = metadata_reg["obj_data"]
#     rscale_factor = np.sqrt(
#         np.sum(metadata_ref["rod_mask"]) / np.sum(metadata_reg["rod_mask"]))
    
#     t1 = time.time()
#     print(f"{(t1-t0):<5.2f}s")
    
#     # Match -------------------------------------------------------------------
    
#     t0 = time.time()
#     print(" - Match : ", end='')
    
#     def get_distances(coords):
#         num_points = len(coords)
#         distance_matrix = np.zeros((num_points, num_points))
#         for i in range(num_points):
#             for j in range(num_points):
#                 if i != j:
#                     distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
#         return distance_matrix
    
#     # Extract properties    
#     props_ref = np.stack([
#         # (data["area"] , data["mtx_dist"], data["solidity"]) 
#         (data["area"] , data["obj_dist"], data["mtx_dist"], data["solidity"]) 
#         for data in obj_data_ref
#         ])
#     props_reg = np.stack([
#         # (data["area"] * rscale_factor ** 3, data["mtx_dist"] * rscale_factor, data["solidity"]) 
#         (data["area"] * rscale_factor ** 3, data["obj_dist"] * rscale_factor, data["mtx_dist"] * rscale_factor, data["solidity"])
#         for data in obj_data_reg
#         ])
    
#     # # Normalize (area, obj_dist, mtx_dist)
#     # for col in [0, 1, 2]:
#     #     props_reg[:, col] *= rscale_factor
    
#     # Test object pairs
#     tests = []
#     for prop_ref in props_ref:
#         test = []
#         for prop_reg in props_reg:
#             ratio = prop_ref / prop_reg
#             ratio_avg = np.abs(1 - np.mean(ratio))
#             ratio_std = np.std(ratio)
#             test.append((ratio_avg + ratio_std) / 2)
#         tests.append(test)
        
#     # Identify matching pairs
#     pairs, scores1 = []
#     for idx_ref, test in enumerate(tests):
#         idx_reg = np.argmin(test)
#         score1 = np.min(test)
#         if score1 < 0.2: # parameter (0.2)
#             pairs.append((idx_ref, idx_reg, score1))
#         scores1.append(scores1) # for checking
#     pairs = np.stack(pairs)
    
#     # Keep best match only
#     for unique in np.unique(pairs[:,1]):
#         idxs = np.where(pairs[:,1] == unique)[0]
#         if len(idxs) > 1:
#             idx = np.argmin(pairs[idxs, 2])
#             idxs = np.delete(idxs, idx)
#             pairs = np.delete(pairs, idxs, axis=0)
            
#     # Isolate pairs coordinates
#     coords_ref, coords_reg = [], []
#     labels_ref = np.zeros_like(obj_labels_ref)
#     labels_reg = np.zeros_like(obj_labels_reg)
#     for pair in pairs:
#         idx_ref, idx_reg = int(pair[0]), int(pair[1])
#         coords_ref.append(obj_data_ref[idx_ref]["centroid"])
#         coords_reg.append(obj_data_reg[idx_reg]["centroid"])
#         labels_ref[obj_labels_ref == idx_ref + 1] = idx_ref + 1
#         labels_reg[obj_labels_reg == idx_reg + 1] = idx_ref + 1
#     coords_ref = np.stack(coords_ref)
#     coords_reg = np.stack(coords_reg)
    
#     # Remove false pairs
#     dist_ref = get_distances(coords_ref) 
#     dist_reg = get_distances(coords_reg) * rscale_factor
#     scores2 = np.median(np.abs(dist_ref - dist_reg), axis=0)
#     pairs = np.column_stack((pairs, scores2))
#     outliers = np.where(scores > 60)[0] # parameter (30)
#     coords_ref = np.delete(coords_ref, outliers, axis=0)
#     coords_reg = np.delete(coords_reg, outliers, axis=0)
#     for outlier in outliers:
#         labels_ref[labels_ref == pairs[outlier, 0] + 1] = 0
#         labels_reg[labels_reg == pairs[outlier, 0] + 1] = 0
    
#     # Plots
#     plt.hist(scores1, bins=20)
#     plt.hist(scores2, bins=20)

#     t1 = time.time()
#     print(f"{(t1-t0):<5.2f}s ({coords_ref.shape[0]} objects)") 
       
#     # Register ----------------------------------------------------------------
       
#     t0 = time.time()
#     print(" - Register : ", end='')
    
#     def get_transform_matrix(coords_ref, coords_reg):
       
#         if coords_ref.shape[0] < coords_ref.shape[1]:
#             coords_ref = coords_ref.T
#             coords_reg = coords_reg.T
#         (n, dim) = coords_ref.shape
        
#         # Compute least squares
#         p, res, rnk, s = lstsq(
#             np.hstack((coords_ref, np.ones([n, 1]))), coords_reg)
#         # Get translations & transform matrix
#         t, T = p[-1].T, p[:-1].T
        
#         # Merge translations and transform matrix
#         transform_matrix = np.eye(4)
#         transform_matrix[:3, :3] = T
#         transform_matrix[:3, 3] = t
        
#         return transform_matrix
    
#     # Compute transformation matrix
#     transform_matrix = get_transform_matrix(coords_ref, coords_reg)
    
#     # Apply transformation
#     transformed_stack = affine_transform(stack_reg, transform_matrix)
    
#     t1 = time.time()
#     print(f"{(t1-t0):<5.2f}s")
    
#%% Execute -------------------------------------------------------------------

# if __name__ == "__main__":
#     for experiment in experiments:
#         experiment_path = data_path / experiment        
#         paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
                
#         for i in range(1, len(paths)):
#             if i == 1:
#                 register_stacks(paths[0], paths[i])
        
#         # stack_reg = Parallel(n_jobs=-1)(
#         #         delayed(register_stacks)(paths[0], paths[i]) 
#         #         for i in range(1, len(paths))
#         #         )


#%% Display -------------------------------------------------------------------

# import napari

# # viewer = napari.Viewer()
# # viewer.add_labels(obj_labels_ref)
# # viewer.add_labels(obj_labels_reg)

# # viewer = napari.Viewer()
# # viewer.add_labels(labels_ref)
# # viewer.add_labels(labels_reg)

# viewer = napari.Viewer()
# viewer.add_image(stack_ref)
# viewer.add_image(transformed_stack)