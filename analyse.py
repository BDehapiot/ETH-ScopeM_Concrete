#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

from functions import shift_stack

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
    # "H5_ICONX_corrosion"
    # "H9_ICONX_DoS",
    ]

#%% Function(s) analyse -------------------------------------------------------

def analyse(paths, experiment_path):
    
    global \
        metadata_list, obj_data_list, df
    
    # Read --------------------------------------------------------------------
       
    metadata_list, obj_data_list = [], []
    for path in paths:

        # Metadata and obj_data
        metadata_path = path.with_name(path.stem + "_metadata.pkl")
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
            metadata_list.append(metadata)  
            obj_data_list.append(metadata["obj_data"])  
            
    # Path
    experiment_reg_path = experiment_path / "registered"
    experiment_out_path = experiment_path / "outputs"
            
    # Plot #1 (barplot) -------------------------------------------------------  

    # Parameters
    binMin, binMax, binSize = 0, 110, 10
    valMin = 3

    # Initialize
    binRange = np.arange(binMin, binMax + 1, binSize)
    plt.figure(figsize=(10, 3 * len(paths)))

    for i, path in enumerate(paths):
        
        # Get time (from name)
        tIdx = path.stem.find("Time")
        t = int(path.stem[tIdx + 4])
        
        # Extract data
        ratio = np.stack([data["ratio"] for data in obj_data_list[i]])
        mtx_dist = np.stack([data["mtx_dist"] for data in obj_data_list[i]])
        category = np.stack([data["category"] for data in obj_data_list[i]])
        
        # Select inner objects
        valid_idx = category == 0
        ratio = ratio[valid_idx]
        mtx_dist = mtx_dist[valid_idx]
        
        # Compute avg and std ratios
        nRatios, avgRatios, stdRatios, labels = [], [], [], []
        for b in range(1, len(binRange)):
            b0, b1 = binRange[b - 1], binRange[b]
            ratios = ratio[(mtx_dist >= b0) & (mtx_dist < b1)]
            nRatios.append(len(ratios)) 
            if len(ratios) < valMin:
                avgRatio = 0
                stdRatio = 0
            else:
                avgRatio = np.nanmean(ratios)
                stdRatio = np.nanstd(ratios)
            avgRatios.append(avgRatio)
            stdRatios.append(stdRatio)
            labels.append(f"{b0}-{b1}\n n={len(ratios)}")
            
            # if b == 1:
            #     print(f"{path.stem}")
            # print(
            #     f"{b0:03d}-{b1:03d} "
            #     f"/ {avgRatio:.3f}-{stdRatio:.3f} "
            #     f"/ {len(ratios)}"
            #     )
        
        # plot
        plt.subplot(len(paths), 1, i + 1)
        plt.axhline(y=1, color='k', linewidth=1, linestyle='--')
        bars = plt.bar(labels, avgRatios, width=0.75)
        plt.errorbar(
            labels, avgRatios, fmt="o", color="k", 
            capsize=10, yerr=stdRatios, linewidth=0.5
            )
        plt.ylim(0, 1.5)
        plt.title(path.stem)
        plt.ylabel("fill ratio")
        plt.xlabel("distance (pixel)")   
        
        # annotate
        for bar, n, avg in zip(bars, nRatios, avgRatios):
            if n < valMin:
                plt.text(
                    bar.get_x() + bar.get_width() / 2, 0.05,
                    'NaN', ha='center', va='bottom', color='black', 
                    fontsize=12, rotation=0,
                    )

    # Save & show
    plt.tight_layout(pad=2)
    plt.savefig(experiment_out_path / (experiment + "_plot.jpg"), format='jpg')
    plt.show()
            
    # Save --------------------------------------------------------------------

    for i, path in enumerate(paths):
        df_path = experiment_out_path / (path.stem + "_obj_data.csv")
        df = pd.DataFrame(obj_data_list[i])
        df.to_csv(df_path, index=False, float_format='%.3f')

    return

#%% Execute -------------------------------------------------------------------

outputs = []
if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_out_path = experiment_path / "outputs"
        experiment_out_path.mkdir(parents=True, exist_ok=True)
        paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
        analyse(paths, experiment_path)

#%%



#%%
   
# # Plot #1 (heatmap) -----------------------------------------------------------  

# i = 7
# path = paths[i]
# # for i, path in enumerate(paths):

# # -----------------------------------------------------------------------------
    
# # Open data
# experiment_reg_path = experiment_path / "registered"
# transform_matrix_path = experiment_reg_path / (path.stem + "_transform_matrix.pkl")
# with open(transform_matrix_path, 'rb') as file:
#     transform_matrix = pickle.load(file)

# # 
# centers = metadata_list[0]["centers"]
# ratio = np.stack([data["ratio"] for data in obj_data_list[i]])
# category = np.stack([data["category"] for data in obj_data_list[i]])
# centroids = np.column_stack((
#     np.stack([data["ctrd_z"] for data in obj_data_list[i]]),
#     np.stack([data["ctrd_y"] for data in obj_data_list[i]]),
#     np.stack([data["ctrd_x"] for data in obj_data_list[i]]),
#     np.ones(ratio.shape[0])
#     ))

# #
# centroids_reg = (centroids @ transform_matrix.T)
# ctrds_z = centroids_reg[:, 0]
# ctrds_y = centroids_reg[:, 1]
# ctrds_x = centroids_reg[:, 2]

# #
# y_corr, x_corr = [], []
# for z, y, x in zip(ctrds_z, ctrds_y, ctrds_x):
#     z = int(z)
#     if z < 0: z = 0
#     if z > len(centers): z = len(centers)
#     y_corr.append(y + centers[int(z)][0])
#     x_corr.append(x + centers[int(z)][1])
# ctrds_y += np.stack(y_corr) 
# ctrds_x += np.stack(x_corr) 

# # Select inner objects
# valid_idx = category == 0
# ratio = ratio[valid_idx]
# ctrds_y = ctrds_y[valid_idx]
# ctrds_x = ctrds_x[valid_idx]

# plt.figure(figsize=(10, 10))
# plt.scatter(ctrds_y, ctrds_x, c=ratio)
    
#%%

# import napari
# viewer = napari.Viewer()
# viewer.add_image(mtx_mask)