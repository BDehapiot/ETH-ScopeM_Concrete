#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
experiments = [
    "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H1_ICONX_DoS",
    # "H9_ICONX_DoS",
    ]


#%% Function(s) analyse -------------------------------------------------------

def analyse(paths, experiment_out_path):
    
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
            
    # Analyse -----------------------------------------------------------------   
            
    # Save --------------------------------------------------------------------

    for i, path in enumerate(paths):
        df_path = experiment_out_path / (path.stem + "_outputs.csv")
        df = pd.DataFrame(obj_data_list[i])
        df.to_csv(df_path, index=False, float_format='%.3f')

    return


#%% Execute -------------------------------------------------------------------

outputs = []
if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_out_path = data_path / experiment / "OUT"
        experiment_out_path.mkdir(parents=True, exist_ok=True)
        paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
        analyse(paths, experiment_out_path)

#%%
   
nBin = 10

nT = len(obj_data_list)
plt.figure(figsize=(10, 5 * nT))

for t in range(nT):
    
    # Extract data
    ratio = np.stack([data["ratio"] for data in obj_data_list[t]])
    mtx_dist = np.stack([data["mtx_dist"] for data in obj_data_list[t]])
    category = np.stack([data["category"] for data in obj_data_list[t]])
    
    # Select inner voids
    valid_idx = category == 0
    ratio = ratio[valid_idx]
    mtx_dist = mtx_dist[valid_idx]
    
    # Compute avg and std ratios
    avgRatios, stdRatios, labels = [], [], []
    maxBin = np.max(mtx_dist)
    rangeBin = np.linspace(0, maxBin, num=nBin + 1)
    for b in range(1, nBin + 1):
        b0, b1 = rangeBin[b - 1], rangeBin[b]
        tmp = ratio[(mtx_dist >= b0) & (mtx_dist < b1)]
        avgRatios.append(np.mean(tmp))
        stdRatios.append(np.std(tmp))
        labels.append(f"{b0:.2f}-{b1:.2f}")
        
        
    plt.subplot(nT, 1, t + 1)
    plt.bar(labels, avgRatios, width=0.75)
    plt.errorbar(labels, avgRatios, fmt="o", yerr=stdRatios)
    plt.ylim(0, 1.5)
    
    # plt.errorbar(labels, avgRatios, yerr=stdRatios, fmt='o', capsize=5, label='Mean with SD')
