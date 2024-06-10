#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:/local_Concrete/data")
experiments = [
    # "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    "H1_ICONX_DoS",
    # "H9_ICONX_DoS",
    ]

# Parameters
df = 4 # downscale factor

#%%

def analyse_stacks(paths):
    
    global \
        metadata_list, object_list,\
        objects
        
    # Read --------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Read : ", end='')
    
    metadata_list, object_list = [], []
    for path in paths:

        metadata_path = path.with_name(path.stem + "_metadata.pkl")
        with open(metadata_path, 'rb') as file:
            metadata = pickle.load(file)
            metadata_list.append(metadata)  
            object_list.append(metadata["objects"])  
            
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Extract -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Extract : ", end='')
    
    nT = len(object_list)
    plt.figure(figsize=(10, 5 * nT))
    for t, objects in enumerate(object_list):
        plt.subplot(nT, 1, t + 1)
        category = np.stack([obj["category"] for obj in objects])
        mtx_dist = np.stack([obj["mtx_dist"] for obj in objects])
        air_area = np.stack([obj["air_area"] for obj in objects])
        liquid_area = np.stack([obj["liquid_area"] for obj in objects])
        ratio = liquid_area / (air_area + liquid_area)
        plt.scatter(mtx_dist, ratio, c=category)
        
        
    
    # results = {
    #     "area" : [],
    #     "mtx_dist" : [],
    #     "air_area" : [],
    #     "liquid_area" : [],
    #     "ratio" : [],
    #     }
    
    # for metadata in metadata_list:
        
    #     # Exrtact data
    #     objects = metadata["objects"]
    #     area = np.stack([obj["area"] for obj in objects])
    #     mtx_dist = np.stack([obj["mtx_dist"] for obj in objects])
    #     air_area = np.stack([obj["air_area"] for obj in objects])
    #     liquid_area = np.stack([obj["liquid_area"] for obj in objects])
    #     ratio = liquid_area / (air_area + liquid_area)
        
    #     # Append results
    #     results["area"].append(area)
    #     results["mtx_dist"].append(mtx_dist)
    #     results["air_area"].append(air_area)
    #     results["liquid_area"].append(liquid_area)
    #     results["ratio"].append(ratio)
        
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Plots -------------------------------------------------------------------
        
    # nBin = 10
    # maxBin = np.max(np.concatenate(results["mtx_dist"]))
    # rangeBin = np.linspace(0, maxBin, num=nBin + 1)
    
    # plt.figure(figsize=(10, 5 * len(metadata_list)))
    
    # for t in range(len(metadata_list)):
        
    #     plt.subplot(len(metadata_list), 1, t + 1)
        
    #     mtx_dist = results["mtx_dist"][t]
    #     ratio = results["ratio"][t]
    #     plt.scatter(mtx_dist, ratio)
        
        # for b in range(1, len(rangeBin)):
        #     b0 = rangeBin[b - 1]
        #     b1 = rangeBin[b]
            
            
    
    
    # # print(maxBin)
    # print(rangeBin)
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment        
        paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))
        analyse_stacks(paths)