#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

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
            
    # Save --------------------------------------------------------------------

    for i, path in enumerate(paths):
        df_path = experiment_out_path / (path.stem + "_outputs.csv")
        df = pd.DataFrame(obj_data_list[i])
        df.to_csv(df_path, index=False)

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
        
        
        # if not test_path.is_file() or overwrite:
        #     for i in range(len(paths)):
        #         outputs.append(register(paths[0], paths[i]))
        #     register_postprocess(outputs, crop=False)

