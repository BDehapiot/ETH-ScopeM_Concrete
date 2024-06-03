#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.transform import downscale_local_mean

# Functions
from functions import crop, preprocess, predict, segment, objects

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
raw_path = Path(data_path, "0-raw")
model_path = Path.cwd() / f"model-weights_void_p0256_d{df}.h5"
experiments = [
    # "D1_ICONX_DoS",
    # "D11_ICONX_DoS",
    # "D12_ICONX_corrosion", 
    # "H1_ICONX_DoS",
    "H9_ICONX_DoS",
    ]

#%% Function(s) ---------------------------------------------------------------

def process_stack(path, experiment_path, df):

    name = path.name    
    print(f"(process) {name}")

    # Read --------------------------------------------------------------------
        
    t0 = time.time()
    print(" - Read : ", end='')
    
    stack = []
    img_paths = list(path.glob("**/*.tif"))
    for img_path in img_paths:
        stack.append(io.imread(img_path))
    stack = np.stack(stack)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Crop --------------------------------------------------------------------
    
    t0 = time.time(); 
    print(" - Crop : ", end='')

    stack, crops = crop(stack, df)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")

    # Downscale ---------------------------------------------------------------

    t0 = time.time()
    print(" - Downscale : ", end='')
       
    stack = downscale_local_mean(stack, df)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
        
    # Preprocess --------------------------------------------------------------
    
    t0 = time.time()
    print(" - Preprocess : ", end='')
    
    centers, med_proj, mtx_mask, rod_mask, mtx_EDM, rod_EDM, stack_norm \
        = preprocess(stack)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Predict -----------------------------------------------------------------
    
    t0 = time.time()

    obj_probs = predict(stack_norm, model_path)
    
    t1 = time.time()
    print(f" - Predict : {(t1-t0):<5.2f}s") 
    
    # Segment -----------------------------------------------------------------
    
    t0 = time.time()
    print(" - Segment : ", end='')
    
    stack_norm_corr, air_mask_3D, liquid_mask_3D \
        = segment(stack_norm, obj_probs, mtx_EDM, centers, df)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    # Objects ---------------------------------------------------------------
    
    t0 = time.time()
    print(" - Objects : ", end='')
        
    obj_labels_3D, obj_data = objects(
            obj_probs, 
            mtx_mask, rod_mask, 
            air_mask_3D, liquid_mask_3D, 
            mtx_EDM, centers,
            df) 
                
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s") 
    
    #%% Save ------------------------------------------------------------------
    
    t0 = time.time()
    print(" - Save : ", end='')
    
    # Data
    io.imsave(
        experiment_path / (name + f"_crop_df{df}.tif"), 
        stack.astype("uint16"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_norm.tif"), 
        stack_norm.astype("float32"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_probs.tif"), 
        obj_probs.astype("float32"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_labels.tif"), 
        obj_labels_3D.astype("uint16"), check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_air_mask.tif"), 
        air_mask_3D.astype("uint8") * 255, check_contrast=False
        )
    io.imsave(
        experiment_path / (name + f"_crop_df{df}_liquid_mask.tif"), 
        liquid_mask_3D.astype("uint8") * 255, check_contrast=False
        )
        
    # Metadata
    metadata_path = experiment_path / (name + f"_crop_df{df}_metadata.pkl") 
    metadata = {
        "crops"    : crops,
        "centers"  : centers,
        "med_proj" : med_proj,
        "mtx_mask" : mtx_mask,
        "rod_mask" : rod_mask,
        "mtx_EDM"  : mtx_EDM,
        "rod_EDM"  : rod_EDM,
        "obj_data" : obj_data,
        }
    
    with open(metadata_path, 'wb') as file:
        pickle.dump(metadata, file)

    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s\n")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for experiment in experiments:
        experiment_path = data_path / experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        for path in raw_path.glob(f"*{experiment}*"):
            # if "Time0" in path.name:
            test_path = experiment_path / (path.name + f"_crop_df{df}.tif")
            if not test_path.is_file():
                process_stack(path, experiment_path, df)   
            elif overwrite:
                process_stack(path, experiment_path, df)