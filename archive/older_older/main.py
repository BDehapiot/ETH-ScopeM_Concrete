#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
from functions import process_stacks, register_stacks

#%% Parameters ----------------------------------------------------------------

rsize_factor = 8 # Image size reduction factor
mtx_thresh_coeff = 1.0 # adjust matrix threshold
rod_thresh_coeff = 1.0 # adjust rod threshold

#%% Paths ---------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

# List stacks 
stack_paths = []
for folder in Path(data_path).iterdir():
    if folder.is_dir():
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                stack_paths.append(subfolder)
                
#%% Execute -------------------------------------------------------------------

# Process stack
stack_data = []
for stack_path in stack_paths:
    if exp_name in stack_path.name: 
        process_stacks(
            stack_path, 
            stack_data,
            rsize_factor,
            mtx_thresh_coeff,
            rod_thresh_coeff,
            )

# Register stack        
stack_rsize_reg = register_stacks(stack_data)

# Save 
io.imsave(
    Path(data_path, f"{exp_name}_registered.tif"),
    stack_rsize_reg.astype("float32"),
    check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )

# import pickle
# with open(Path(data_path, f"{exp_name}_stack_data.pkl"), 'wb') as f:
#     pickle.dump(stack_data, f)

#%% Save ----------------------------------------------------------------------

# from skimage import io

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
    
#     # -------------------------------------------------------------------------
    
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_obj_mask_3D.tif"),
#         data["obj_mask_3D"].astype("uint8") * 255, check_contrast=False,
#         )
#     io.imsave(
#         Path(data_path, f"{data['stack_path'].stem}_obj_labels_3D.tif"),
#         data["obj_labels_3D"].astype("uint16"), check_contrast=False,
#         )

