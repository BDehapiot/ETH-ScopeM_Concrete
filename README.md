# ETH-ScopeM_Concrete
Process and display concrete cylinders X-Ray tomographs

## Content
- Model:  
    - `model_extract.py` --> extract train. images 
    - `model_annotate.py` --> annotate train. images (Napari)
    - `model_train.py` --> train model

- Process:  
    - `process_main.py` --> mask, norm. & obj. data 
    
- Register
    - `register_main.py` --> spatial registration & histogram matching 

- Display  
    - `display_mask.py` --> show air/liquid masks (Napari)
    - `display_probs.py` --> show model predictions (Napari)

- Analyse
    - `analyse_main.py`

## Outputs

### `data` folder

- **...Time0_crop_df4.tif** --> crop + downscale raw images  
- **...Time0_crop_df4_norm.tif** --> crop + downscale + norm. images  
- **...Time0_crop_df4_probs.tif** --> obj. predictions (model)  
- **...Time0_crop_df4_air_mask.tif** --> labelled mask of obj.  
- **...Time0_crop_df4_liquid_mask.tif** --> mask of air-filled regions  
- **...Time0_crop_df4_labels.tif** --> mask of liquid-filled regions  
- **...Time0_crop_df4_metadata.pkl** --> processing metadata  

### `registered` folder

- **...crop_df4_reg.tif** --> registered crop + downscale raw images
- **...crop_df4_reg_norm.tif** --> registered crop + downscale + norm. images
- **...crop_df4_reg_probs.tif** --> registered obj. predictions
- **...crop_df4_reg_air_mask.tif** --> registered air masks
- **...crop_df4_reg_liquid_mask.tif** --> registered liquid masks
- **...crop_df4_transform_matrix.pkl** --> registration transformation matrices


### `outputs` folder

- `obj_data.csv`  

    - **label** : obj. identification label (see `labels.tif`)
    - **ctrd_z** : obj. z position 
    - **ctrd_y** : obj. y position
    - **ctrd_x** : obj. x position
    - **area** : obj. number of voxel (3D volume)
    - **air_area** : air number of voxel (3D volume)
    - **liquid_area** : liquid number of voxel (3D volume)
    - **ratio** : liquid / obj. volume
    - **solidity** : obj. volume / obj. convex hull volume
    - **obj_dist** : distance from neighbouring obj.
    - **mtx_dist** : distance from external matrix surface
    - **category** : 0 = inner obj., 1 = surface obj., 2 = rod obj.

## To fix
- issue #1
    - expl `D1_ICONX_DOS_reg_norm.tif` z=70 t=3
    - expl `D12_ICONX_Corrosion_reg_norm.tif` z=70 t=3
    
