## Main procedure
### `process_main.py`
Main processing tasks executed on original image stacks

1) **Downscale** (reduce processing time)  
```bash
# For downscale factor (df) = 4  
original stack (z, y, x)  = 1948 x 1788 x 1788 = 13 Gb  
downscaled stack (z, y, x)  = 487 x 447 x 447 = 186 Mb
```
#

2) **Preprocess**
    - center images over Z axis
    - compute median projection (`med_proj`)
    - normalize images (divide by `med_proj` to get `norm`)
    - determine matrix and rod masks (`mtx_mask`, `rod_mask`)
    - compute distance maps  
        `mtx_EDM` - distance from outer surface  
        `rod_EDM` - distance from inner rod

<img src="figures/masks&EDM.png" width="600" alt="masks&EDM">

#

3) **Predict** (U-Net - resnet34)
    - manually annotate data (Napari)
    - train semantic segmentation model (~ 25 image pairs)
    - save weights (`model-weights_void.hdf5`)
    - predict un-seen images (`obj_probs`)

<img src="figures/predict.png" width="400" alt="predict">

#

4) **Segment**
    - segment voids from `obj_probs`
    - normalize void brightness (custom fitting procedure)
    - determine air and liquid masks (`air_mask`, `liquid_mask`)

<img src="figures/segment.png" width="600" alt="segment">

#

5) **Objects**
    - extract `obj_data` (see [outputs](###Outputsfolder) for more info)

## Registration
### `register_main.py`

3D stack Registration from timepoint to timepoint  

<img src="figures/misalignment.png" width="600" alt="misalignment">

1) **Match object pairs**

2) **Wrap 3D stacks**


## Outputs

### data folder

```bash
- Time0_crop_df4.tif # crop + downscale raw images  
- Time0_crop_df4_norm.tif # crop + downscale + norm. images  
- Time0_crop_df4_probs.tif # obj. predictions (model)  
- Time0_crop_df4_air_mask.tif # labelled mask of obj.  
- Time0_crop_df4_liquid_mask.tif # mask of air-filled regions  
- Time0_crop_df4_labels.tif # mask of liquid-filled regions  
- Time0_crop_df4_metadata.pkl # processing metadata  
```

### registered folder

```bash
- crop_df4_reg.tif # registered crop + downscale raw images
- crop_df4_reg_norm.tif # registered crop + downscale + norm. images
- crop_df4_reg_probs.tif # registered obj. predictions
- crop_df4_reg_air_mask.tif # registered air masks
- crop_df4_reg_liquid_mask.tif # registered liquid masks
- crop_df4_transform_matrix.pkl # registration transformation matrices
```

### outputs folder

`obj_data.csv`  
```bash
- label # obj. identification label (see `labels.tif`)
- ctrd_z # obj. z position 
- ctrd_y # obj. y position
- ctrd_x # obj. x position
- area # obj. number of voxel (3D volume)
- air_area # air number of voxel (3D volume)
- liquid_area # liquid number of voxel (3D volume)
- ratio # liquid / obj. volume
- solidity # obj. volume / obj. convex hull volume
- obj_dist # distance from neighbouring obj.
- mtx_dist # distance from external matrix surface
- category # 0 = inner obj., 1 = surface obj., 2 = rod obj
```
`plot.jpg`

## Comments
- **To fix #1**
    - expl `D1_ICONX_DOS_reg_norm.tif` z=70 t=3
    - expl `D12_ICONX_Corrosion_reg_norm.tif` z=70 t=3