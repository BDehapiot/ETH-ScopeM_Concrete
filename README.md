![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2023--11--06-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Concrete  
Concrete rebar CT scan analysis tool  
## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment-tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment-tf-nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate concrete
```
Your prompt should now start with `(concrete)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment-tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment-tf-nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate concrete
```
Your prompt should now start with `(concrete)` instead of `(base)`

</details>  
  
## Procedure

### process_main.py
Main processing tasks executed on original image stacks

- **Downscale** (reduce processing time)  
```bash
# For downscale factor (df) = 4  
original stack (z, y, x)  = 1948 x 1788 x 1788 = 13 Gb  
downscaled stack (z, y, x)  = 487 x 447 x 447 = 186 Mb
```
#

- **Preprocess**
    - center images over Z axis
    - compute median projection (`med_proj`)
    - normalize images (divide by `med_proj` to get `norm`)
    - determine matrix and rod masks (`mtx_mask`, `rod_mask`)
    - compute distance maps  
        `mtx_EDM` - distance from outer surface  
        `rod_EDM` - distance from inner rod

<img src="figures/masks&EDM.png" width="600" alt="masks&EDM">

#

- **Predict** (U-Net - resnet34)
    - manually annotate data (Napari)
    - train semantic segmentation model (~ 25 image pairs)
    - save weights (`model-weights_void.hdf5`)
    - predict un-seen images (`obj_probs`)

<img src="figures/predict.png" width="400" alt="predict">

#

- **Segment**
    - segment voids from `obj_probs`
    - normalize void brightness (custom fitting procedure)
    - determine air and liquid masks (`air_mask`, `liquid_mask`)

<img src="figures/segment.png" width="600" alt="segment">

#

- **Objects**
    - segment voids from `obj_probs`
    - normalize void brightness (custom fitting procedure)
    - determine air and liquid masks (`air_mask`, `liquid_mask`)

## Content
### Process  
- **process_main.py** - mask, norm, prediction & obj. data 
    
### Register
- **register_main.py** - spatial registration & histogram matching 

### Display  
- **display_mask.py** - show air/liquid masks (Napari)
- **display_probs.py** - show model predictions (Napari)

### Analyse
- **analyse_main.py**

### Model:  
- **model_extract.py** - extract train. images 
- **model_annotate.py** - annotate train. images (Napari)
- **model_train.py** - train model

### Others
- **functions.py** - contains all required functions
- **environment-gpu.yml** - dependencies with GPU support (NVIDIA GPU required)
- **environment-nogpu.yml** - dependencies with no GPU support
- **model-weights_void.h5** - model weights for void segmentation

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

- obj_data.csv  
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
- plot.jpg  

## Comments
- **To fix #1**
    - expl `D1_ICONX_DOS_reg_norm.tif` z=70 t=3
    - expl `D12_ICONX_Corrosion_reg_norm.tif` z=70 t=3  
