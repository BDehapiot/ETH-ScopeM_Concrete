# ETH-ScopeM_Concrete
Process and display concrete cylinders X-Ray tomographs

## Content
- Model:  
Train a U-Net deep-learning model to segment voids contained in the concrete 
matrix. 

    - `model_extract.py`
    - `model_annotate.py`
    - `model_train.py`

- Process:  

    - `process_preprocess.py`
    - `process_predict.py`
    - `process_segment.py`
    - `process_objects.py`  
    - `process_main.py`

- Register

    - `register_postprocess.py`  
    - `register_main.py`

## Outputs
- `obj_data.csv`

## To fix
- issue #1
    - expl `D1_ICONX_DOS_reg_norm.tif` z=70 t=3
    - expl `D12_ICONX_Corrosion_reg_norm.tif` z=70 t=3
    
