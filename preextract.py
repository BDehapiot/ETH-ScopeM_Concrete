#%% Imports -------------------------------------------------------------------

from pathlib import Path
from functions import import_stack

#%% Inputs --------------------------------------------------------------------

experiments = [
    "D1_ICONX_DoS",
    "D11_ICONX_DoS",
    "D12_ICONX_corrosion", 
    "H9_ICONX_DoS",
    ]
    
# Paths
data_path = Path("D:/local_Concrete/data", "0-raw")
save_path = Path("D:/local_Concrete/data", experiment)



# Parameter

save_path.mkdir(parents=True, exist_ok=True)

#%% Import --------------------------------------------------------------------

for exp in experiments:
    exp_path = Path("D:/local_Concrete/data", exp)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    
    for stack_path in data_path.iterdir():
        


           
    test_path = Path(save_path, stack_path.name + "_crop_d1.tif")

    if experiment == "all":
        if not test_path.is_file():
            import_stack(stack_path, save_path)
    
    else:
        if not test_path.is_file() and experiment in stack_path.name:
            import_stack(stack_path, save_path)