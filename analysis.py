#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path
from scipy.ndimage import shift

#%% Inputs --------------------------------------------------------------------

data_path = "D:/local_Concrete/data/DIA"
exp_name = (
    "D1_ICONX_DoS"
    # "D11_ICONX_DoS"
    # "D12_ICONX_corrosion"
    # "H9_ICONX_DoS"
    )

#%%

# -----------------------------------------------------------------------------

def roll_image(img_rsize):
    idx = np.argwhere((img_rsize > 30000) == 1)
    y0, x0 = img_rsize.shape[0] // 2, img_rsize.shape[1] // 2
    y1, x1 = np.mean(idx, axis=0)
    yx_shift = [y0 - y1, x0 - x1]
    return shift(img_rsize, yx_shift, mode='wrap'), yx_shift 

# -----------------------------------------------------------------------------

# Open data
stack_reg = io.imread(Path(data_path, f"{exp_name}_reg.tif"))
stack_reg_mean = np.mean(stack_reg, axis=0)

# Save 
io.imsave(
    Path(data_path, f"{exp_name}_stack_reg_mean.tif"),
    stack_reg_mean.astype("float32"),
    check_contrast=False,
    )

# # Roll stack
# print("  Roll       :", end='')
# t0 = time.time()
# outputs = Parallel(n_jobs=-1)(
#         delayed(roll_image)(img_rsize) 
#         for img_rsize in stack_rsize
#         )
# stack_roll = np.stack([data[0] for data in outputs])
# yx_shifts = [data[1] for data in outputs]
# t1 = time.time()
# print(f" {(t1-t0):<5.2f}s") 