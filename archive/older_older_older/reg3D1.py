#%% Imports 

import numpy as np
from skimage import io
from pathlib import Path
import scipy.ndimage as ndi

#%%

data_path = "D:/local_Concrete/data/3D"

#%%

def draw_ball(center, radius, hstack1):    
    z, y, x = np.ogrid[:hstack1.shape[0], :hstack1.shape[1], :hstack1.shape[2]]
    dist = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    mask = dist <= radius**2
    hstack1[mask] = 1
    return hstack1

def rotate_3d(matrix, angle, axis):
    if axis == 0: axes = (1, 2)
    elif axis == 1: axes = (0, 2)
    elif axis == 2: axes = (0, 1)
    else: raise ValueError('Invalid axis for 3D rotation')
    return ndi.rotate(matrix, angle, axes=axes, reshape=False, mode='nearest')

def apply_transform(hstack1, rotation_angles, translation):
    for axis, angle in enumerate(rotation_angles):
        hstack1 = rotate_3d(hstack1, angle, axis)
    hstack1 = ndi.shift(hstack1, shift=translation)
    return hstack1

# -----------------------------------------------------------------------------

hstack1 = np.zeros((256, 256, 256), dtype=float)
for i in range(10):
    center = (
        np.random.randint(32, 224),
        np.random.randint(32, 224),
        np.random.randint(32, 224),
        )
    radius = np.random.randint(8, 16)
    hstack1 = draw_ball(center, radius, hstack1)
    
# -----------------------------------------------------------------------------

rotation_angles = (30, 45, 60)
translation = (10, 20, 30)
hstack2 = apply_transform(hstack1, rotation_angles, translation)

# -----------------------------------------------------------------------------

io.imsave(
    Path(data_path, f"hstack1.tif"),
    hstack1.astype("float32"), check_contrast=False,
    )

io.imsave(
    Path(data_path, f"hstack2.tif"),
    hstack2.astype("float32"), check_contrast=False,
    )

# -----------------------------------------------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(hstack2, rendering="attenuated_mip")
viewer.add_image(hstack1, rendering="attenuated_mip")
