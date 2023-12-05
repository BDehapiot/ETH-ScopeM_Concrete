#%% Imports 

import numpy as np
from skimage import io
from pathlib import Path
import scipy.ndimage as ndi

#%%

data_path = "D:/local_Concrete/data/3D"

#%%

from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------

def draw_ball(center, radius, hstack1):    
    z, y, x = np.ogrid[:hstack1.shape[0], :hstack1.shape[1], :hstack1.shape[2]]
    dist = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    mask = dist <= radius**2
    hstack1[mask] = 1
    return hstack1

# -----------------------------------------------------------------------------

# Define centers and radii
centersA, radii = [], []
for i in range(10):
    centersA.append((
        np.random.randint(64, 192),
        np.random.randint(64, 192),
        np.random.randint(64, 192),
        ))   
    radii.append(np.random.randint(8, 16))
    
# Rotation matrices
theta_z, theta_y, theta_x = 5, 10, 20
rot_z = R.from_euler('z', np.radians(theta_z)).as_matrix()
rot_y = R.from_euler('y', np.radians(theta_y)).as_matrix()
rot_x = R.from_euler('x', np.radians(theta_x)).as_matrix()
rotation_matrix = rot_z @ rot_y @ rot_x

# Translation vector
translation_vector = np.array([10, 10, 5])

# Transform centers
centersB = []
for centerA in centersA:
    centersB.append(rotation_matrix @ centerA + translation_vector)

# Fill 3D arrays
hstackA = np.zeros((256, 256, 256), dtype=float)
hstackB = np.zeros((256, 256, 256), dtype=float)
for centerA, centerB, radius in zip(centersA, centersB, radii):
    hstackA = draw_ball(centerA, radius, hstackA)
    hstackB = draw_ball(centerB, radius, hstackB)
    
# # 
# hstackA = ndi.distance_transform_edt(1 - hstackA) 
# hstackB = ndi.distance_transform_edt(1 - hstackB)
    
# import napari
# viewer = napari.Viewer()
# viewer.add_image(hstackB, colormap='green', rendering="attenuated_mip")
# viewer.add_image(hstackA, colormap='gray', rendering="attenuated_mip")

#%%

from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import RigidTransform3D

# Set up the Mutual Information metric
metric = MutualInformationMetric(nbins=32, sampling_proportion=100)

# Initialize the Affine registration object with the Rigid transform
affreg = AffineRegistration(
    metric=metric, 
    level_iters=[10000, 1000, 100], 
    sigmas=[3.0, 1.0, 0.0], 
    factors=[4, 2, 1]
    )

# Apply the rigid body registration
rigid = affreg.optimize(
    static=hstackA, 
    moving=hstackB, 
    transform=RigidTransform3D(),
    params0=None, 
    starting_affine=np.eye(4)
    )

# Apply the transformation to hstackB for alignment
hstackB_aligned = rigid.transform(hstackB)

#%%

import napari
viewer = napari.Viewer()
viewer.add_image(hstackB_aligned, colormap='magenta', rendering="attenuated_mip")
viewer.add_image(hstackB, colormap='green', rendering="attenuated_mip")
viewer.add_image(hstackA, colormap='gray', rendering="attenuated_mip")