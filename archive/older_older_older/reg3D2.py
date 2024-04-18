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
        np.random.randint(32, 224),
        np.random.randint(32, 224),
        np.random.randint(32, 224),
        ))   
    radii.append(np.random.randint(8, 16))
    
# Rotation matrices
theta_z, theta_y, theta_x = 0, 5, 0
rot_z = R.from_euler('z', np.radians(theta_z)).as_matrix()
rot_y = R.from_euler('y', np.radians(theta_y)).as_matrix()
rot_x = R.from_euler('x', np.radians(theta_x)).as_matrix()
rotation_matrix = rot_z @ rot_y @ rot_x

# Translation vector
translation_vector = np.array([0, 0, 0])

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

import SimpleITK as sitk
import gui
import registration_gui as rgui

sitkA = sitk.GetImageFromArray(hstackA)
sitkB = sitk.GetImageFromArray(hstackB)

# Choose Registration Components
transform_type = sitk.CenteredTransformInitializer(
    sitkA, sitkB, 
    sitk.Euler3DTransform(), 
    sitk.CenteredTransformInitializerFilter.MOMENTS
    )

# Initialize registration method
registration_method = sitk.ImageRegistrationMethod()
registration_method.SetMetricAsMeanSquares()
# registration_method.SetMetricAsJointHistogramMutualInformation()

# Set optimizer
registration_method.SetOptimizerAsGradientDescent(
    learningRate=10, 
    numberOfIterations=100,
    convergenceMinimumValue=1e-4,
    convergenceWindowSize=20,
    )

# Set interpolator
registration_method.SetInterpolator(sitk.sitkLinear)

# Initialize the Registration Method
registration_method.SetInitialTransform(transform_type)

# Execute the Registration
final_transform = registration_method.Execute(sitkA, sitkB)
print("Transformation Parameters:", final_transform.GetParameters())
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

# Apply the Transformation
RegB = sitk.Resample(
    sitkB, sitkA, 
    final_transform, 
    sitk.sitkLinear, 
    0.0, sitkB.GetPixelID()
    )

RawA = sitk.GetArrayFromImage(sitkA)
RawB = sitk.GetArrayFromImage(sitkB)
RegB = sitk.GetArrayFromImage(RegB)

import napari
viewer = napari.Viewer()
viewer.add_image(RegB, colormap='magenta', rendering="attenuated_mip")
viewer.add_image(RawB, colormap='green', rendering="attenuated_mip")
viewer.add_image(RawA, colormap='gray', rendering="attenuated_mip")