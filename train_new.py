#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from functions import preprocess
from bdtools import extract_patches
from joblib import Parallel, delayed 
from skimage.transform import downscale_local_mean

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train')

# Prepare
mask_type = "matrix"
df = 4
size = 1024 // df
overlap = size // 4

# Augment
iterations = 200

# Train
n_epochs = 500
batch_size = 8
patience = 50
learning_rate = 0.001
validation_split = 0.2 

#%% Prepare -------------------------------------------------------------------

# def prepare(path):
    
#     # Open image and mask
#     img = io.imread(Path(train_path, path.name.replace(f"_mask-{mask_type}", "")))
#     msk = (io.imread(path) > 0).astype("float32")
    
#     # Downscale data
#     if downscale_factor > 1:
#         img = downscale_local_mean(img, downscale_factor)
#         msk = downscale_local_mean(msk, downscale_factor)
    
#     # Preprocess image
#     img = preprocess(img, 0.01, 99.99, 10 // downscale_factor)
        
#     # Extract patches
#     img_patches = extract_patches(img, size, overlap)
#     msk_patches = extract_patches(msk, size, overlap)
       
#     return img_patches, msk_patches

# outputs = Parallel(n_jobs=-1)(
#     delayed(prepare)(path)
#     for path in list(train_path.glob(f"*{mask_type}*"))
#     )

# # Format outputs
# img_patches = [data[0] for data in outputs]
# msk_patches = [data[1] for data in outputs]
# img_patches = np.stack([arr for sublist in img_patches for arr in sublist])
# msk_patches = np.stack([arr for sublist in msk_patches for arr in sublist])

# # Remove empty patches
# keep = np.full(len(msk_patches), True)
# for i, msk_patch in enumerate(msk_patches):
#     if np.max(msk_patch) == 0:
#         keep[i] = False
# img_patches = img_patches[keep, ...]
# msk_patches = msk_patches[keep, ...]

# # # Display 
# # viewer = napari.Viewer()
# # viewer.add_image(img_patches, contrast_limits=(0, 1))
# # viewer.add_image(msk_patches, contrast_limits=(0, 1)) 

#%% Augment -------------------------------------------------------------------

# augment = True if iterations > 0 else False

# if augment:
    
#     np.random.seed(42)
    
#     # Define augmentation operations
#     operations = A.Compose([
#         A.VerticalFlip(p=0.5),              
#         A.RandomRotate90(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.Transpose(p=0.5),
#         A.GridDistortion(p=0.5),
#         ])

#     # Augment data
#     def augment_data(images, masks, operations):      
#         idx = np.random.randint(0, len(images) - 1)
#         outputs = operations(image=images[idx,...], mask=masks[idx,...])
#         return outputs['image'], outputs['mask']
#     outputs = Parallel(n_jobs=-1)(
#         delayed(augment_data)(img_patches, msk_patches, operations)
#         for i in range(iterations)
#         )
#     img_patches = np.stack([data[0] for data in outputs])
#     msk_patches = np.stack([data[1] for data in outputs])
    
#     # # Display 
#     # viewer = napari.Viewer()
#     # viewer.add_image(img_patches, contrast_limits=(0, 1))
#     # viewer.add_image(msk_patches, contrast_limits=(0, 1)) 

#%% Train ---------------------------------------------------------------------

# # Define & compile model
# model = sm.Unet(
#     'resnet18', # ResNet 18, 34, 50, 101 or 152 
#     input_shape=(None, None, 1), 
#     classes=1, 
#     activation='sigmoid', 
#     encoder_weights=None,
#     )
# model.compile(
#     optimizer=Adam(learning_rate=learning_rate),
#     loss='binary_crossentropy', 
#     metrics=['mse'],
#     )

# # Checkpoint & callbacks
# model_checkpoint_callback = ModelCheckpoint(
#     filepath=f"model-weights_{mask_type}_p{size:04d}_d{downscale_factor}.h5",
#     save_weights_only=True,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True
#     )
# callbacks = [
#     EarlyStopping(patience=patience, monitor='val_loss'),
#     model_checkpoint_callback
#     ]

# # train model
# history = model.fit(
#     x=img_patches, y=msk_patches,
#     validation_split=validation_split,
#     batch_size=batch_size,
#     epochs=n_epochs,
#     callbacks=callbacks,
#     )

# # Plot training results
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()