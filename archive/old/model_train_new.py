#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train')

# Prepare
mask_type = ["void", "liquid"]
df = 4
size = 1024 // df
overlap = size // 4

# Augment
iterations = 1000

# Train
n_epochs = 500
batch_size = 8
patience = 50
learning_rate = 0.001
validation_split = 0.2 

#%% Prepare -------------------------------------------------------------------

def prepare(path):
    
    # Open image and masks
    img = io.imread(Path(train_path, path.name.replace(f"_mask-{mask_type[0]}", "")))
    msk1 = (io.imread(path) > 0).astype("float32")
    msk2 = io.imread(Path(train_path, path.name.replace(f"{mask_type[0]}", f"{mask_type[1]}")))
    msk = msk1 + msk2 // 2
       
    # Prepare images
    img = norm_gcn(img, mask=img != 0)
    img = norm_pct(img, 0.01, 99.99, mask=img != 0)
    
    # Extract patches
    img_patches = extract_patches(img, size, overlap)
    msk_patches = extract_patches(msk, size, overlap)
       
    return img_patches, msk_patches

outputs = Parallel(n_jobs=-1)(
    delayed(prepare)(path)
    for path in list(train_path.glob(f"*{mask_type[0]}*"))
    )

# Format outputs
img_patches = [data[0] for data in outputs]
msk_patches = [data[1] for data in outputs]
img_patches = np.stack([arr for sublist in img_patches for arr in sublist])
msk_patches = np.stack([arr for sublist in msk_patches for arr in sublist])

# # Display 
# viewer = napari.Viewer()
# viewer.add_image(img_patches, contrast_limits=(0, 1))
# viewer.add_labels(msk_patches.astype(int)) 

#%% Augment -------------------------------------------------------------------

augment = True if iterations > 0 else False

if augment:
    
    np.random.seed(42)
    
    # Define augmentation operations
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])

    # Augment data
    def augment_data(images, masks, operations):      
        idx = np.random.randint(0, len(images) - 1)
        outputs = operations(image=images[idx,...], mask=masks[idx,...])
        return outputs['image'], outputs['mask']
    outputs = Parallel(n_jobs=-1)(
        delayed(augment_data)(img_patches, msk_patches, operations)
        for i in range(iterations)
        )
    img_patches = np.stack([data[0] for data in outputs])
    msk_patches = np.stack([data[1] for data in outputs])
    
    # -------------------------------------------------------------------------
    
    def to_categorical(msk, num_classes):
        return np.eye(num_classes)[msk.astype(int)]

    # Convert msk_patches to one-hot encoding
    msk_patches = np.stack([to_categorical(msk, num_classes=3) for msk in msk_patches])
    
    # -------------------------------------------------------------------------
    
    # Display 
    viewer = napari.Viewer()
    viewer.add_image(img_patches, contrast_limits=(0, 1))
    viewer.add_image(msk_patches) 
    
#%% Train ---------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', # ResNet 18, 34, 50, 101 or 152 
    input_shape=(None, None, 1), 
    classes=3, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy', 
    metrics=['mse'],
    )

# Checkpoint & callbacks
model_checkpoint_callback = ModelCheckpoint(
    filepath=f"model-weights_{mask_type[0]}-{mask_type[1]}_p{size:04d}_d{df}.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
    )
callbacks = [
    EarlyStopping(patience=patience, monitor='val_loss'),
    model_checkpoint_callback
    ]

# train model
history = model.fit(
    x=img_patches, y=msk_patches,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=callbacks,
    )

# Plot training results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()