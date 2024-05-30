#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
from functions import preprocess

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train') 

# Parameters
edit = True
mask_type = "matrix"
randomize, seed = True, 42
contrast_limits = (0, 1)
brush_size = 1200

#%% Initialize ----------------------------------------------------------------

metadata = []
for path in train_path.iterdir():
    if "mask" not in path.name:
        metadata.append({
            "name"  : path.name,
            "path"  : path,
            })
       
idx = 0
np.random.seed(seed)
if randomize:
    idxs = np.random.permutation(len(metadata))
else:
    idxs = np.arange(len(metadata))
    
#%% Functions -----------------------------------------------------------------

def update_info_text():
    
    img_path = metadata[idxs[idx]]["path"]
    msk_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
    img_name = img_path.name
    if msk_path.exists() and edit:
        msk_name = msk_path.name 
    elif not msk_path.exists():
        msk_name = "None"

    info_text = (
        
        "<p><b>Image</b><br>"
        f"<span style='color: Khaki;'>{img_name}</span>"
        
        "<p><b>Mask</b><br>"
        f"<span style='color: Khaki;'>{msk_name}</span>"
        
        "<p><b>Shortcuts</b><br>"
        f"- Save Mask      {'&nbsp;' * 5}:<span style='color: LightSteelBlue;'> Enter</span><br>"
        f"- Next Image     {'&nbsp;' * 4}:<span style='color: LightSteelBlue;'> PageUp</span><br>"
        f"- Previous Image {'&nbsp;' * 0}:<span style='color: LightSteelBlue;'> PageDown</span><br>"
        f"- Next Label     {'&nbsp;' * 4}:<span style='color: LightSteelBlue;'> ArrowUp</span><br>"
        f"- Previous Label {'&nbsp;' * 0}:<span style='color: LightSteelBlue;'> ArrowDown</span><br>"
        f"- Increase brush {'&nbsp;' * 0}:<span style='color: LightSteelBlue;'> ArrowRight</span><br>"
        f"- Decrease brush {'&nbsp;' * 0}:<span style='color: LightSteelBlue;'> ArrowLeft</span><br>"
        f"- Erase tool     {'&nbsp;' * 4}:<span style='color: LightSteelBlue;'> End</span><br>"
        f"- Fill tool      {'&nbsp;' * 5}:<span style='color: LightSteelBlue;'> Home</span><br>"
        f"- Pan Image      {'&nbsp;' * 5}:<span style='color: LightSteelBlue;'> Space</span><br>"
        
        )
    
    return info_text

# -----------------------------------------------------------------------------

def open_image():
    
    viewer.layers.clear()
    img_path = metadata[idxs[idx]]["path"]
    msk_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type}.tif"))
    
    if msk_path.exists() and edit:   
        img = io.imread(img_path)
        msk = io.imread(msk_path)
    elif not msk_path.exists():
        img = io.imread(img_path)
        msk = np.zeros_like(img, dtype="uint8")
        
    # Preprocess image
    img = preprocess(img, 0.01, 99.99, 10)
        
    if "img" in locals():
        viewer.add_image(img, name="image", metadata=metadata[idxs[idx]])
        viewer.add_labels(msk, name="mask")
        viewer.layers["image"].contrast_limits = contrast_limits
        viewer.layers["image"].gamma = 0.66
        viewer.layers["mask"].brush_size = brush_size
        viewer.layers["mask"].selected_label = 1
        viewer.layers["mask"].mode = 'paint'
        viewer.layers["mask"].opacity = 0.5
        
def save_mask():
    path = viewer.layers["image"].metadata["path"]
    path = Path(str(path).replace(".tif", f"_mask-{mask_type}.tif"))
    io.imsave(path, viewer.layers["mask"].data, check_contrast=False)  
    info.setText(update_info_text())
    
def next_image(): 
    global idx, info
    idx = idx + 1
    open_image()
    info.setText(update_info_text())
        
def previous_image():
    global idx, info
    idx = idx - 1
    open_image()
    info.setText(update_info_text())

# -----------------------------------------------------------------------------

def next_label():
    viewer.layers["mask"].selected_label += 1 

def previous_label():
    if viewer.layers["mask"].selected_label > 1:
        viewer.layers["mask"].selected_label -= 1 
        
def increase_brush_size():
    viewer.layers["mask"].brush_size += 2
    
def decrease_brush_size():
    viewer.layers["mask"].brush_size -= 2
        
def paint():
    viewer.layers["mask"].mode = 'paint'
        
def erase():
    viewer.layers["mask"].mode = 'erase'
    
def fill():
    viewer.layers["mask"].mode = 'fill'
    
def pan():
    viewer.layers["mask"].mode = 'pan_zoom'
    
#%% Shortcuts -----------------------------------------------------------------
    
@napari.Viewer.bind_key('Enter', overwrite=True)
def save_mask_key(viewer):
    save_mask()
    
@napari.Viewer.bind_key('PageUp', overwrite=True)
def next_image_key(viewer):
    next_image()
    
@napari.Viewer.bind_key('PageDown', overwrite=True)
def previous_image_key(viewer):
    previous_image()
    
# -----------------------------------------------------------------------------
    
@napari.Viewer.bind_key('Up', overwrite=True)
def next_label_key(viewer):
    next_label()
    
@napari.Viewer.bind_key('Down', overwrite=True)
def previous_label_key(viewer):
    previous_label()
    
@napari.Viewer.bind_key('Right', overwrite=True)
def increase_brush_size_key(viewer):
    increase_brush_size()
    
@napari.Viewer.bind_key('Left', overwrite=True)
def decrease_brush_size_key(viewer):
    decrease_brush_size()
    
@napari.Viewer.bind_key('End', overwrite=True)
def erase_switch(viewer):
    erase()
    yield
    paint()
       
@napari.Viewer.bind_key('Home', overwrite=True)
def fill_switch(viewer):
    fill()
    yield
    paint()
    
@napari.Viewer.bind_key('Space', overwrite=True)
def pan_switch(viewer):
    pan()
    yield
    paint()
 
#%% Execute -------------------------------------------------------------------
 
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QLabel

# -----------------------------------------------------------------------------

# Initialize viewer
viewer = napari.Viewer()
viewer.text_overlay.visible = True

# Create a QWidget to hold buttons
widget = QWidget()
layout = QVBoxLayout()

# Create buttons
btn_save_mask = QPushButton("Save Mask")
btn_next_image = QPushButton("Next Image")
btn_previous_image = QPushButton("Previous Image")

# Create texts
info = QLabel()
info.setFont(QFont("Consolas", 6))
info.setText(update_info_text())

# Add buttons and text to layout
layout.addWidget(btn_save_mask)
layout.addWidget(btn_next_image)
layout.addWidget(btn_previous_image)
layout.addSpacing(20)
layout.addWidget(info)

# Set layout to the widget
widget.setLayout(layout)

# Connect buttons to their respective functions
btn_save_mask.clicked.connect(save_mask)
btn_next_image.clicked.connect(next_image)
btn_previous_image.clicked.connect(previous_image)

# Add the QWidget as a dock widget to the Napari viewer
viewer.window.add_dock_widget(widget, area='right', name="Annotate")
open_image()
napari.run()