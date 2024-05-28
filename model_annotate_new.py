#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), 'data', 'train') 

# Parameters
edit = True
mask_type = ["void", "liquid"]
randomize, seed = True, 42
contrast_limits = (0, 1.5)
brush_size = 5

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
    msk1_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type[0]}.tif"))
    msk2_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type[1]}.tif"))
    img_name = img_path.name
    if msk1_path.exists() and edit:
        msk1_name = msk1_path.name 
    elif not msk1_path.exists():
        msk1_name = "None"
    if msk2_path.exists() and edit:
        msk2_name = msk2_path.name 
    elif not msk2_path.exists():
        msk2_name = "None"

    info_text = (
        
        "<p><b>Image</b><br>"
        f"<span style='color: Khaki;'>{img_name}</span>"
        
        "<p><b>Mask void</b><br>"
        f"<span style='color: Khaki;'>{msk1_name}</span>"
        
        "<p><b>Mask liquid</b><br>"
        f"<span style='color: Khaki;'>{msk2_name}</span>"
        
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
    msk1_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type[0]}.tif"))
    msk2_path = Path(str(img_path).replace(".tif", f"_mask-{mask_type[1]}.tif"))
    
    img = io.imread(img_path)
    if msk1_path.exists() and edit:   
        msk1 = io.imread(msk1_path)
    elif not msk1_path.exists():
        msk1 = np.zeros_like(img, dtype="uint8")
    if msk2_path.exists() and edit:   
        msk2 = io.imread(msk2_path)
    elif not msk2_path.exists():
        msk2 = np.zeros_like(img, dtype="uint8")
                
    if "img" in locals():
        viewer.add_image(img, name="image", metadata=metadata[idxs[idx]])
        viewer.add_labels(msk1, name=mask_type[0])
        viewer.add_labels(msk2, name=mask_type[1])
        viewer.layers["image"].contrast_limits = contrast_limits
        viewer.layers["image"].gamma = 0.66
        viewer.layers[mask_type[0]].brush_size = brush_size
        viewer.layers[mask_type[0]].selected_label = 1
        viewer.layers[mask_type[0]].mode = 'paint'
        viewer.layers[mask_type[0]].opacity = 0.5
        viewer.layers[mask_type[1]].brush_size = brush_size
        viewer.layers[mask_type[1]].selected_label = 2
        viewer.layers[mask_type[1]].mode = 'paint'
        viewer.layers[mask_type[1]].opacity = 0.5
     
def save_mask():
    
    # Update
    msk1 = viewer.layers[mask_type[0]].data
    msk2 = viewer.layers[mask_type[1]].data
    msk2[msk1 == 0] = 0
    viewer.layers[mask_type[1]].data = msk2
    
    # Save
    path = viewer.layers["image"].metadata["path"]
    msk1_path = Path(str(path).replace(".tif", f"_mask-{mask_type[0]}.tif"))
    msk2_path = Path(str(path).replace(".tif", f"_mask-{mask_type[1]}.tif"))
    io.imsave(msk1_path, msk1, check_contrast=False)  
    io.imsave(msk2_path, msk2, check_contrast=False) 
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
       
def increase_brush_size():
    viewer.layers[mask_type[0]].brush_size += 1
    viewer.layers[mask_type[1]].brush_size += 1
    
def decrease_brush_size():
    viewer.layers[mask_type[0]].brush_size -= 1
    viewer.layers[mask_type[1]].brush_size -= 1
        
def paint():
    viewer.layers[mask_type[0]].mode = 'paint'
    viewer.layers[mask_type[1]].mode = 'paint'
        
def erase():
    viewer.layers[mask_type[0]].mode = 'erase'
    viewer.layers[mask_type[1]].mode = 'erase'
    
def fill():
    viewer.layers[mask_type[0]].mode = 'fill'
    viewer.layers[mask_type[1]].mode = 'fill'
    
def pan():
    viewer.layers[mask_type[0]].mode = 'pan_zoom'
    viewer.layers[mask_type[1]].mode = 'pan_zoom'
  
# def show_mask1():
#     viewer.layers[mask_type[0]].visible = True   
# def hide_mask1():
#     viewer.layers[mask_type[0]].visible = False
   
# def show_mask2():
#     viewer.layers[mask_type[1]].visible = True   
# def hide_mask2():
#     viewer.layers[mask_type[1]].visible = False
    
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
    
@napari.Viewer.bind_key('Insert', overwrite=True)
def mask1_switch(viewer):
    mask_layer = viewer.layers[mask_type[0]]
    mask_layer.visible = not mask_layer.visible
    
@napari.Viewer.bind_key('Delete', overwrite=True)
def mask2_switch(viewer):
    mask_layer = viewer.layers[mask_type[1]]
    mask_layer.visible = not mask_layer.visible
 
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