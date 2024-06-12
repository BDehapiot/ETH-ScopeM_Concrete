#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

# QT
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel

#%% Inputs --------------------------------------------------------------------

# Parameters
overwrite = True
df = 4 # downscale factor

# Paths
data_path = Path("D:/local_Concrete/data")
# experiment = "D1_ICONX_DoS"
# experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
experiment = "H5_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"

#%% Initialize ----------------------------------------------------------------

# Paths
experiment_path = data_path / experiment
experiment_reg_path = data_path / experiment / "registered"
paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))

# Open data
stack_reg_norm = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_reg_norm.tif"))
air_mask_reg = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_air_mask_reg.tif"))
liquid_mask_reg = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_liquid_mask_reg.tif"))

#%% Functions -----------------------------------------------------------------

def update_info_text(t):
    
    rnames, frnames = [], []
    anames, fanames = [], []
    lnames, flnames = [], []
    for i, path in enumerate(paths):
        
        # rnames
        rname = path.stem
        rname += "_reg.tif" if i > 0 else ".tif"
        rnames.append(rname)
        
        # anames
        aname = path.stem
        aname += "_air_mask_reg.tif" if i > 0 else "_air_mask.tif"
        anames.append(aname)
        
        # lnames
        lname = path.stem
        lname += "_liquid_mask_reg.tif" if i > 0 else "_liquid_mask.tif"
        lnames.append(lname)
        
        # Formated names
        color = "Gainsboro" if t != i else "Khaki"
        frnames.append(f"<span style='color: {color};'>{rname}<br></span>")
        fanames.append(f"<span style='color: {color};'>{aname}<br></span>")
        flnames.append(f"<span style='color: {color};'>{lname}<br></span>")
    
    frnames = '\n'.join(frnames)
    fanames = '\n'.join(fanames)
    flnames = '\n'.join(flnames)
            
    info_text = (
        
        "<p><b>Experiment</b><br>"
        f"<span style='color: Khaki;'>{experiment}</span>"
        
        "<p><b>File (raw)</b><br>"
        f"{frnames}"
        
        "<p><b>File (air_mask)</b><br>"
        f"{fanames}"
        
        "<p><b>File (liquid_mask)</b><br>"
        f"{flnames}"
        
        "<p><b>Shortcuts</b><br>"
        f"- Hide masks {'&nbsp;' * 0}:<span style='color: LightSteelBlue;'> Enter</span><br>"

        )
    
    return info_text

def update_viewer():
    info.setText(update_info_text(viewer.dims.point[0]))
    
def hide_masks():
    viewer.layers["air_mask"].visible = False
    viewer.layers["liquid_mask"].visible = False
    
def show_masks():
    viewer.layers["air_mask"].visible = True
    viewer.layers["liquid_mask"].visible = True
    
#%%

@napari.Viewer.bind_key('Enter', overwrite=True)
def mask_switch(viewer):
    hide_masks()
    yield
    show_masks()

#%% Execute -------------------------------------------------------------------

# Initialize viewer
viewer = napari.Viewer()
viewer.add_image(stack_reg_norm, name="raw")
viewer.add_image(
    air_mask_reg,
    name="air_mask",
    blending="additive", 
    rendering="attenuated_mip",
    colormap="bop orange",
    opacity=0.5, 
    visible=True
    )
viewer.add_image(
    liquid_mask_reg, 
    name="liquid_mask",
    blending="additive", 
    rendering="attenuated_mip",
    colormap="bop blue",
    opacity=0.5, 
    visible=True
    )

# Qwidget
widget = QWidget()
layout = QVBoxLayout()
widget.setLayout(layout)

# Qlabel
info = QLabel()
info.setFont(QFont("Consolas", 6))
info.setText(update_info_text(viewer.dims.point[0]))
layout.addWidget(info)

# Display & update widget
viewer.window.add_dock_widget(widget, area='right', name="Display")
viewer.dims.events.point.connect(update_viewer)

napari.run()