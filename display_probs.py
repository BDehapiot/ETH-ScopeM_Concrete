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
experiment = "D11_ICONX_DoS"
# experiment = "D12_ICONX_corrosion"
# experiment = "H1_ICONX_DoS"
# experiment = "H5_ICONX_corrosion"
# experiment = "H9_ICONX_DoS"

#%% Initialize ----------------------------------------------------------------

# Paths
experiment_path = data_path / experiment
experiment_reg_path = data_path / experiment / "registered"
paths = list(experiment_path.glob(f"*_crop_df{df}.tif*"))

# Open data
stack_reg_norm = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_reg_norm.tif"))
probs_reg = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_probs_reg.tif"))
norm_reg = io.imread(experiment_reg_path / (experiment + f"_crop_df{df}_norm_reg.tif"))

# Prepare data
mask = norm_reg > 0
probs_reg[norm_reg == 0] = 0
probs_reg[probs_reg < 0] = 0

#%% Functions -----------------------------------------------------------------

def update_info_text(t):
    
    rnames, frnames = [], []
    pnames, fpnames = [], []
    for i, path in enumerate(paths):
        
        # rnames
        rname = path.stem
        rname += "_reg.tif" if i > 0 else ".tif"
        rnames.append(rname)
        
        # pnames
        pname = path.stem
        pname += "_probs_reg.tif" if i > 0 else "_probs.tif"
        pnames.append(pname)
        
        # Formated names
        color = "Gainsboro" if t != i else "Khaki"
        frnames.append(f"<span style='color: {color};'>{rname}<br></span>")
        fpnames.append(f"<span style='color: {color};'>{pname}<br></span>")
    
    frnames = '\n'.join(frnames)
    fpnames = '\n'.join(fpnames)
            
    info_text = (
        
        "<p><b>Experiment</b><br>"
        f"<span style='color: Khaki;'>{experiment}</span>"
        
        "<p><b>File (raw)</b><br>"
        f"{frnames}"
        
        "<p><b>File (probs)</b><br>"
        f"{fpnames}"

        )
    
    return info_text

def update_viewer():
    info.setText(update_info_text(viewer.dims.point[0]))

#%% Execute -------------------------------------------------------------------

# Initialize viewer
viewer = napari.Viewer()
viewer.add_image(stack_reg_norm, name="raw")
viewer.add_image(
    probs_reg, 
    name="probs",
    blending="additive", 
    rendering="attenuated_mip",
    colormap="bop orange",
    opacity=0.5, 
    visible=False
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