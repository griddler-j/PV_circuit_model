# %% [markdown]
# # Solar Cell Circuit Demo
# This notebook shows how to build and run a silicon wafer solar cell circuit model.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *

# %% [markdown]
# ## Create a quick solar cell

#%%
cell = quick_solar_cell()

# %% [markdown]
# ## Visualize wafer geometry

#%%
cell.draw_cells()
cell.show()

# %% [markdown]
# ## Circuit model representation

#%%
cell.draw(display_value=True, title="Cell Model")

# %% [markdown]
# ## Plot IV curve

#%%
cell.plot(title="Cell I-V Curve")
cell.show()

# %% [markdown]
# # PV Module Demo
# This notebook shows how to build and run a circuit model of a PV module.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *

# %% [markdown]
# ## Create a butterfly PV module

#%%
module = quick_butterfly_module()

# %% [markdown]
# ## Visualize module cell layout

#%%
_ = module.draw_cells(show_names=True,colour_what=None)

# %% [markdown]
# ## Circuit model representation

#%%
module.draw(title="Module model")

# %% [markdown]
# ## Plot IV curve

#%%
module.plot(title="Module I-V Curve")
module.show()

# %% [markdown]
# ## Introduce some cells JL and J01 inhomogenity

#%%
for cell in module.cells:
    cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.01)))
    cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
module.build_IV()

# %% [markdown]
# ## Replot I-V Curve with the inhomogenity

#%%
module.plot(title="Module I-V Curve with inhomogenity")
module.show()

# %% [markdown]
# ## Simulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias

#%%
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cells Vint with inhomogenity",colour_bar=True)

# %% [markdown]
# ## Introduce high series resistance to cell #1 inside the module 

#%%
module.cells[0].set_specific_Rs(40.0)

# %% [markdown]
# ## Replot I-V Curve with the cell with high series resistance

#%%
module.set_Suns(1.0) 
module.plot(title="Module I-V Curve with additional high Rs cell")
module.show()

# %% [markdown]
# ## Resimulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias

#%%
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cell Vint with additional high Rs cell",colour_bar=True)

# %% [markdown]
# # Tandem Cell Demo
# This notebook shows how to build and run a perovskite-silicon tandem cell circuit model.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.cell_analysis import *
from PV_Circuit_Model.multi_junction_cell import *

# %% [markdown]
# ## Create a bottom cell

#%%
bottom_cell = quick_solar_cell()
bottom_cell.set_JL(19.0e-3)

# %% [markdown]
# ## Create a top cell

#%%
Jsc_top_cell = 20.5e-3
Voc_top_cell = 1.18
J01_PC = 5e-24

J01, J02 = estimate_cell_J01_J02(Jsc=Jsc_top_cell,Voc=Voc_top_cell,Si_intrinsic_limit=False)
top_cell = make_solar_cell(Jsc_top_cell, J01, J02, 
                        area=bottom_cell.area, 
                        Si_intrinsic_limit=False,J01_photon_coupling=J01_PC)

# %% [markdown]
# ## Put them together to make a tandem cell

#%%
tandem_cell = MultiJunctionCell([bottom_cell,top_cell])

# %% [markdown]
# ## Circuit model representation

#%%
tandem_cell.draw(display_value=True,title="Tandem Cell Model")

# %% [markdown]
# ## Plot IV curve

#%%
tandem_cell.plot(title="Tandem Cell I-V Curve")
tandem_cell.show()

# %% [markdown]
# # Tandem Cell Measurement Fitting Demo

# This example fits a tandem cell model to the tandem measurements of a large area
# perovskite-silicon solar cell.  Each measurement is stored in a json file 
# (see json_directory) in the format that PV_Circuit_Model Measurement class can read in.
# There are three kinds of measurements:
# Light I-V at different top, bottom cell JLs (i.e. spectrometric IV)
# "Dark I-V" where one subcell is in the "dark" and the other cell is illuminated
# Suns-Voc, namely with blue, red (IR) and white light spectra

#%%

from PV_Circuit_Model.data_fitting_tandem_cell import *
import os

json_directory = r"examples/tandem measurement json files/"
if not os.path.exists(json_directory):
    json_directory = r"tandem measurement json files/"
sample_info = {"area":244.26,"bottom_cell_thickness":180e-4}

measurements = get_measurements(json_directory)
ref_cell_model, interactive_fit_dashboard = analyze_solar_cell_measurements(measurements,sample_info=sample_info,is_tandem=True)

# %% [markdown]
# # Draw best fit circuit representation
# Draw the resultant tandem cell model with the best fit parameters

#%%
ref_cell_model.draw(title="Tandem Cell with Best Fit Parameters",display_value=True)


# %% [markdown]
# # Topcon Cell Measurement Fitting Demo

# This example fits a single junction silicon cell model to the measurements of a large area
# Topcon silicon wafer solar cell.  Each measurement is stored in a json file 
# (see json_directory) in the format that PV_Circuit_Model Measurement class can read in.
# There are three kinds of measurements:
# Light I-V at 1 Sun, 0.5 Sun
# Dark I-V, Suns-Voc

#%%

from PV_Circuit_Model.data_fitting_tandem_cell import *

json_directory = r"examples/topcon measurement json files/"
if not os.path.exists(json_directory):
    json_directory = r"topcon measurement json files/"
sample_info = {"area":165.34,"bottom_cell_thickness":180e-4}

measurements = get_measurements(json_directory)

ref_cell_model, interactive_fit_dashboard = analyze_solar_cell_measurements(measurements,sample_info=sample_info,is_tandem=False,num_of_rounds=15)

# %% [markdown]
# # Draw best fit circuit representation
# Draw the resultant cell model with the best fit parameters

#%%
ref_cell_model.draw(title="Tandem Cell with Best Fit Parameters",display_value=True)






