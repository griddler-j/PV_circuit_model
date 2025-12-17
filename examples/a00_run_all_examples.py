# %% [markdown]
# # Solar Cell Circuit Demo
# This notebook shows how to build and run a silicon wafer solar cell circuit model.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

# %% [markdown]
# ## Construction from the ground up, from circuit elements

#%%
# A solar cell can be made of these circuit elements.  Note these notations below:
# Note that A | B means "connect A, B in parallel", and A + B means "connect A, B in series"
circuit_group = (CurrentSource(41e-3) | 
                 ForwardDiode(I0=10e-15,n=1) | 
                 ForwardDiode(I0=5e-9,n=2) | 
                 Intrinsic_Si_diode(base_thickness=180e-4) | 
                 ReverseDiode(V_shift=10) | 
                 Resistor(cond=1/1e5)) + Resistor(3)

#%%
# .draw will draw the circuit diagram
circuit_group.draw(display_value=True)

#%% 
# .plot will plot the I-V Curve
circuit_group.plot(title="Cell Parts I-V Curve")
circuit_group.show()

# %% [markdown]
# ## Type Casting

#%%
# We can cast circuit_group as type Cell to give it additional shape and area
cell_ = circuit_group.as_type(Cell, **wafer_shape(format="M10",half_cut=True))
#%%
# Now cell has a shape and size that we can see
_ = cell_.draw_cells()
#%%
# Also, plotting a cell will show the I-V curve with the current density multiplied by the cell area
cell_.plot(title="Cell I-V Curve")
cell_.show()

# %% [markdown]
# ## Short Cut

#%%
# Because cells are frequently defined, we offer a short cut definition
# quick_solar_cell has the advantage that you can specify target I-V parameters for the diode parameters to tune to
cell = quick_solar_cell(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, wafer_format="M10",half_cut=True)

#%%
cell.plot(title="Cell I-V Curve")
cell.show()

#%% 
# Verify that the cells defined these two ways have the same structure
print("Does cell_ and cell have the same structure? ", "Yes" if cell_.structure()==cell.structure() else "No")

# %% [markdown]
# # PV Module Demo
# This notebook shows how to build and run a circuit model of a PV module.

# %% [markdown]
# ## Construction from the ground up, from a cell

#%%
# Let's put 24 x 2 x 3 = 144 cells together to make a module.  Note these notations below:
# A*24 = A + A .... + A = connect 24 copies of A's together in series
# tile_subgroups is optional to arrange the cells spatially, just for visualization
half_string = (cell*24 + Resistor(cond=20)).tile_subgroups(cols=2,x_gap=0.1,y_gap=0.1,turn=True)

#%%
# B**2 = B | B = connect 2 copies of B's together in parallel
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
section = (half_string**2 | ReverseDiode()).tile_subgroups(cols = 1, y_gap = 1, yflip=True)

#%%
# C*3 = C + C + C = connect 3 copies of C's together in series
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
circuit_group = (section*3).tile_subgroups(rows=1, x_gap = 1)

#%%
# type cast to Module just for encapsulation
module_ = circuit_group.as_type(Module) 
_ = module_.draw_cells()

#%%
module_.plot(title="Module I-V Curve")
module_.show()

# %% [markdown]
# ## Short Cut

#%%
# Because modules are frequently defined, we offer a short cut definition
module = quick_module(Isc=14, Voc=0.72*72, FF=0.8, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, half_cut=True, butterfly=True)
#%%
module.plot(title="Module I-V Curve")
module.show()

#%% 
# Verify that the modules defined these two ways have the same structure
print("Does module_ and module have the same structure? ", "Yes" if module_.structure()==module.structure() else "No")

# %% [markdown]
# ## Introduce some cells JL and J01 inhomogenity

#%%
# Manipulate the cell properties
np.random.seed(0)
for cell in module.cells:
    cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.01)))
    cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))

#%%
module.plot(title="Module I-V Curve with inhomogenity")
module.show()

#%%
# Simulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cells Vint with inhomogenity",colour_bar=True) 

# %% [markdown]
# ## Introduce high series resistance to cell #1 inside the module 

#%%
# Give one of the cells very large series resistance
module.cells[0].set_specific_Rs(40.0)
module.set_Suns(1.0) 
module.plot(title="Module I-V Curve with additional high Rs cell")
module.show()

#%%
# Resimulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cell Vint with additional high Rs cell",colour_bar=True)


# %% [markdown]
# # Tandem Cell Demo
# This notebook shows how to build and run a perovskite-silicon tandem cell circuit model.

#%%
# Let's use the previously saved silicon cell as bottom cell
# optionally, strip the series resistor of this subcell as the tandem cell itself has a lumped resistor
bottom_cell = cell.diode_branch.as_type(Cell,shape=cell.shape,area=cell.area)
# we reduce its JL since we expect it to receive less of the Sun's light when operated as bottom cell
bottom_cell.set_JL(19.0e-3)

# %% [markdown]
# ## Create a top cell

#%%
# These are short cuts to making a top cell (key: set Si_intrinsic_limit=False)
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
# Equivalent definitions: One can either do...
tandem_cell = (bottom_cell + top_cell).as_type(MultiJunctionCell)
# or....
tandem_cell = MultiJunctionCell([bottom_cell,top_cell])

#%%
tandem_cell.draw(display_value=True,title="Tandem Cell Model")

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

json_directory =  f"{THIS_DIR}/tandem measurement json files/"
sample_info = {"area":244.26,"bottom_cell_thickness":180e-4}

measurements = get_measurements(json_directory)
ref_cell_model, interactive_fit_dashboard = analyze_solar_cell_measurements(measurements,sample_info=sample_info,is_tandem=True)

# %% [markdown]
# # Draw best fit circuit representation
# Draw the resultant tandem cell model with the best fit parameters

#%%
ref_cell_model.draw(title="Tandem Cell with Best Fit Parameters",display_value=True)

# %% [markdown]
# # Interactive Dashboard

# Pop up an interactive dashboard where you can use sliders to change the tandem cell model
# parameter values and then see how well the resultant simulated measurement data match up
# with the experimental data

#%%
interactive_fit_dashboard.run()


# %% [markdown]
# # Topcon Cell Measurement Fitting Demo

# This example fits a single junction silicon cell model to the measurements of a large area
# Topcon silicon wafer solar cell.  Each measurement is stored in a json file 
# (see json_directory) in the format that PV_Circuit_Model Measurement class can read in.
# There are three kinds of measurements:
# Light I-V at 1 Sun, 0.5 Sun
# Dark I-V, Suns-Voc

#%%

json_directory = f"{THIS_DIR}/topcon measurement json files/"
sample_info = {"area":165.34,"bottom_cell_thickness":180e-4}

measurements = get_measurements(json_directory)

ref_cell_model, interactive_fit_dashboard = analyze_solar_cell_measurements(measurements,sample_info=sample_info,is_tandem=False,num_of_rounds=12)

# %% [markdown]
# # Draw best fit circuit representation
# Draw the resultant cell model with the best fit parameters

#%%
ref_cell_model.draw(title="Tandem Cell with Best Fit Parameters",display_value=True)

# %% [markdown]
# # Interactive Dashboard

# Pop up an interactive dashboard where you can use sliders to change the cell model
# parameter values and then see how well the resultant simulated measurement data match up
# with the experimental data

#%%
interactive_fit_dashboard.run()






