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
# ## A solar cell can be made of these circuit elements.  

#%%

# Note that A | B means "connect A, B in parallel", and A + B means "connect A, B in series"
circuit_group = (CurrentSource(41e-3) | ForwardDiode(I0=10e-15,n=1) | ForwardDiode(I0=5e-9,n=2) | Intrinsic_Si_diode(base_thickness=180e-4) | ReverseDiode(V_shift=10) | Resistor(cond=1/1e5)) + Resistor(3)
circuit_group.draw(display_value=True)
circuit_group.plot(title="Cell Parts I-V Curve")
circuit_group.show()

# %% [markdown]
# ## We can cast circuit_group as type Cell to give it additional shape and area

cell_ = circuit_group.as_type(Cell, **wafer_shape(format="M10",half_cut=True))
# Now cell has a shape and size that we can see
cell_.draw_cells()
# Also, plotting a cell will show the I-V curve with the current density multiplied by the cell area
cell_.plot(title="Cell I-V Curve")
cell_.show()

# %% [markdown]
# ## Because cells are frequently defined, if we want to be lazy, here's a short cut

#%%
# quick_solar_cell has the advantage that you can specify target I-V parameters for the diode parameters to tune to
cell = quick_solar_cell(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, wafer_format="M10",half_cut=True)
cell.plot(title="Cell I-V Curve")
cell.show()
# save cell2 for next example
cell.dump(THIS_DIR / "cell.bson")

#%% 
# Verify that the cells defined these two ways have the same structure

print("Does cell_ and cell have the same structure? ", "Yes" if cell_.structure()==cell.structure() else "No")
