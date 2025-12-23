# %% [markdown]
# # Solar Cell Circuit Demo
# This notebook shows how to build and run a silicon wafer solar cell circuit model.

#%%
from PV_Circuit_Model.circuit_model import IL, D1, D2, Dintrinsic_Si, Drev, R
from PV_Circuit_Model.device import Cell, wafer_shape
from PV_Circuit_Model.device_analysis import Cell_
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

# %% [markdown]
# ## A solar cell can be made of these circuit elements.  

#%%

# Notation: A | B means "connect A, B in parallel", and A + B means "connect A, B in series"
# IL(41e-3) = CurrentSource with IL = 41e-3A
# D1(10e-15) = ForwardDiode with I0 = 10e-15A, n=1
# D2(5e-9) = ForwardDiode with I0 = 5e-9A, n=2
# Dintrinsic_Si(180e-4) = Intrinsic_Si_diode in silicon with base thickness 180e-4 (doping, doping type set to default values)
# Drev(V_shift=10) = ReverseDiode with breakdown voltage 10V
# R(1e5), R(1/3) = Resistor(s) of 1e5ohm, 1/3ohm
circuit_group = ( 
    (IL(41e-3) | D1(10e-15) | D2(5e-9) | Dintrinsic_Si(180e-4) | Drev(V_shift=10) | R(1e5)) 
    + R(1/3)
)

circuit_group.get_Pmax() # this sets the operating point to MPP, so that the animation will proceed in the next draw step
circuit_group.draw(display_value=True,animate=True)
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
cell = Cell_(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, wafer_format="M10",half_cut=True)
cell.plot(title="Cell I-V Curve")
cell.show()
# save cell2 for next example
cell.dump(THIS_DIR / "cell.bson")

#%% 
# Verify that the cells defined these two ways have the same structure

print("Does cell_ and cell have the same structure? ", "Yes" if cell_.structure()==cell.structure() else "No")
