# %% [markdown]
# # PV Module Demo
# This notebook shows how to build and run a circuit model of a PV module.

#%%
from PV_Circuit_Model.circuit_model import R
from PV_Circuit_Model.device import Module, Device, Dbypass
from PV_Circuit_Model.device_analysis import Module_
from PV_Circuit_Model.utilities import Artifact
import numpy as np
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

np.random.seed(0)

# %% [markdown]
# ## Let's put 24 x 2 x 3 = 144 cells together to make a module

cell = Artifact.load(THIS_DIR / "cell.bson")

# A*24 = A + A .... + A = connect 24 copies of A's together in series
# tile_subgroups is optional to arrange the cells spatially, for ease of visualization
half_string = (cell*24 + R(0.05)).tile_subgroups(cols=2,x_gap=0.1,y_gap=0.1,turn=True)

# B**2 = B | B = connect 2 copies of B's together in parallel
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
section = (half_string**2 | Dbypass()).tile_subgroups(cols = 1, y_gap = 1, yflip=True)

# C*3 = C + C + C = connect 3 copies of C's together in series
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
circuit_group = (section*3).tile_subgroups(rows=1, x_gap = 1)

# type cast to Module just for encapsulation
module_ = circuit_group.as_type(Module) 
module_.draw_cells()

module_.plot(title="Module I-V Curve")
module_.show()

# %% [markdown]
# ## Because modules are frequently defined, if we want to be lazy, here's a short cut

module = Module_(Isc=14, Voc=0.72*72, FF=0.8, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, half_cut=True, butterfly=True)
module.plot(title="Module I-V Curve")
module.show()

#%% 
# Verify that the modules defined these two ways have the same structure

print("Does module_ and module have the same structure? ", "Yes" if module_.structure()==module.structure() else "No")

# %% [markdown]
# ## Introduce some cells JL and J01 inhomogenity

#%%
for cell in module.cells:
    cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.01)))
    cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
module.build_IV()
module.plot(title="Module I-V Curve with inhomogenity")
module.show()

# %% [markdown]
# ## Simulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias

#%%
module.set_Suns(0.0) 
module.set_operating_point(I=10)
module.draw_cells(title="Cells Vint with inhomogenity",colour_bar=True) 

# %% [markdown]
# ## Introduce high series resistance to cell #1 inside the module 

#%%
module.cells[0].set_specific_Rs(40.0)
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

string = module*26
print(type(string))
string = string.as_type(Device,name="string")
block = string + string.clone() + string.clone()

print(len(block.parts))
print(len(string.parts))