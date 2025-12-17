# %% [markdown]
# # Tandem Cell Demo
# This notebook shows how to build and run a perovskite-silicon tandem cell circuit model.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.cell_analysis import *
from PV_Circuit_Model.multi_junction_cell import *
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

# %% [markdown]
# ## Let's use the previously saved silicon cell as bottom cell

#%%
cell = Artifact.load(THIS_DIR / "cell.bson")
# optionally, strip the series resistor of this subcell as the tandem cell itself has a lumped resistor
bottom_cell = cell.diode_branch.as_type(Cell,shape=cell.shape,area=cell.area)
# we reduce its JL since we expect it to receive less of the Sun's light when operated as bottom cell
bottom_cell.set_JL(19.0e-3)

# %% [markdown]
# ## Create a top cell (use short cut)

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
# Equivalent definitions: One can either do...
tandem_cell = (bottom_cell + top_cell).as_type(MultiJunctionCell)
# or....
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