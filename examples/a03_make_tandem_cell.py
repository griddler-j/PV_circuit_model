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
                        thickness=200e-9, area=bottom_cell.area, 
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