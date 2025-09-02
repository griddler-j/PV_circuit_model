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
