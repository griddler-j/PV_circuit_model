# %% [markdown]
# # PV Module Demo
# This notebook shows how to build and run a circuit model of a PV module.

#%%
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *

np.random.seed(0)

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
module.draw_cells(title="Cells Vint with inhomogenity",colour_bar=True) 

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