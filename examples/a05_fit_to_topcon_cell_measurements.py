# %% [markdown]
# # Topcon Cell Measurement Fitting Demo

# This example fits a single junction silicon cell model to the measurements of a large area
# Topcon silicon wafer solar cell.  Each measurement is stored in a json file 
# (see json_directory) in the format that PV_Circuit_Model Measurement class can read in.
# There are three kinds of measurements:
# Light I-V at 1 Sun, 0.5 Sun
# Dark I-V, Suns-Voc

#%%

from PV_Circuit_Model.data_fitting_tandem_cell import analyze_solar_cell_measurements
from PV_Circuit_Model.measurement import get_measurements
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

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




