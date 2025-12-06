import numpy as np
from PV_Circuit_Model.utilities import ParameterSet
from pathlib import Path

get_ni = lambda temperature: 9.15e19*((temperature+273.15)/300)**2*np.exp(-6880/(temperature+273.15))

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

ParameterSet(name="wafer_formats",filename=PARAM_DIR / "wafer_formats.json")()
wafer_formats = ParameterSet.get_set("wafer_formats")()

ParameterSet(name="silicon_constants",filename=PARAM_DIR / "silicon_constants.json")()
silicon_constants = ParameterSet.get_set("silicon_constants")
bandgap_narrowing_RT = silicon_constants["bandgap_narrowing_RT"]
Jsc_fractional_temp_coeff = silicon_constants["Jsc_fractional_temp_coeff"]