import numpy as np
from PV_Circuit_Model.utilities import ParameterSet, zero_C
from pathlib import Path

def get_ni(temperature: float) -> float:
    """Intrinsic carrier concentration as a function of temperature (°C)."""
    return 9.15e19*((temperature+zero_C)/300)**2*np.exp(-6880/(temperature+zero_C))

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

ParameterSet(name="wafer_formats",filename=PARAM_DIR / "wafer_formats.json")
wafer_formats = ParameterSet.get_set("wafer_formats")()

ParameterSet(name="silicon_constants",filename=PARAM_DIR / "silicon_constants.json")
silicon_constants = ParameterSet.get_set("silicon_constants")
bandgap_narrowing_RT = silicon_constants["bandgap_narrowing_RT"]
Jsc_fractional_temp_coeff = silicon_constants["Jsc_fractional_temp_coeff"]