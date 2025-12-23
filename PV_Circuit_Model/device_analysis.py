import numpy as np
import PV_Circuit_Model.circuit_model as circuit
import PV_Circuit_Model.device as device_module
import PV_Circuit_Model.utilities as utilities
import matplotlib
from matplotlib import pyplot as plt
from contextlib import nullcontext
import matplotlib.ticker as mticker
import os
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from PV_Circuit_Model import __version__, __git_hash__, __git_date__, __dirty__
from datetime import datetime

try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
except Exception:
    tk = None  # headless / no-tk environment
    ScrolledText = None

# Treat CI or forced Agg as headless
HEADLESS = (
    os.environ.get("CI")  # GitHub Actions / other CI
    or os.environ.get("MPLBACKEND", "").lower() == "agg"
)

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None and hasattr(get_ipython(), "kernel")
    except Exception:
        return False

IN_NOTEBOOK = _in_notebook()
if not IN_NOTEBOOK and tk is not None and not HEADLESS:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        # Fall back silently if TkAgg isn't available
        pass

BASE_UNITS = {
    "Pmax": ("W",  "W"),
    "Vmp":  ("V",  "V"),
    "Imp":  ("A",  "A"),
    "Voc":  ("V",  "V"),
    "Isc":  ("A",  "A"),
    "FF":   ("%",  r"%"),
    "Eff":   ("%",  r"%"),
    "Area": ("cm²", r"cm^2"),
    "Jsc":  ("mA/cm²", r"mA/cm^2"),
    "Jmp":  ("mA/cm²", r"mA/cm^2"),
}

DISPLAY_DECIMALS = {
    "Pmax": (3,2),
    "Vmp":  (4,2),
    "Imp":  (3,3),
    "Voc":  (4,2),
    "Isc":  (3,3),
    "FF":   (3,3),
    "Eff":   (3,3),
    "Area": (3,3),
    "Jsc":  (3,3),
    "Jmp":  (3,3),
}

solver_env_variables = utilities.ParameterSet.get_set("solver_env_variables")
REFINE_V_HALF_WIDTH = solver_env_variables["REFINE_V_HALF_WIDTH"]


def get_Voc(argument: Union[circuit.CircuitGroup, np.ndarray], interpolation_method: int = 0) -> float:
    """Compute open-circuit voltage (Voc) from an IV curve or CircuitGroup.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (Union[CircuitGroup, np.ndarray]): CircuitGroup or IV curve.
        interpolation_method (int): 0 linear, 1 upper bound, 2 lower bound.

    Returns:
        float: Open-circuit voltage.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_Voc
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_Voc(cell)
        ```
    """
    if isinstance(argument,circuit.CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        
        if argument.IV_I.size==0 or argument.IV_I[-1] < 0: # can't reach OC
            diodes = argument.findElementType(circuit.Diode)
            while argument.IV_I.size==0 or argument.IV_I[-1] < 0: # can't reach OC
                for diode in diodes:
                    diode.max_I *= 10
                argument.null_all_IV()
                argument.build_IV()

        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    
    Voc = utilities.interp_(0,IV_curve[1,:],IV_curve[0,:])  

    if interpolation_method>0: 
        index = np.searchsorted(IV_curve[1,:], 0, side="right") - 1
        if index>0 and index+1<IV_curve.shape[1]-1:
            V_ = np.linspace(IV_curve[0,index],IV_curve[0,index+1],1000)
            I_ = utilities.interp_(V_,IV_curve[0,:],IV_curve[1,:])
            slopes = (IV_curve[1,index:index+3]-IV_curve[1,index-1:index+2])/(IV_curve[0,index:index+3]-IV_curve[0,index-1:index+2])
            left_slope = slopes[0]
            right_slope = slopes[2]
            this_slope = slopes[1]
            if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                I_ref_left = IV_curve[1,index] + (V_-IV_curve[0,index])*left_slope
                I_ref_right = IV_curve[1,index+1] + (V_-IV_curve[0,index+1])*right_slope
                I_ref_mid = utilities.interp_(V_,IV_curve[0,:],IV_curve[1,:])
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_ = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                else:
                    I_ = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))
            else:
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[:] = IV_curve[1,index-1]
                else: 
                    I_[:] = IV_curve[1,index]

            Voc = utilities.interp_(0,I_,V_)  

    return Voc
circuit.CircuitGroup.get_Voc = get_Voc

def get_Isc(argument: Union[circuit.CircuitGroup, np.ndarray], interpolation_method: int = 0) -> float:
    """Compute short-circuit current (Isc) from an IV curve or CircuitGroup.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (Union[CircuitGroup, np.ndarray]): CircuitGroup or IV curve.
        interpolation_method (int): 0 linear, 1 upper bound, 2 lower bound.

    Returns:
        float: Short-circuit current.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_Isc
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_Isc(cell)
        ```
    """
    if isinstance(argument,circuit.CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        IV_curve = argument.IV_table
        Isc = -utilities.interp_(0,IV_curve[0,:],IV_curve[1,:],argument.extrapolation_dI_dV[0],argument.extrapolation_dI_dV[1])
    else:
        IV_curve = argument
        Isc = -utilities.interp_(0,IV_curve[0,:],IV_curve[1,:])

    if interpolation_method>0: 
        index = np.searchsorted(IV_curve[0,:], 0, side="right") - 1
        if index>0 and index+1<IV_curve.shape[1]-1:
            V_ = np.linspace(IV_curve[0,index],IV_curve[0,index+1],1000)
            I_ = utilities.interp_(V_,IV_curve[0,:],IV_curve[1,:])
            slopes = (IV_curve[1,index:index+3]-IV_curve[1,index-1:index+2])/(IV_curve[0,index:index+3]-IV_curve[0,index-1:index+2])
            left_slope = slopes[0]
            right_slope = slopes[2]
            this_slope = slopes[1]
            if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                I_ref_left = IV_curve[1,index] + (V_-IV_curve[0,index])*left_slope
                I_ref_right = IV_curve[1,index+1] + (V_-IV_curve[0,index+1])*right_slope
                I_ref_mid = utilities.interp_(V_,IV_curve[0,:],IV_curve[1,:])
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_ = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                else:
                    I_ = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))
            else:
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[:] = IV_curve[1,index-1]
                else: 
                    I_[:] = IV_curve[1,index]
            Isc = -utilities.interp_(0,V_,I_)  

    return Isc
circuit.CircuitGroup.get_Isc = get_Isc

def get_Jsc(argument: circuit.CircuitGroup) -> float:
    """Compute short-circuit current density (Jsc).

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (CircuitGroup): CircuitGroup with IV data.

    Returns:
        float: Short-circuit current density.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_Jsc
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_Jsc(cell)
        ```
    """
    Jsc = argument.get_Isc()
    if hasattr(argument,"area"):
        Jsc /= argument.area
    return Jsc
circuit.CircuitGroup.get_Jsc = get_Jsc

# interpolation_method 0-linear, 1-upper bound, 2-lowerbound
def get_Pmax(
    argument: Union[circuit.CircuitGroup, np.ndarray],
    return_op_point: bool = False,
    refine_IV: bool = True,
    interpolation_method: int = 0,
) -> Union[float, Tuple[float, float, float]]:
    """Compute maximum power from an IV curve or group.

    Optionally refines IV around the operating point.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (Union[CircuitGroup, np.ndarray]): CircuitGroup or IV curve.
        return_op_point (bool): If True, return (Pmax, Vmp, Imp).
        refine_IV (bool): If True, refine IV around the peak.
        interpolation_method (int): 0 linear, 1 upper bound, 2 lower bound.

    Returns:
        Union[float, Tuple[float, float, float]]: Pmax or (Pmax, Vmp, Imp).

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_Pmax
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_Pmax(cell)
        ```
    """
    if isinstance(argument,circuit.CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        if argument.IV_I.size==0 or argument.IV_I[-1] < 0: # can't reach OC
            diodes = argument.findElementType(circuit.Diode)
            while argument.IV_I.size==0 or argument.IV_I[-1] < 0: # can't reach OC
                for diode in diodes:
                    diode.max_I *= 10
                argument.null_all_IV()
                argument.build_IV()
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    V_ = IV_curve[0,:]
    I_ = IV_curve[1,:]
    power = -V_*I_
    index = np.argmax(power)
    if not (index==0 or index==power.size-1):
        V_ = np.linspace(IV_curve[0,index-1],IV_curve[0,index+1],1000)
        I_ = utilities.interp_(V_,IV_curve[0,:],IV_curve[1,:])
        if interpolation_method>0 and index-1>0 and index+1<IV_curve.shape[1]-1:
            slopes = (IV_curve[1,index-1:index+3]-IV_curve[1,index-2:index+2])/(IV_curve[0,index-1:index+3]-IV_curve[0,index-2:index+2])
            find_ = np.where(V_ <= IV_curve[0,index])[0]
            left_slope = slopes[0]
            right_slope = slopes[2]
            this_slope = slopes[1]
            if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                I_ref_left = IV_curve[1,index-1] + (V_[find_]-IV_curve[0,index-1])*left_slope
                I_ref_right = IV_curve[1,index] + (V_[find_]-IV_curve[0,index])*right_slope
                I_ref_mid = utilities.interp_(V_[find_],IV_curve[0,:],IV_curve[1,:])
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[find_] = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                else:
                    I_[find_] = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))
            else:
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[find_] = IV_curve[1,index-1]
                else: 
                    I_[find_] = IV_curve[1,index]

            find_ = np.where(V_ > IV_curve[0,index])[0]
            left_slope = slopes[1]
            right_slope = slopes[3]
            this_slope = slopes[2]
            if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                I_ref_left = IV_curve[1,index] + (V_[find_]-IV_curve[0,index])*left_slope
                I_ref_right = IV_curve[1,index+1] + (V_[find_]-IV_curve[0,index+1])*right_slope
                I_ref_mid = utilities.interp_(V_[find_],IV_curve[0,:],IV_curve[1,:])
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[find_] = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                else:
                    I_[find_] = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))
            else:
                if interpolation_method==1: # get upper bound curve, meaning to go under
                    I_[find_] = IV_curve[1,index]
                else: 
                    I_[find_] = IV_curve[1,index+1]

        power = -V_*I_
        index = np.argmax(power)
    Vmp = V_[index]
    
    if isinstance(argument,circuit.CircuitGroup) and refine_IV and not argument.refined_IV:
        if not hasattr(argument,"job_heap"):
            argument.build_IV()
        argument.set_operating_point(V=Vmp, refine_IV=refine_IV)
        V_ = argument.IV_V
        I_ = argument.IV_I
        power = -V_*I_
        index = np.argmax(power)
        if not (index==0 or index==power.size-1):
            V_ = np.linspace(argument.IV_V[index-1],argument.IV_I[index+1],1000)
            I_ = utilities.interp_(V_,argument.IV_V,argument.IV_I)
            if interpolation_method>0 and index-1>0 and index+1<IV_curve.shape[1]-1:
                slopes = (IV_curve[1,index-1:index+3]-IV_curve[1,index-2:index+2])/(IV_curve[0,index-1:index+3]-IV_curve[0,index-2:index+2])
                find_ = np.where(V_ <= IV_curve[0,index])[0]
                left_slope = slopes[0]
                right_slope = slopes[2]
                this_slope = slopes[1]
                if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                    I_ref_left = IV_curve[1,index-1] + (V_[find_]-IV_curve[0,index-1])*left_slope
                    I_ref_right = IV_curve[1,index] + (V_[find_]-IV_curve[0,index])*right_slope
                    I_ref_mid = utilities.interp_(V_[find_],IV_curve[0,:],IV_curve[1,:])
                    if interpolation_method==1: # get upper bound curve, meaning to go under
                        I_[find_] = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                    else:
                        I_[find_] = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))

                find_ = np.where(V_ > IV_curve[0,index])[0]
                left_slope = slopes[1]
                right_slope = slopes[3]
                this_slope = slopes[2]
                if not (np.isnan(left_slope) or np.isinf(left_slope) or np.isnan(right_slope) or np.isinf(right_slope) or np.isnan(this_slope) or np.isinf(this_slope)):
                    I_ref_left = IV_curve[1,index-1] + (V_[find_]-IV_curve[0,index-1])*left_slope
                    I_ref_right = IV_curve[1,index] + (V_[find_]-IV_curve[0,index])*right_slope
                    I_ref_mid = utilities.interp_(V_[find_],IV_curve[0,:],IV_curve[1,:])
                    if interpolation_method==1: # get upper bound curve, meaning to go under
                        I_[find_] = np.minimum(I_ref_mid, np.maximum(I_ref_left,I_ref_right))
                    else:
                        I_[find_] = np.maximum(I_ref_mid, np.minimum(I_ref_left,I_ref_right))

            power = -V_*I_
            index = np.argmax(power)
    Vmp = V_[index]
    Imp = I_[index]
    Pmax = power[index]
    if isinstance(argument,circuit.CircuitGroup):
        argument.operating_point = [Vmp,Imp]

    if return_op_point:
        return Pmax, Vmp, Imp
    return Pmax
circuit.CircuitGroup.get_Pmax = get_Pmax

def get_Eff(argument: circuit.CircuitGroup) -> float:
    """Compute efficiency from maximum power and area.

    Uses `get_Pmax` and scales by area when available.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (CircuitGroup): CircuitGroup with IV data.

    Returns:
        float: Efficiency value.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_Eff
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_Eff(cell)
        ```
    """
    Eff = argument.get_Pmax()
    if hasattr(argument,"area"):
        Eff *= 10.0/argument.area
    return Eff
circuit.CircuitGroup.get_Eff = get_Eff

def get_FF(argument: Union[circuit.CircuitGroup, np.ndarray], interpolation_method: int = 0) -> float:
    """Compute fill factor from IV curve or group.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (Union[CircuitGroup, np.ndarray]): CircuitGroup or IV curve.
        interpolation_method (int): 0 linear, 1 upper bound, 2 lower bound.

    Returns:
        float: Fill factor value.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import get_FF
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        get_FF(cell)
        ```
    """
    Voc = get_Voc(argument,interpolation_method=interpolation_method)
    Isc = get_Isc(argument,interpolation_method=interpolation_method)
    Pmax = get_Pmax(argument,interpolation_method=interpolation_method)
    FF = Pmax/(Isc*Voc)
    return FF
circuit.CircuitGroup.get_FF = get_FF

def Rs_extraction_two_light_IVs(IV_curves: Sequence[np.ndarray]) -> float:
    """Estimate series resistance from two light IV curves.

    Uses the shift between full-sun and half-sun operating points.

    Args:
        IV_curves (Sequence[np.ndarray]): Two IV curves at different irradiance.

    Returns:
        float: Estimated series resistance.

    Example:
        ```python
        import numpy as np
        from PV_Circuit_Model.device_analysis import Rs_extraction_two_light_IVs
        IV1 = np.array([....])
        IV2 = np.array([....])
        Rs_extraction_two_light_IVs([IV1, IV2])
        ```
    """
    Isc0 = -1*get_Isc(IV_curves[0])
    Isc1 = -1*get_Isc(IV_curves[1])
    _, Vmp0, Imp0 = get_Pmax(IV_curves[0],return_op_point=True)
    delta_I = -Isc0+Imp0
    delta_Is_halfSun = -Isc1+IV_curves[1][1,:]
    V_point = np.interp(delta_I,delta_Is_halfSun,IV_curves[1][0,:])
    Rs = (Vmp0-V_point)/(Isc0-Isc1)
    return Rs

def Rshunt_extraction(IV_curve: np.ndarray, base_point: float = 0) -> float:
    """Estimate shunt resistance from the IV curve near a base point.

    Performs a local linear fit around the specified voltage region.

    Args:
        IV_curve (np.ndarray): IV curve array with shape (2, N).
        base_point (float): Voltage around which to estimate slope.

    Returns:
        float: Estimated shunt resistance.

    Example:
        ```python
        import numpy as np
        from PV_Circuit_Model.device_analysis import Rshunt_extraction
        IV = np.array([....])
        Rshunt_extraction(IV)
        ```
    """
    base_point = max(base_point,np.min(IV_curve[0,:]))
    indices = np.where((IV_curve[0,:]>=base_point) & (IV_curve[0,:]<=base_point+0.1))[0]
    indices = list(indices)
    if len(indices)<2 or abs(IV_curve[0,indices[-1]]-IV_curve[0,indices[0]])<0.01:
        indices1 = np.where(IV_curve[0,:]<=base_point)[0]
        indices = [indices1[-1]] + indices
        indices2 = np.where(IV_curve[0,:]>=base_point+0.1)[0]
        indices = indices + [indices2[0]]
    m, _ = np.polyfit(IV_curve[0,indices], IV_curve[1,indices], deg=1)
    if m <= 0:
        Rshunt = 100000
    else:
        Rshunt = 1/m
    Rshunt = min(Rshunt,100000)
    return Rshunt

def estimate_cell_J01_J02(
    Jsc: float,
    Voc: float,
    Pmax: Optional[float] = None,
    FF: float = 1.0,
    Rs: float = 0.0,
    Rshunt: float = 1e6,
    temperature: float = 25,
    Sun: float = 1.0,
    Si_intrinsic_limit: bool = True,
    **kwargs: Any,
) -> Tuple[float, float]:
    """Estimate J01 and J02 for a cell that matches target IV metrics.

    Iteratively adjusts diode saturation currents to match Voc and Pmax.

    Args:
        Jsc (float): Short-circuit current density.
        Voc (float): Open-circuit voltage.
        Pmax (Optional[float]): Target maximum power.
        FF (float): Target fill factor if Pmax is not provided.
        Rs (float): Series resistance.
        Rshunt (float): Shunt resistance.
        temperature (float): Temperature in Celsius.
        Sun (float): Irradiance multiplier.
        Si_intrinsic_limit (bool): If True, include intrinsic Si diode.
        **kwargs (Any): Extra parameters for intrinsic diode.

    Returns:
        Tuple[float, float]: Estimated (J01, J02) values.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import estimate_cell_J01_J02
        estimate_cell_J01_J02(Jsc=0.04, Voc=0.7, FF=0.8)
        ```
    """
    if Pmax is None:
        Pmax = Jsc*Voc*FF          
    VT = utilities.get_VT(temperature)
    max_J01 = Jsc/np.exp(Voc/VT)
    for inner_k in range(100):
        trial_cell = device_module.make_solar_cell(Jsc, max_J01, 0.0, Rshunt, 
                                     Rs, Si_intrinsic_limit=Si_intrinsic_limit, **kwargs)
        trial_cell.set_temperature(temperature)
        trial_cell.set_Suns(Sun)
        Voc_ = trial_cell.get_Voc()
        if abs(Voc_-Voc) < 1e-10:
            break 
        max_J01 *= np.exp((Voc_-Voc)/VT)
    max_J02 = Jsc/np.exp(Voc/(2*VT))
    for inner_k in range(100):
        trial_cell = device_module.make_solar_cell(Jsc, 0.0, max_J02, Rshunt, Rs, 
                                     Si_intrinsic_limit=Si_intrinsic_limit,**kwargs)
        trial_cell.set_temperature(temperature)
        trial_cell.set_Suns(Sun)
        Voc_ = trial_cell.get_Voc()
        if abs(Voc_-Voc) < 1e-10:
            break 
        max_J02 *= np.exp((Voc_-Voc)/(2*VT))
    outer_record = []
    for outer_k in range(100):
        if outer_k==0:
            trial_J01 = 0.0
        elif outer_k==1:
            trial_J01 = max_J01
        else:
            outer_record_ = np.array(outer_record)
            indices = np.argsort(outer_record_[:,0])
            outer_record_ = outer_record_[indices,:]
            trial_J01 = utilities.interp_(Pmax, outer_record_[:,1], outer_record_[:,0])
            trial_J01 = max(trial_J01, 0.0)
            trial_J01 = min(trial_J01, max_J01)
        inner_record = []
        for inner_k in range(100):
            if inner_k==0:
                trial_J02 = 0.0
                if outer_k==0 and not Si_intrinsic_limit:
                    trial_J02 = max_J02/2
            elif inner_k==1:
                trial_J02 = max_J02
            else:
                inner_record_ = np.array(inner_record)
                indices = np.argsort(inner_record_[:,1])
                inner_record_ = inner_record_[indices,:]
                trial_J02 = utilities.interp_(Voc, inner_record_[:,1], inner_record_[:,0])
                trial_J02 = max(trial_J02, 0.0)
                trial_J02 = min(trial_J02, max_J02)
            trial_cell = device_module.make_solar_cell(Jsc, trial_J01, trial_J02, Rshunt, Rs,
                                         Si_intrinsic_limit=Si_intrinsic_limit,**kwargs)
            trial_cell.set_temperature(temperature)
            trial_cell.set_Suns(Sun)
            Voc_ = trial_cell.get_Voc()
            if abs(Voc_-Voc) < 1e-10 or (trial_J02==0 and Voc_<Voc) or (trial_J02==max_J02 and Voc_>Voc):
                break 
            inner_record.append([trial_J02,Voc_])
        Pmax_ = trial_cell.get_Pmax()
        outer_record.append([trial_J01,Pmax_])
        if abs(Voc_-Voc)<1e-10 and abs(Pmax_-Pmax)/Pmax<1e-10:
            break
        if outer_k==1 and Pmax_ < Pmax: # will never be bigger then
            break
    return trial_J01, trial_J02

def get_IV_parameter_words(
    self: circuit.CircuitGroup,
    display_or_latex: int = 0,
    cell_or_module: int = 0,
    cap_decimals: bool = True,
    include_bounds: bool = False,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    words = {}
    curves = {"normal":np.array([self.IV_V,self.IV_I])}
    if hasattr(self,"IV_V_lower"):
        curves["lower"] = np.array([self.IV_V_lower,self.IV_I_lower])
        curves["upper"] = np.array([self.IV_V_upper,self.IV_I_upper])
    all_parameters = {}
    for key, _ in curves.items():
        interpolation_method = 0
        if key=="upper":
            interpolation_method = 1
        elif key=="lower":
            interpolation_method = 2
        all_parameters[key] = {}
        parameters = all_parameters[key]
        parameters["Pmax"], parameters["Vmp"], parameters["Imp"] = get_Pmax(curves[key],return_op_point=True,interpolation_method=interpolation_method)
        parameters["Imp"] *= -1
        parameters["Voc"] = get_Voc(curves[key],interpolation_method=interpolation_method)
        parameters["Isc"] = get_Isc(curves[key],interpolation_method=interpolation_method)
        parameters["FF"] = get_FF(curves[key],interpolation_method=interpolation_method)*100
        if hasattr(self,"area"):
            parameters["Jsc"] = parameters["Isc"]/self.area*1000
            parameters["Jmp"] = parameters["Imp"]/self.area*1000
            parameters["Area"] = self.area
            parameters["Eff"] = parameters["Pmax"]/self.area*1000
    for key, value in all_parameters["normal"].items():
        error_word = ""
        if hasattr(self,"IV_V_lower"):
            error_word = f" \u00B1 {0.5*abs(all_parameters['upper'][key]-all_parameters['lower'][key]):.1e}"
        if cap_decimals:
            decimals = DISPLAY_DECIMALS[key][cell_or_module]
        else:
            decimals = 12
        words[key] = f"{key} = {value:.{decimals}f}{error_word} {BASE_UNITS[key][display_or_latex]}"
        if hasattr(self,"IV_V_lower") and include_bounds:
            words[key] += f" (from lower bound curve: {all_parameters['lower'][key]:.{decimals}f} {BASE_UNITS[key][display_or_latex]}"
            words[key] += f", from upper bound curve: {all_parameters['upper'][key]:.{decimals}f} {BASE_UNITS[key][display_or_latex]})"
    return words, parameters

def plot(
    self: circuit.CircuitComponent,
    fourth_quadrant: bool = True,
    show_IV_parameters: bool = True,
    title: str = "I-V Curve",
    show_solver_summary: bool = False,
) -> None:
    """Plot the IV curve and optionally annotate key parameters.

    Produces a matplotlib figure with current and power axes when requested.

    Warning:
        This function is monkey-patched onto `CircuitComponent` at import time.

    Args:
        self (CircuitComponent): Component with IV data.
        fourth_quadrant (bool): If True, plot in power-generating quadrant.
        show_IV_parameters (bool): If True, annotate IV metrics.
        title (str): Figure window title.
        show_solver_summary (bool): If True, open solver summary window.

    Returns:
        None

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import plot
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        plot(cell)
        ```
    """
    ctx = plt.ioff() if not show_solver_summary else nullcontext()
    with ctx:
        if self.IV_V is None:
            self.build_IV()
        if (fourth_quadrant or show_IV_parameters) and isinstance(self,circuit.CircuitGroup):
            _, Vmp, Imp = self.get_Pmax(return_op_point=True)
            Voc = self.get_Voc()
            Isc = self.get_Isc()

        find_near_op = None

        IV_V = self.IV_V
        IV_I = self.IV_I
        
        fig = plt.figure("IV")
        if fourth_quadrant and isinstance(self,circuit.CircuitGroup):
            if len(fig.axes) < 2:
                fig.clf()
                ax1 = fig.add_subplot(111)
                ax2 = ax1.twinx()
            else:
                ax1, ax2 = fig.axes[0], fig.axes[1]

            # Left Y-axis
            ax1.plot(IV_V,-IV_I)
            if find_near_op is not None:
                ax1.plot(IV_V[find_near_op],-IV_I[find_near_op],color="red")
            if self.operating_point is not None:
                ax1.plot(self.operating_point[0],-self.operating_point[1],marker='o',color="blue")
            ax1.set_xlim((0,Voc*1.1))
            ax1.set_ylim((0,Isc*1.1))
            ax1.set_xlabel("Voltage (V)")
            ax1.set_ylabel("Current (A)")

            P = -IV_V*IV_I
            # Right Y-axis (shares same X)
            ax2.plot(IV_V,P,color="orange")
            if find_near_op is not None:
                ax2.plot(IV_V[find_near_op],P[find_near_op],color="red")
            if self.operating_point is not None:
                ax2.plot(self.operating_point[0],-self.operating_point[0]*self.operating_point[1],marker='o',color="orange")
            ax2.set_ylim((0,np.max(P)*1.1))
            ax2.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: f"{x:.0e}")
            )
            ax2.set_ylabel("Power (W)")
            if show_IV_parameters and fourth_quadrant and isinstance(self,circuit.CircuitGroup):
                cell_or_module=1
                params = ["Isc","Voc","FF","Pmax"]
                if isinstance(self,device_module.Cell) or isinstance(self,device_module.MultiJunctionCell): # cell or MJ cell
                    cell_or_module=0
                    params = ["Isc","Jsc","Voc","FF","Pmax","Eff","Area"]
                words, _ = get_IV_parameter_words(self, display_or_latex=0, cell_or_module=cell_or_module, cap_decimals=True)
            
                y_space = 0.07
                ax1.plot(Voc,0,marker='o',color="blue")
                ax1.plot(0,Isc,marker='o',color="blue")
                if fourth_quadrant:
                    Imp *= -1
                ax1.plot(Vmp,Imp,marker='o',color="blue")
                ax2.plot(Vmp,Imp*Vmp,marker='o',color="orange")
                for i, param in enumerate(params):
                    ax1.text(Voc*0.05, Isc*(0.8-i*y_space), words[param])
            fig.tight_layout()

            if show_solver_summary:
                self.show_solver_summary(fig=fig)

        else:
            ax1 = fig.axes[0] if fig.axes else fig.add_subplot(111)
            ax1.plot(IV_V,IV_I)
            if find_near_op is not None:
                ax1.plot(IV_V[find_near_op],IV_I[find_near_op],color="red")
            if self.operating_point is not None:
                ax1.plot(self.operating_point[0],self.operating_point[1],marker='o')
            ax1.set_xlabel("Voltage (V)")
            ax1.set_ylabel("Current (A)")
            
        fig.canvas.manager.set_window_title(title)
circuit.CircuitComponent.plot = plot

def show(self: circuit.CircuitComponent) -> None:
    """Show the IV figure in notebook or GUI environments.

    In notebooks, the figure is non-blocking; otherwise it blocks until closed.

    Warning:
        This function is monkey-patched onto `CircuitComponent` at import time.

    Args:
        self (CircuitComponent): Component with plotted IV data.

    Returns:
        None

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import show
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        cell.plot()
        cell.show()
        ```
    """
    # In notebooks, figures are auto-shown; don't block
    if IN_NOTEBOOK:
        plt.show(block=False)   # or even just `return`
    else:
        plt.show()
    plt.close("IV")
circuit.CircuitComponent.show = show

def quick_solar_cell(
    Jsc: float = 0.042,
    Voc: float = 0.735,
    FF: float = 0.82,
    Rs: float = 0.3333,
    Rshunt: float = 1e6,
    wafer_format: str = "M10",
    half_cut: bool = True,
    **kwargs: Any,
) -> device_module.Cell:
    """Create a quick solar cell model from target IV parameters.

    Uses the estimator to derive J01/J02 and constructs a cell.

    Args:
        Jsc (float): Target short-circuit current density.
        Voc (float): Target open-circuit voltage.
        FF (float): Target fill factor.
        Rs (float): Series resistance.
        Rshunt (float): Shunt resistance.
        wafer_format (str): Wafer format key for geometry.
        half_cut (bool): If True, use half-cut geometry.
        **kwargs (Any): Extra parameters forwarded to `make_solar_cell`.

    Returns:
        Cell: Constructed cell instance.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import quick_solar_cell
        cell = quick_solar_cell(Jsc = 0.04, Voc = 0.7, FF = 0.81)
        ```
    """
    J01, J02 = estimate_cell_J01_J02(Jsc,Voc,FF=FF,Rs=Rs,Rshunt=Rshunt,**kwargs)
    return device_module.make_solar_cell(Jsc, J01, J02, Rshunt, Rs, **device_module.wafer_shape(format=wafer_format,half_cut=half_cut), **kwargs)

def Cell_(*args, **kwargs):
    """
    This is a shorthand for `quick_solar_cell`.
    """
    return quick_solar_cell(*args, **kwargs)

def quick_module(
    Isc: Optional[float] = None,
    Voc: Optional[float] = None,
    FF: Optional[float] = None,
    Pmax: Optional[float] = None,
    wafer_format: str = "M10",
    num_strings: int = 3,
    num_cells_per_halfstring: int = 24,
    special_conditions: Optional[Dict[str, Any]] = None,
    half_cut: bool = False,
    butterfly: bool = False,
    **kwargs: Any,
) -> device_module.Module:
    """Create a module from target IV parameters.

    Builds cells and tunes parameters to match target Pmax or FF.

    Args:
        Isc (Optional[float]): Target short-circuit current.
        Voc (Optional[float]): Target open-circuit voltage.
        FF (Optional[float]): Target fill factor.
        Pmax (Optional[float]): Target maximum power.
        wafer_format (str): Wafer format key for geometry.
        num_strings (int): Number of parallel strings.
        num_cells_per_halfstring (int): Cells per half-string.
        special_conditions (Optional[Dict[str, Any]]): Special options (e.g., "force_n1").
        half_cut (bool): If True, use half-cut geometry.
        butterfly (bool): If True, use butterfly layout.
        **kwargs (Any): Extra parameters forwarded to cell creation.

    Returns:
        Module: Constructed module instance.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import quick_module
        module = quick_module(Isc = 13, Voc = 72*0.7, FF = 0.79)
        ```
    """
    force_n1 = False
    if special_conditions is not None:
        if "force_n1" in special_conditions:
            force_n1 = special_conditions["force_n1"]
    area = device_module.wafer_shape(format=wafer_format, half_cut=half_cut)["area"]
    Jsc = 0.042
    cell_num_factor = 1
    if butterfly:
        cell_num_factor = 2
    if Isc is not None:
        Jsc = Isc / area /cell_num_factor
    else:
        Isc = Jsc * area * cell_num_factor 
    cell_Voc = 0.735
    if Voc is not None:
        cell_Voc = Voc / (num_strings*num_cells_per_halfstring)
    else:
        Voc = cell_Voc * (num_strings*num_cells_per_halfstring)
    target_Pmax = 0.8*Voc*Isc
    if Pmax is not None:
        target_Pmax = Pmax
    elif FF is not None:
        target_Pmax = Voc*Isc*FF
    if force_n1: # vary the module Rs
        cell = quick_solar_cell(Jsc=Jsc, Voc=cell_Voc, FF=1.0, wafer_format=wafer_format,half_cut=half_cut,**kwargs)
        cells = [circuit.circuit_deepcopy(cell) for _ in range(cell_num_factor*num_strings*num_cells_per_halfstring)]
        try_R = 0.02
        record = []
        for _ in tqdm(range(20),desc="Tweaking module cell parameters..."):
            module = device_module.make_module(cells, num_strings=num_strings, num_cells_per_halfstring=num_cells_per_halfstring, halfstring_resistor = try_R, butterfly=butterfly)
            module.set_Suns(1.0)
            module.build_IV()
            Pmax = module.get_Pmax()
            record.append([try_R, Pmax, cell.get_Pmax()])
            if np.abs(Pmax-target_Pmax) < 1e-6:
                break
            record_ = np.array(record)
            record_ = record_[record_[:, 1].argsort()]
            if np.max(record_[:,1])>=target_Pmax and np.min(record_[:,1])<=target_Pmax:
                try_R = utilities.interp_(target_Pmax, record_[:,1], record_[:,0])
            elif np.max(record_[:,1])<target_Pmax:
                try_R /= 10
            else:
                try_R *= 10
    else:
        try_FF = target_Pmax/Isc/Voc
        record = []
        for _ in tqdm(range(20),desc="Tweaking module cell parameters..."):
            cell = quick_solar_cell(Jsc=Jsc, Voc=cell_Voc, FF=try_FF, wafer_format=wafer_format,half_cut=half_cut,**kwargs)
            cells = [circuit.circuit_deepcopy(cell) for _ in range(cell_num_factor*num_strings*num_cells_per_halfstring)]
            module = device_module.make_module(cells, num_strings=num_strings, num_cells_per_halfstring=num_cells_per_halfstring, butterfly=butterfly)
            module.set_Suns(1.0)
            module.build_IV()
            Pmax = module.get_Pmax()
            record.append([try_FF, Pmax, cell.get_Pmax()])
            if np.abs(Pmax-target_Pmax) < 1e-6:
                break
            record_ = np.array(record)
            record_ = record_[record_[:, 0].argsort()]
            if len(record)>1:
                find_ = np.where(record_[1:, 1]<record_[:-1, 1])[0]
                if len(find_)>0:
                    break
            if np.max(record_[:,1])>=target_Pmax and np.min(record_[:,1])<=target_Pmax:
                try_FF = np.interp(target_Pmax, record_[:,1], record_[:,0])
            else:
                try_FF += 2*(target_Pmax - Pmax)/cell_Voc/Isc
    return module

def Module_(*args, **kwargs):
    """
    This is a shorthand for `quick_solar_cell`.
    """
    return quick_module(*args, **kwargs)

def quick_butterfly_module(
    Isc: Optional[float] = None,
    Voc: Optional[float] = None,
    FF: Optional[float] = None,
    Pmax: Optional[float] = None,
    wafer_format: str = "M10",
    num_strings: int = 3,
    num_cells_per_halfstring: int = 24,
    special_conditions: Optional[Dict[str, Any]] = None,
    half_cut: bool = True,
    **kwargs: Any,
) -> device_module.Module:
    """Create a butterfly module from target IV parameters.

    This is a convenience wrapper around `quick_module`.

    Args:
        Isc (Optional[float]): Target short-circuit current.
        Voc (Optional[float]): Target open-circuit voltage.
        FF (Optional[float]): Target fill factor.
        Pmax (Optional[float]): Target maximum power.
        wafer_format (str): Wafer format key for geometry.
        num_strings (int): Number of parallel strings.
        num_cells_per_halfstring (int): Cells per half-string.
        special_conditions (Optional[Dict[str, Any]]): Special options.
        half_cut (bool): If True, use half-cut geometry.
        **kwargs (Any): Extra parameters forwarded to cell creation.

    Returns:
        Module: Constructed module instance.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import quick_butterfly_module
        module = quick_butterfly_module(Isc = 13, Voc = 72*0.7, FF = 0.79)
        module.connection
        ```
    """
    return quick_module(Isc, Voc, FF, Pmax, wafer_format, num_strings, num_cells_per_halfstring, special_conditions, half_cut, butterfly=True,**kwargs)

def quick_tandem_cell(
    Jscs: Sequence[float] = (0.019, 0.020),
    Vocs: Sequence[float] = (0.710, 1.2),
    FFs: Sequence[float] = (0.8, 0.78),
    Rss: Sequence[float] = (0.3333, 0.5),
    Rshunts: Sequence[float] = (1e6, 5e4),
    thicknesses: Sequence[float] = (160e-4, 1e-6),
    wafer_format: str = "M10",
    half_cut: bool = True,
) -> device_module.MultiJunctionCell:
    """Create a quick tandem (multi-junction) cell model.

    Builds subcells from target metrics and combines them in series.

    Args:
        Jscs (Sequence[float]): Target short-circuit current densities.
        Vocs (Sequence[float]): Target open-circuit voltages.
        FFs (Sequence[float]): Target fill factors.
        Rss (Sequence[float]): Series resistances per subcell.
        Rshunts (Sequence[float]): Shunt resistances per subcell.
        thicknesses (Sequence[float]): Subcell thicknesses.
        wafer_format (str): Wafer format key for geometry.
        half_cut (bool): If True, use half-cut geometry.

    Returns:
        MultiJunctionCell: Constructed multi-junction cell.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import quick_tandem_cell
        cell = quick_tandem_cell()
        ```
    """
    cells = []
    for i in range(len(Jscs)):
        Si_intrinsic_limit = True
        if i > 0:
            Si_intrinsic_limit = False
        J01, J02 = estimate_cell_J01_J02(Jscs[i],Vocs[i],FF=FFs[i],Rs=Rss[i],Rshunt=Rshunts[i],Si_intrinsic_limit=Si_intrinsic_limit,thickness=thicknesses[i])
        cells.append(device_module.make_solar_cell(Jscs[i], J01, J02, Rshunts[i], Rss[i], **device_module.wafer_shape(format=wafer_format, half_cut=half_cut), thickness=thicknesses[i]))
    return device_module.MultiJunctionCell(cells)

def solver_summary_heap(job_heap: Any, display_or_latex: int = 0) -> str:
    build_time = job_heap.timers["build"]
    IV_time = job_heap.timers["IV"]
    refine_time = job_heap.timers["refine"]
    bounds_time = job_heap.timers["bounds"]
    component = job_heap.components[0]
    if component.refined_IV:
        paragraph = "I-V Parameters:\n"
    else:
        paragraph = "I-V Parameters (coarse - run get_Pmax() to get refinement!):\n"
    cell_or_module=1
    params = ["Isc","Imp","Voc","Vmp","FF","Pmax"]
    if isinstance(component,device_module.Cell) or isinstance(component,device_module.MultiJunctionCell):
        cell_or_module=0
        params = ["Isc","Jsc","Imp","Jmp","Voc","Vmp","FF","Pmax","Eff","Area"]
    words, _ = get_IV_parameter_words(component, display_or_latex=display_or_latex, cell_or_module=cell_or_module, cap_decimals=False, include_bounds=True)
    for param in params:
        paragraph += words[param]
        paragraph += "\n"
    paragraph += "----------------------------------------------------------------------------\n"
    if component.operating_point is not None:
        paragraph += "Operating Point:\n"
        paragraph += f"V = {component.operating_point[0]:.6f} V, I = {-component.operating_point[1]:.6f} A\n"
        paragraph += "----------------------------------------------------------------------------\n"
    if hasattr(component,"bottom_up_operating_point"):
        paragraph += "Calculation Error of Operating Point:\n"
        worst_V_error, worst_I_error = job_heap.calc_Kirchoff_law_errors()
        paragraph += f"Kirchhoff's Voltage Law deviation: V error <= {worst_V_error:.3e} V\n"
        paragraph += f"Kirchhoff's Current Law deviation: I error <= {worst_I_error:.3e} A\n"
        paragraph += "----------------------------------------------------------------------------\n"
    paragraph += "Calculation Times:\n"
    total_time = build_time + IV_time
    paragraph += f"Build: {build_time:.6f}s\n"
    paragraph += f"I-V curve stacks: {IV_time:.6f}s\n"
    if hasattr(component,"bottom_up_operating_point"):
        total_time += refine_time
        paragraph += f"Refinement around operating point: {refine_time:.6f}s\n"
    if hasattr(component,"IV_V_lower"):
        total_time += bounds_time
        paragraph += f"Uncertainty Calculations: {bounds_time:.6f}s\n"
    paragraph += f"Total: {total_time:.6f}s\n"

    return paragraph

def solver_summary(self: circuit.CircuitGroup, display_or_latex: int = 0) -> str:
    """Generate a solver summary for a circuit group.

    Reports IV metrics, solver timings, and environment settings.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        self (CircuitGroup): CircuitGroup with solver data.
        display_or_latex (int): 0 for display, 1 for LaTeX units.

    Returns:
        str: Formatted solver summary text.

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import solver_summary
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        cell.build_IV()
        print(cell.solver_summary())
        ```
    """
    __now__ = datetime.now().astimezone().replace(microsecond=0).isoformat()
    paragraph = "----------------------------------------------------------------------------\n"
    paragraph += "I-V Solver Summary for "
    if getattr(self,"name",None) is not None and self.name != "":
        paragraph += f"{self.name} of "
    paragraph += f" type {type(self).__name__}:\nReported on {__now__}\n"
    paragraph += "----------------------------------------------------------------------------\n"
    if hasattr(self,"job_heap") and self.IV_V is not None:
        paragraph += solver_summary_heap(self.job_heap,display_or_latex=display_or_latex)
    else:
        paragraph += "I-V Curve has not been calculated\n"
    paragraph += "----------------------------------------------------------------------------\n"
    paragraph += "CircuitGroup Information:\n"
    paragraph += f"Circuit Depth: {self.circuit_depth}\n"
    paragraph += f"Number of Circuit Elements: {self.num_circuit_elements}\n"
    paragraph += "----------------------------------------------------------------------------\n"
    paragraph += "Solver Environment Variables:\n"
    solver_env_variables_dict = utilities.ParameterSet.get_set("solver_env_variables")()
    for key, value in solver_env_variables_dict.items():
        paragraph += f"{key}: {value}\n"
    paragraph += "----------------------------------------------------------------------------\n"
    paragraph += f"PV Circuit Model Version: {__version__}\n"
    paragraph += f"git commit: {__git_hash__}\n"
    paragraph += f"git commit date: {__git_date__}\n"
    paragraph += f"git commit dirty: {__dirty__}\n"
    paragraph += "----------------------------------------------------------------------------\n"

    return paragraph

def save_solver_summary(self: circuit.CircuitGroup, filepath: str) -> None:
    """Save solver summary text to a file.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        self (CircuitGroup): CircuitGroup with solver data.
        filepath (str): Output file path.

    Returns:
        None

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import save_solver_summary
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        cell.build_IV()
        cell.save_solver_summary("summary.txt")
        ```
    """
    text = self.solver_summary(display_or_latex=1)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

def save_IV_curve(argument: Union[circuit.CircuitComponent, np.ndarray], filepath: str) -> None:
    """Save an IV curve to a tab-delimited text file.

    Warning:
        This function is monkey-patched onto `CircuitGroup` at import time.

    Args:
        argument (Union[CircuitComponent, np.ndarray]): Component or IV array.
        filepath (str): Output file path.

    Returns:
        None

    Example:
        ```python
        from PV_Circuit_Model.device_analysis import save_IV_curve
        from PV_Circuit_Model.device import make_solar_cell
        cell = make_solar_cell()
        cell.build_IV()
        cell.save_IV_curve("iv.txt")
        ```
    """
    if not isinstance(argument,circuit.CircuitComponent):
        V_col = argument[0,:]
        I_col = argument[1,:]
    elif argument.IV_V is not None:
        V_col = argument.IV_V
        I_col = argument.IV_I
    else:
        return
    with open(filepath, "w") as f:
        f.write("V(V)\tI(A)\n")
        for V_, I_ in zip(V_col, I_col):
            f.write(f"{V_:.17e}\t{I_:.17e}\n")

def show_solver_summary(self: circuit.CircuitGroup, fig: Optional[Any] = None) -> None:
    text = self.solver_summary()
    if IN_NOTEBOOK:
        print(text)
        return
    
    """
    If fig is provided:
        - Shows Matplotlib figure + Tk summary window
        - Closing either closes both

    If fig is None:
        - Shows only the Tk summary window
        - No Matplotlib involved
    """
    if fig is None:
        root = tk.Tk()
        root.title("I-V Solver Summary")
        root.geometry("720x600")

        text_box = ScrolledText(
            root,
            wrap="word",
            bg="white",
            fg="black",
            font=("Consolas", 11)
        )
        text_box.pack(expand=True, fill="both", padx=10, pady=10)

        text_box.insert("1.0", text)
        text_box.configure(state="disabled")

        root.mainloop()
        return
    

    manager = plt.get_current_fig_manager()
    root = manager.window    

    # Create a non-blocking popup
    win = tk.Toplevel(root)
    win.title("I-V Solver Summary")
    win.geometry("720x600")
    win.configure(bg="white")

    text_box = ScrolledText(
        win,
        wrap="word",
        bg="white",
        fg="black",
        font=("Consolas", 11)
    )
    text_box.pack(expand=True, fill="both", padx=10, pady=10)

    text_box.insert("1.0", text)
    text_box.configure(state="disabled")  # read-only

    def close_all():
        plt.close(fig)   # closes the Matplotlib figure
        try:
            root.destroy()
        except tk.TclError:
            pass  # already closed

    # Close button on the Matplotlib figure window
    root.protocol("WM_DELETE_WINDOW", close_all)
    # Close button on the summary window
    win.protocol("WM_DELETE_WINDOW", close_all)

    # Bring the Matplotlib window to the front *after* the event loop starts
    def bring_fig_to_front():
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(150, lambda: root.attributes("-topmost", False))
            root.focus_force()
        except tk.TclError:
            pass

    root.after(0, bring_fig_to_front)

circuit.CircuitGroup.solver_summary = solver_summary
circuit.CircuitGroup.show_solver_summary = show_solver_summary
circuit.CircuitGroup.save_solver_summary = save_solver_summary
circuit.CircuitGroup.save_IV_curve = save_IV_curve
