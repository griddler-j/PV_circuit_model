import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import PV_Circuit_Model.utilities as utilities
import PV_Circuit_Model.measurement as measurement_module
import PV_Circuit_Model.device as device_module
import inspect
import numbers
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    tk = None          # headless / no-tk environment
    ttk = None
    FigureCanvasTkAgg = None

from types import SimpleNamespace
from joblib import Parallel, delayed
from pathlib import Path
try:
    from PV_Circuit_Model.ivkernel import set_parallel_mode, get_parallel_mode
except Exception as e:
    raise ImportError(
        "Failed to import the compiled extension 'PV_Circuit_Model.ivkernel'.\n"
        "This usually means the C++/Cython extension was not built correctly.\n\n"
        "Try:\n"
        "  pip install -e .\n"
        "or (if installing from source):\n"
        "  pip install PV_Circuit_Model\n\n"
        "If you are on Windows, ensure Visual Studio Build Tools are installed.\n"
    ) from e

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None and hasattr(get_ipython(), "kernel")
    except Exception:
        return False

try:
    # only available in notebooks
    from IPython.display import display as _ip_display
except Exception:
    _ip_display = None

class Fit_Parameter(utilities.Artifact):
    """Single fit parameter with constraints and scaling.

    Tracks nominal values, bounds, and linear/log scaling for optimization.

    Args:
        name (str): Parameter name.
        value (float): Current parameter value.
        nominal_value (Optional[float]): Nominal value for regularization.
        d_value (Optional[float]): Differential step for sensitivity.
        abs_min (float): Absolute minimum bound.
        abs_max (float): Absolute maximum bound.
        is_log (bool): If True, parameter is stored in log10 space.

    Returns:
        Fit_Parameter: The constructed fit parameter.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import Fit_Parameter
        param = Fit_Parameter(name="Rs", value=0.1)
        param.get_parameter()
        ```
    """
    def __init__(
        self,
        name: str = "variable",
        value: float = 0.0,
        nominal_value: Optional[float] = None,
        d_value: Optional[float] = None,
        abs_min: float = -np.inf,
        abs_max: float = np.inf,
        is_log: bool = False,
    ) -> None:
        self.name = name
        self.value = value
        if nominal_value is None:
            self.nominal_value = value
        else:
            self.nominal_value = nominal_value
        self.is_log = is_log
        self.abs_min = abs_min
        self.abs_max = abs_max
        self.set_d_value(d_value)
        self.this_min = -np.inf
        self.this_max = np.inf
        self.enabled = True
        self.is_differential = False
        self.aux = {}
    def set_nominal(self) -> None:
        self.nominal_value = self.value
    def get_parameter(self) -> float:
        value_ = self.value
        if self.is_differential:
            value_ += self.d_value
        if self.is_log:
            return 10**(value_)
        else:
            return value_
    def set_d_value(self, d_value: Optional[float] = None) -> None:
        if d_value is not None:
            self.d_value = d_value
        else:
            if self.is_log:
                self.d_value = np.log10(2)
            else:
                self.d_value = self.value / 100
    def limit_order_of_mag(self, order_of_mag: float = 1.0) -> None:
        self.aux["limit_order_of_mag"] = order_of_mag
        if self.is_log:
            self.this_min = self.value - order_of_mag
            self.this_max = self.value + order_of_mag
        else:
            self.this_min = self.value/10**(order_of_mag)
            self.this_max = self.value*10**(order_of_mag)
    def limit_delta(self, delta: float) -> None:
        self.this_min = self.value - delta
        self.this_max = self.value + delta
    def get_min(self) -> float:
        return max(self.this_min,self.abs_min)       
    def get_max(self) -> float:
        return min(self.this_max,self.abs_max)    
    def check_max_min(self) -> None:
        self.nominal_value = max(self.nominal_value,self.abs_min)    
        self.nominal_value = min(self.nominal_value,self.abs_max)  
        self.value = max(self.value,self.abs_min)    
        self.value = min(self.value,self.abs_max)  

class Fit_Parameters(utilities.Artifact):
    """Collection of Fit_Parameter objects with helper utilities.

    Supports enabling/disabling parameters, bounds, and differential mode.

    Args:
        fit_parameters (Optional[List[Fit_Parameter]]): Predefined parameters.
        names (Optional[Sequence[str]]): Names to create default parameters.

    Returns:
        Fit_Parameters: The constructed parameter collection.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import Fit_Parameters
        params = Fit_Parameters(names=["Rs", "Rsh"])
        params.num_of_parameters()
        ```
    """
    _parent_pointer_name = "ref_sample"
    _parent_pointer_class = device_module.Device
    _critical_fields = utilities.Artifact._critical_fields + ("fit_parameters",)
    def __init__(
        self,
        fit_parameters: Optional[List[Fit_Parameter]] = None,
        names: Optional[Sequence[str]] = None,
    ) -> None:
        if fit_parameters is not None:
            self.fit_parameters = fit_parameters
        elif names is not None:
            self.fit_parameters = [Fit_Parameter(name=name) for name in names]
        else:
            self.fit_parameters = []
        self.is_differential = False
        self.ref_sample = None
        self.aux = {}
    def initialize_from_sample(self, sample: Any) -> None:
        """Set a reference sample for parameter application.

        Args:
            sample (Any): Reference object used during fitting.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters()
            params.initialize_from_sample(sample=None)
            params.ref_sample is None
            ```
        """
        self.ref_sample = sample
    def add_fit_parameter(self, fit_parameter: Fit_Parameter) -> None:
        """Add a Fit_Parameter to the collection.

        Args:
            fit_parameter (Fit_Parameter): Parameter to add.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters, Fit_Parameter
            params = Fit_Parameters()
            params.add_fit_parameter(Fit_Parameter(name="Rs"))
            params.num_of_parameters()
            ```
        """
        self.fit_parameters.append(fit_parameter)
    def enable_parameter(self, name: Optional[str] = None) -> None:
        """Enable all or a named parameter.

        Args:
            name (Optional[str]): Parameter name to enable; None for all.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.disable_parameter("Rs")
            params.enable_parameter("Rs")
            ```
        """
        for element in self.fit_parameters:
            if name is None or element.name==name:
                element.enabled = True
    def disable_parameter(self, name: Optional[str] = None) -> None:
        """Disable all or a named parameter.

        Args:
            name (Optional[str]): Parameter name to disable; None for all.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.disable_parameter("Rs")
            params.fit_parameters[0].enabled
            ```
        """
        for element in self.fit_parameters:
            if name is None or element.name==name:
                element.enabled = False
    def delete_fit_parameter(self, name: str) -> None:
        """Remove the first parameter with a matching name.

        Args:
            name (str): Parameter name to delete.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.delete_fit_parameter("Rs")
            params.num_of_parameters()
            ```
        """
        for element in self.fit_parameters:
            if element.name==name:
                self.fit_parameters.remove(element)
                break
    def get(
        self,
        attribute: str,
        names: Optional[Union[str, List[str]]] = None,
        enabled_only: bool = True,
    ) -> Union[Any, List[Any]]:
        """Get an attribute list (or scalar if single) for parameters.

        Args:
            attribute (str): Attribute name or special key ("min", "max").
            names (Optional[Union[str, List[str]]]): Filter by name(s).
            enabled_only (bool): If True, include only enabled parameters.

        Returns:
            Union[Any, List[Any]]: Attribute values.

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.get("name")
            ```
        """
        if names is not None:
            if not isinstance(names,list):
                names = [names]
        list_ = []
        for element in self.fit_parameters:
            if (names is not None and element.name in names) or (names is None and ((not enabled_only) or element.enabled)):
                if attribute=="min":
                    list_.append(element.get_min())
                elif attribute=="max":
                    list_.append(element.get_max())
                elif attribute=="limit_order_of_mag":
                    if attribute in element.aux:
                        list_.append(element.aux[attribute])
                    else:
                        list_.append(np.nan)
                else:
                    if attribute in element.aux:
                        list_.append(element.aux[attribute])
                    elif hasattr(element,attribute):
                        list_.append(getattr(element, attribute))
                    else:
                        list_.append(np.nan)
        if len(list_)==1:
            return list_[0]
        return list_
    def set(
        self,
        attribute: str,
        values: Union[Any, List[Any], np.ndarray],
        names: Optional[Union[str, List[str]]] = None,
        enabled_only: bool = True,
    ) -> None:
        """Set an attribute for parameters.

        Args:
            attribute (str): Attribute name or auxiliary key.
            values (Union[Any, List[Any], np.ndarray]): Values to set.
            names (Optional[Union[str, List[str]]]): Filter by name(s).
            enabled_only (bool): If True, apply only to enabled parameters.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.set("value", 0.1)
            params.get("value")
            ```
        """
        if names is not None:
            if not isinstance(names,list):
                names = [names]
        if not isinstance(values,list) and not isinstance(values,np.ndarray):
            values = [values]*self.num_of_parameters()
        count = 0
        for element in self.fit_parameters:
            if (names is not None and element.name in names) or (names is None and ((not enabled_only) or element.enabled)):
                if hasattr(element,attribute):
                    setattr(element, attribute,values[count])
                else:
                    element.aux[attribute] = values[count]
                element.check_max_min()
                count += 1
    def initialize(
        self,
        values: Union[Any, List[Any], np.ndarray],
        names: Optional[Union[str, List[str]]] = None,
        enabled_only: bool = True,
    ) -> None:
        """Initialize current and nominal values together.

        Args:
            values (Union[Any, List[Any], np.ndarray]): Values to set.
            names (Optional[Union[str, List[str]]]): Filter by name(s).
            enabled_only (bool): If True, apply only to enabled parameters.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.initialize(0.1)
            params.get("nominal_value")
            ```
        """
        self.set("value",values,names=names,enabled_only=enabled_only)
        self.set("nominal_value",values,names=names,enabled_only=enabled_only)
    def set_nominal(self) -> None:
        """Set all nominal values to their current values.

        Args:
            None

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.set("value", 0.2)
            params.set_nominal()
            params.get("nominal_value")
            ```
        """
        for element in self.fit_parameters:
            element.set_nominal()
    def set_d_value(self) -> None:
        """Reset differential steps for all parameters.

        Args:
            None

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.set_d_value()
            params.get("d_value")
            ```
        """
        for element in self.fit_parameters:
            element.set_d_value()
    def set_differential(self, which: int = -1, enabled_only: bool = True) -> None:
        """Enable differential mode for a single parameter index.

        Args:
            which (int): Index of parameter to mark; -1 disables.
            enabled_only (bool): If True, count only enabled parameters.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs", "Rsh"])
            params.set_differential(0)
            params.is_differential
            ```
        """
        count = 0
        if which<0:
            self.is_differential = False
        else:
            self.is_differential = True
        for element in self.fit_parameters:
            element.is_differential = False
            if (not enabled_only) or element.enabled:
                if count==which:
                    element.is_differential = True
                count += 1
    def get_parameters(self) -> Dict[str, float]:
        """Return a dictionary of effective parameter values.

        Args:
            None

        Returns:
            Dict[str, float]: Name-to-value mapping.

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.get_parameters()
            ```
        """
        dict_ = {}
        for element in self.fit_parameters:
            dict_[element.name] = element.get_parameter()
        return dict_
    def limit_order_of_mag(self, order_of_mag: float = 1.0) -> None:
        """Limit parameter bounds by order of magnitude.

        Args:
            order_of_mag (float): Number of decades to allow.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.limit_order_of_mag(1.0)
            params.get("min")
            ```
        """
        if not isinstance(order_of_mag,numbers.Number):
            order_of_mag = 1.0
        for element in self.fit_parameters:
            element.limit_order_of_mag(order_of_mag=order_of_mag)
    def num_of_parameters(self) -> int:
        """Return the total number of parameters.

        Args:
            None

        Returns:
            int: Total parameter count.

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs", "Rsh"])
            params.num_of_parameters()
            ```
        """
        return len(self.fit_parameters)
    def num_of_enabled_parameters(self) -> int:
        """Return the number of enabled parameters.

        Args:
            None

        Returns:
            int: Enabled parameter count.

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters(names=["Rs"])
            params.num_of_enabled_parameters()
            ```
        """
        count = 0
        for element in self.fit_parameters:
            if element.enabled:
                count += 1
        return count
    def apply_to_ref(self, aux_info: Any) -> None:
        """Apply parameters to the reference sample.

        Warning:
            This is a stub meant to be overridden in subclasses.

        Args:
            aux_info (Any): Auxiliary data for applying parameters.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters()
            params.apply_to_ref(aux_info=None)
            ```
        """
        pass
    def apply_to_device(self, device: Any) -> None:
        """Apply parameters to a device instance.

        Warning:
            This is a stub meant to be overridden in subclasses.

        Args:
            device (Any): Device object to modify.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting import Fit_Parameters
            params = Fit_Parameters()
            params.apply_to_device(device=None)
            ```
        """
        pass
    def __str__(self) -> str:
        return str(self.get_parameters())

# generally, measurment samples may not be equal to fit_parameters.ref_sample
def compare_experiments_to_simulations(
    fit_parameters: Optional[Fit_Parameters],
    measurement_samples: Any,
    aux: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare experimental measurements with simulated results.

    Applies fit parameters to a reference sample and computes error vectors.

    Args:
        fit_parameters (Optional[Fit_Parameters]): Parameters to apply.
        measurement_samples (Any): Measurement samples or collection.
        aux (Dict[str, Any]): Auxiliary options for comparison.

    Returns:
        Dict[str, Any]: Output dict with error and baseline vectors.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import compare_experiments_to_simulations, Fit_Parameters
        output = compare_experiments_to_simulations(Fit_Parameters(), [], {})
        "error_vector" in output or "differential_vector" in output
        ```
    """
    if fit_parameters is not None:
        fit_parameters.apply_to_ref(aux)
    measurements = measurement_module.collate_device_measurements(measurement_samples)
    for measurement in measurements:
        measurement.simulate()
    output = {}
    if fit_parameters is None or not fit_parameters.is_differential: # baseline case
        measurement_module.set_simulation_baseline(measurements)
        output["error_vector"] = measurement_module.get_measurements_error_vector(measurements)
        output["baseline_vector"] = measurement_module.get_measurements_baseline_vector(measurements)
        output["measurement_samples"] = measurement_samples
    else:
        output["differential_vector"] = measurement_module.get_measurements_differential_vector(measurements)
    return output

class Fit_Dashboard():
    """Dashboard for visualizing fit progress and comparisons.

    Supports multiple plot types and notebook or GUI rendering.

    Args:
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        save_file_name (Optional[str]): Base filename for saved images.
        measurements (Optional[list]): Measurement objects to display.
        RMS_errors (Optional[list]): Error values for plotting progress.

    Returns:
        Fit_Dashboard: The constructed dashboard.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import Fit_Dashboard
        dashboard = Fit_Dashboard(1, 1)
        dashboard.close()
        ```
    """
    def __init__(
        self,
        nrows: int,
        ncols: int,
        save_file_name: Optional[str] = None,
        measurements: Optional[List[Any]] = None,
        RMS_errors: Optional[List[float]] = None,
    ) -> None:
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(6, 5), constrained_layout=True)
        self._display_handle = None   # for notebook live-updates
        self._shown = False           # for desktop GUI
        self._in_nb = _in_notebook()
        if self._in_nb:
            # prevent Jupyter from auto-rendering a static copy the moment it's created
            plt.close(self.fig)
        else:
            # for scripts/desktop: enable interactive redraws
            plt.ion()

        self.fig.canvas.manager.set_window_title("Fit Dashboard")
        self.nrows = nrows
        self.ncols = ncols
        for ax in self.axs.flatten():
            ax.set_visible(False)
        self.plot_what = []
        self.define_plot_what(which_axs=0,plot_type="error")
        self.measurements = measurements # pointer
        self.RMS_errors = RMS_errors # pointer
        self.save_file_name = save_file_name
    def define_plot_what(
        self,
        which_axs: Optional[int] = None,
        measurement_type: Optional[Any] = None,
        key_parameter: Optional[str] = None,
        measurement_condition: Optional[Dict[str, Any]] = None,
        plot_type: str = "exp_vs_sim",
        x_axis: Optional[str] = None,
        title: Optional[str] = None,
        plot_style_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if measurement_condition is None:
            measurement_condition = {}
        if plot_style_parameters is None:
            plot_style_parameters = {}
        # plot type is "error", "exp_vs_sim", "overlap_curves" or "overlap_key_parameter"
        # if measurement_type is None, then plot_type defaults to "error"
        # if plot type is "exp_vs_sim" or "overlap_key_parameter", need to specify key_parameter
        # if plot type is "overlap_key_parameter", need to additionally specify x_axis
        if which_axs is None:
            if len(self.plot_what)==0:
                which_axs=0
            else:
                which_axs = self.plot_what[-1]["which_axs"] + 1
        self.plot_what.append({"which_axs":which_axs,
                               "measurement_type":measurement_type, 
                             "key_parameter": key_parameter, 
                             "measurement_condition": measurement_condition,
                             "plot_type": plot_type,
                             "x_axis": x_axis,
                             "title": title,
                             "plot_style_parameters": plot_style_parameters})
    @staticmethod 
    def convert_scatter_valid_kwargs(plot_style_parameters: Dict[str, Any]) -> Dict[str, Any]:
        scatter_args = inspect.signature(plt.Axes.scatter).parameters
        kwargs = {}
        for key, value in plot_style_parameters.items():
            if key in scatter_args:
                kwargs[key] = value
        return kwargs
    def prep_plot(self) -> None:
        for ax in self.axs.flatten():
            ax.clear()
        for i in range(len(self.plot_what)):
            which_axs = self.plot_what[i]["which_axs"]
            title = self.plot_what[i]["title"]
            ax = self.axs.flatten()[which_axs]
            ax.tick_params(labelsize=6) 
            kwargs = self.convert_scatter_valid_kwargs(self.plot_what[i]["plot_style_parameters"])
            if self.plot_what[i]["plot_type"] == "error":
                if self.RMS_errors is not None:
                    ax.set_visible(True)
                    ax.scatter(np.arange(0,len(self.RMS_errors)), np.log10(np.array(self.RMS_errors)),s=3,**kwargs) 
                    ax.set_title("RMS_error", fontsize=6)
                    ax.set_xlabel("Iteration", fontsize=6)
                    ax.set_ylabel("log10(Error)", fontsize=6)
            elif self.plot_what[i]["plot_type"] == "overlap_curves":
                measurement_type = self.plot_what[i]["measurement_type"]
                measurement_condition = self.plot_what[i]["measurement_condition"]
                for measurement in self.measurements:
                    if isinstance(measurement,measurement_type):
                        meets_all_conditions = True
                        for key, value in measurement_condition.items():
                            if not (key in measurement.measurement_condition and measurement.measurement_condition[key]==value):
                                meets_all_conditions = False
                        if meets_all_conditions:
                            ax.set_visible(True)
                            measurement.plot_func(measurement.measurement_data,color="gray",ax=ax,title=title,kwargs=None)
                            if hasattr(measurement,"simulated_data"):
                                measurement.plot_func(measurement.simulated_data,ax=ax,title=title,kwargs=self.plot_what[i]["plot_style_parameters"])
                            if title is not None:
                                ax.set_title(title, fontsize=6)
            else:
                measurement_type = self.plot_what[i]["measurement_type"]
                key_parameter = self.plot_what[i]["key_parameter"]
                measurement_condition = self.plot_what[i]["measurement_condition"]
                categories = []
                values = []
                for key, value in measurement_condition.items():
                    categories.append(key)
                    values.append(value)
                cond_key = self.plot_what[i]["x_axis"]
                result = measurement_module.get_measurements_groups(self.measurements,
                                measurement_class=measurement_type,
                                categories=categories,
                                optional_x_axis=cond_key,plot_offset=True)
                exp_groups = result[0]
                sim_groups = result[1]
                index = (key_parameter, *values)
                if index in exp_groups:
                    ax.set_visible(True)
                    exp_data = exp_groups[index]
                    sim_data = sim_groups[index]
                    if self.plot_what[i]["x_axis"] is not None:
                        x_axis_groups = result[2]
                        x_axis_data = x_axis_groups[index]
                    match self.plot_what[i]["plot_type"]:
                        case "exp_vs_sim":
                            ax.plot(exp_data,exp_data,color="gray",linewidth=0.5)
                            ax.scatter(exp_data, sim_data,s=3,**kwargs)
                            ax.set_xlabel(key_parameter+"(exp)", fontsize=6)
                            ax.set_ylabel(key_parameter+"(sim)", fontsize=6)
                        case "overlap_key_parameter":
                            ax.scatter(x_axis_data,exp_data,color="gray",s=3)
                            size1 = x_axis_data.size if hasattr(x_axis_data, 'size') else len(x_axis_data)
                            size2 = sim_data.size if hasattr(sim_data, 'size') else len(sim_data)
                            if size1==size2:
                                ax.scatter(x_axis_data,sim_data,s=3,**kwargs)
                            ax.set_xlabel(cond_key, fontsize=6)
                            ax.set_ylabel(key_parameter, fontsize=6)
                            if size1==size2:
                                ax.set_ylim(min(np.min(exp_data)-(np.max(exp_data)-np.min(exp_data))*0.05,np.min(sim_data)-(np.max(sim_data)-np.min(sim_data))*0.05,np.mean(exp_data)*(1-1e-3),np.mean(sim_data)*(1-1e-3)), 
                                            max(np.max(exp_data)+(np.max(exp_data)-np.min(exp_data))*0.05,np.max(sim_data)+(np.max(sim_data)-np.min(sim_data))*0.05,np.mean(exp_data)*(1+1e-3),np.mean(sim_data)*(1+1e-3)))
                            else:
                                ax.set_ylim(min(np.min(exp_data)-(np.max(exp_data)-np.min(exp_data))*0.05,np.mean(exp_data)*(1-1e-3)),
                                            max(np.max(exp_data)+(np.max(exp_data)-np.min(exp_data))*0.05,np.mean(exp_data)*(1+1e-3)))
                    if title is not None:
                        ax.set_title(title, fontsize=6)
        # self.fig.tight_layout()
    def plt_plot(self) -> None:
        # draw/flush the existing fig; do NOT call plt.show() in loops
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self._in_nb and _ip_display is not None:
            # single output that updates in place
            if self._display_handle is None:
                self._display_handle = _ip_display(self.fig, display_id=True)
            else:
                self._display_handle.update(self.fig)
        else:
            # normal Python: one non-blocking window, then just redraw
            if not self._shown:
                self._shown = True
                try:
                    # modern Matplotlib
                    self.fig.show()
                except Exception:
                    # fallback
                    plt.show(block=False)
            # give the GUI event loop a breath
            plt.pause(0.001)
        if self.save_file_name is not None:
            base = Path(self.save_file_name)
            # directory containing the file
            folder = base.parent
            folder.mkdir(parents=True, exist_ok=True)
            word = self.save_file_name + "_fit_round_"+str(len(self.RMS_errors)-1)+".jpg"
            self.fig.savefig(word, format='jpg', dpi=300)
        plt.pause(0.1)
    def plot(self) -> None:
        self.prep_plot()
        self.plt_plot()
    def close(self) -> None:
        # stop trying to draw/update
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            # close just this figure (not all)
            import matplotlib.pyplot as plt
            plt.close(self.fig)
        except Exception:
            pass
    def __enter__(self) -> "Fit_Dashboard":
        return self
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

class Interactive_Fit_Dashboard(Fit_Dashboard):
    """Interactive Tk-based dashboard with parameter sliders.

    Warning:
        Requires a Tkinter environment; will not work in headless mode.

    Args:
        measurement_samples (Any): Samples with measurements to simulate.
        fit_parameters (Fit_Parameters): Parameter collection to adjust.
        nrows (Optional[int]): Number of subplot rows.
        ncols (Optional[int]): Number of subplot columns.
        ref_fit_dashboard (Optional[Fit_Dashboard]): Reference dashboard settings.
        **kwargs (Any): Optional configuration, including apply_function.

    Returns:
        Interactive_Fit_Dashboard: The constructed interactive dashboard.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import Interactive_Fit_Dashboard, Fit_Parameters
        params = Fit_Parameters(names=["Rs", "Rsh"])
        dashboard = Interactive_Fit_Dashboard([], params, nrows=1, ncols=1)
        dashboard.close()
        ```
    """
    def __init__(
        self,
        measurement_samples: Any,
        fit_parameters: Fit_Parameters,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        ref_fit_dashboard: Optional[Fit_Dashboard] = None,
        **kwargs: Any,
    ) -> None:
        self.apply_function = None
        if "apply_function" in kwargs:
            self.apply_function = kwargs["apply_function"]
        self.aux = None
        if ref_fit_dashboard is not None:
            nrows = ref_fit_dashboard.nrows
            ncols = ref_fit_dashboard.ncols
            if hasattr(ref_fit_dashboard,"aux"):
                self.aux = ref_fit_dashboard.aux
        self.nrows = nrows
        self.ncols = ncols
        self.RMS_errors = None
        if ref_fit_dashboard is not None:
            self.plot_what = ref_fit_dashboard.plot_what
        self.measurements = measurement_module.collate_device_measurements(measurement_samples)
        self.parameter_names = fit_parameters.get("name",enabled_only=False)
        self.default_values = fit_parameters.get("value",enabled_only=False)
        self.measurement_samples = measurement_samples
        fit_parameters.limit_order_of_mag(2)
        fit_parameters.set_differential(-1)
        self.min = fit_parameters.get("min",enabled_only=False)
        self.max = fit_parameters.get("max",enabled_only=False)
        is_log = fit_parameters.get("is_log",enabled_only=False)
        for i in range(len(is_log)):
            if not is_log[i]:
                self.min[i] = fit_parameters.get("abs_min",enabled_only=False)[i]
                self.max[i] = fit_parameters.get("abs_max",enabled_only=False)[i]
                if np.isinf(self.min[i]):
                    self.min[i] = -2*abs(self.default_values[i])
                if np.isinf(self.max[i]):
                    self.max[i] = 2*abs(self.default_values[i])
        self.fit_parameters = fit_parameters

    def sync_slider_to_entry(self, i: int) -> None:
        val = self.sliders[i].get()
        if max(abs(self.min[i]),abs(self.max[i])) >= 0.1:
            self.display_values[i].config(text=f"{val:.2f}")
        else:
            self.display_values[i].config(text=f"{val:.2e}")
        self.plot()

    def reset_slider(self, i: int) -> None:
        self.sliders[i].set(self.default_values[i])
        self.sync_slider_to_entry(i)

    def on_close(self) -> None:
        self.control_root.quit()
        self.plot_root.destroy()
        self.control_root.destroy()

    def plot(self) -> None:
        update_values = []
        for slider in self.sliders:
            update_values.append(slider.get())
        self.fit_parameters.set("value",update_values,enabled_only=False)
        self.fit_parameters.apply_to_ref(aux_info=self.aux)
        if self.apply_function is not None:
            self.apply_function(self.fit_parameters, self.measurement_samples, self.aux)
        else:
            for measurement in self.measurements:
                measurement.simulate()
        self.prep_plot()
        self.canvas.draw()

    def run(self) -> None:
        plot_what = self.plot_what
        super().__init__(self.nrows,self.ncols,measurements=self.measurements)
        self.plot_what = plot_what

        # Controls window
        self.control_root = tk.Tk()
        self.control_root.title("Sliders")
        self.control_root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Plot window (separate)
        self.plot_root = tk.Toplevel(self.control_root)
        self.plot_root.title("Fit Dashboard")
        self.plot_root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Setup figure in plot window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.sliders = []
        self.reset_buttons = []
        self.display_values = []
        for i, name in enumerate(self.parameter_names):
            frame = tk.Frame(self.control_root)
            frame.pack(pady=5, padx=10, fill='x')
            ttk.Label(frame, text=name).pack(side='left', padx=5)
            if max(abs(self.min[i]),abs(self.max[i])) >= 0.1:
                ttk.Label(frame, text=f"{self.min[i]:.2f}").pack(side='left', padx=5)
            else:
                ttk.Label(frame, text=f"{self.min[i]:.2e}").pack(side='left', padx=5)
            self.sliders.append(ttk.Scale(frame, from_=self.min[i], to=self.max[i], orient='horizontal'))
            self.sliders[-1].set(self.default_values[i])
            self.sliders[-1].pack(side='left', expand=True, fill='x')
            self.sliders[-1].bind("<ButtonRelease-1>", lambda e, i=i: self.sync_slider_to_entry(i))
            if max(abs(self.min[i]),abs(self.max[i])) >= 0.1:
                ttk.Label(frame, text=f"{self.max[i]:.2f}").pack(side='left', padx=5)
            else:
                ttk.Label(frame, text=f"{self.max[i]:.2e}").pack(side='left', padx=5)
            self.reset_buttons.append(ttk.Button(frame, text="Reset", command=lambda i=i: self.reset_slider(i)))
            self.reset_buttons[-1].pack(side='left', padx=5)
            if max(abs(self.min[i]),abs(self.max[i])) >= 0.1:
                self.display_values.append(ttk.Label(frame, text=f"{self.default_values[i]:.2f}"))
            else:
                self.display_values.append(ttk.Label(frame, text=f"{self.default_values[i]:.2e}"))
            self.display_values[-1].pack(side="right", padx=5)

        self.plot()
        self.control_root.mainloop()

def linear_regression(
    M: np.ndarray,
    Y: np.ndarray,
    fit_parameters: Fit_Parameters,
    aux: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if aux is None:
        aux = {}
    alpha = 1e-5 
    regularization_method=0 
    if "alpha" in aux:
        alpha = aux["alpha"]
    if "regularization_method" in aux:
        regularization_method = aux["regularization_method"]
    if "limit_order_of_mag" in aux:
        if aux["limit_order_of_mag"]:
            fit_parameters.limit_order_of_mag(aux["limit_order_of_mag"])
    this_min_values = fit_parameters.get("this_min")
    this_max_values = fit_parameters.get("this_max")
    abs_min_values = fit_parameters.get("abs_min")
    abs_max_values = fit_parameters.get("abs_max")
    min_values = fit_parameters.get("min")
    max_values = fit_parameters.get("max")
    values = np.array(fit_parameters.get("value"))
    nominal_values = np.array(fit_parameters.get("nominal_value"))
    dvalues = np.array(fit_parameters.get("d_value"))
    too_high_indices = []
    too_low_indices = []  
    included = np.ones_like(nominal_values)
    included_indices = np.where(included==1)[0]
    Ybias = np.zeros_like(Y)
    Xbias = np.zeros_like(nominal_values)

    while True:
        Y_ = Y - Ybias
        M_ = M[:,included_indices]
        Y2 = np.vstack([Y_[:,None],
                        (alpha*(nominal_values[included_indices] - values[included_indices])/dvalues[included_indices])[:,None]])
        M2 = np.vstack([M_,alpha*np.identity(M_.shape[1])])
        # another regularization on how much variables can change at a time
        alpha2 = 1e-7
        memory = [[too_high_indices.copy(),too_low_indices.copy()],included.copy(),included_indices.copy(),Ybias.copy(),Xbias.copy(),M2.copy(),Y2.copy()]
        len_excluded_indices = len(too_low_indices)+len(too_high_indices)
        while True:
            too_high_indices = memory[0][0].copy()
            too_low_indices = memory[0][1].copy()
            included = memory[1].copy()
            included_indices = memory[2].copy()
            Ybias = memory[3].copy()   
            Xbias = memory[4].copy()
            M2 = memory[5].copy()
            Y2 = memory[6].copy()
            if regularization_method==0:
                M2 = np.vstack([M_,alpha2*np.identity(M_.shape[1])])
                Y2 = np.vstack([Y_[:,None],np.zeros((M_.shape[1],1))])

            MTM = M2.T @ M2
            MTY = M2.T @ Y2
            X_ = np.linalg.solve(MTM, MTY)

            X = Xbias.copy()
            X[included_indices] = X_[:,0]

            delta = X*dvalues
            new_values = values + delta

            find_ = np.where(new_values < this_min_values)[0]
            if len(find_) > 0:
                too_low_indices.extend(find_)
            find2_ = np.where(new_values > this_max_values)[0]
            if len(find2_) > 0:
                too_high_indices.extend(find2_)
            if len(too_low_indices) > 0:
                too_low_indices = list(np.unique(np.array(too_low_indices)))

            if regularization_method==1 or len(too_low_indices)+len(too_high_indices) == len_excluded_indices:
                break
            alpha2 *= 3
        
        find_ = np.where(new_values < abs_min_values)[0]
        if len(find_) > 0:
            too_low_indices.extend(find_)
        find_ = np.where(new_values > abs_max_values)[0]
        if len(find_) > 0:
            too_high_indices.extend(find_) 
        if len(too_low_indices) > 0:
            too_low_indices = list(np.unique(np.array(too_low_indices)))
        if len(too_high_indices) > 0:
            too_high_indices = list(np.unique(np.array(too_high_indices)))

        if len(too_low_indices)+len(too_high_indices) == len_excluded_indices:
            break
        min_values = np.array(min_values)
        max_values = np.array(max_values)
        dvalues = np.array(dvalues)
        Xbias[too_low_indices] = (min_values[too_low_indices]-values[too_low_indices])/dvalues[too_low_indices]
        Xbias[too_high_indices] = (max_values[too_high_indices]-values[too_high_indices])/dvalues[too_high_indices]
        Ybias = M @ Xbias
        included = np.ones_like(values)
        included[too_low_indices] = 0
        included[too_high_indices] = 0
        included_indices = np.where(included==1)[0]
        assert(len(included_indices)>0)
    fit_parameters.set("value",new_values)
    return new_values

def uncertainty_analysis(M: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate parameter resolution and error using SVD.

    Args:
        M (np.ndarray): Sensitivity matrix.
        Y (np.ndarray): Error vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (resolution, error) arrays.

    Example:
        ```python
        import numpy as np
        from PV_Circuit_Model.data_fitting import uncertainty_analysis
        M = np.eye(2)
        Y = np.array([0.1, 0.2])
        uncertainty_analysis(M, Y)
        ```
    """
    U, S, VT = np.linalg.svd(M)
    # resolve Y into the Us
    YintoU = Y**2 @ U[:,:len(S)]**2
    error = np.sqrt(YintoU/S**2 @ VT**2)
    resolution = np.sqrt(1/S**2 @ VT**2)
    # in units of the parameter deltas
    return resolution, error

def construct_M(
    iteration: int,
    measurement_samples: Any,
    fit_parameters: Fit_Parameters,
    comparison_function: Any,
    aux: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct a differential measurement output for a fit iteration.

    Args:
        iteration (int): Iteration index for differential parameter selection.
        measurement_samples (Any): Measurement sample collection.
        fit_parameters (Fit_Parameters): Parameters to perturb.
        comparison_function (Any): Function to compare measurements.
        aux (Dict[str, Any]): Auxiliary configuration for progress.

    Returns:
        Dict[str, Any]: Output from the comparison function.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import construct_M, Fit_Parameters
        params = Fit_Parameters(names=["Rs"])
        def compare(fp, samples, aux):
            return {"differential_vector": [0.0]}
        construct_M(0, [], params, compare, {})
        ```
    """
    fit_parameters.set_differential(iteration-1)
    if "pbar" in aux:
        pbar_before = aux["pbar"].n
    output = comparison_function(fit_parameters,measurement_samples,aux)
    if "pbar" in aux:
        pbar_after = aux["pbar"].n
        if pbar_after == pbar_before:
            aux["pbar"].update(aux["comparison_function_iterations"])
    if "f_out" in aux:
        aux["f_out"].write(
            f"STATUS:Fitting progress: {aux['pbar'].n} of {aux['pbar'].total}\n"
        )
        aux["f_out"].flush()
    return output
        
# measurement_samples = collection of devices (Cell, Module, etc)
# each with its measurements stored inside .measurements attribute
# could be one sample only
# could be multiple samples
def fit_routine(
    measurement_samples: Any,
    fit_parameters: Fit_Parameters,
    routine_functions: Dict[str, Any],
    fit_dashboard: Optional[Fit_Dashboard] = None,
    aux: Optional[Dict[str, Any]] = None,
    num_of_epochs: int = 10,
    enable_pbar: bool = True,
    parallel: bool = False,
) -> Any:
    """Run an iterative fitting routine over measurement samples.

    Supports optional parallel evaluation and fit dashboards.

    Args:
        measurement_samples (Any): Measurement samples to fit.
        fit_parameters (Fit_Parameters): Parameter collection to update.
        routine_functions (Dict[str, Any]): Functions for comparison and update.
        fit_dashboard (Optional[Fit_Dashboard]): Optional visualization dashboard.
        aux (Optional[Dict[str, Any]]): Auxiliary configuration.
        num_of_epochs (int): Number of fitting epochs.
        enable_pbar (bool): If True, show progress bar updates.
        parallel (bool): If True, use joblib parallel evaluation.

    Returns:
        Any: Output from the final comparison function or intermediate data.

    Example:
        ```python
        from PV_Circuit_Model.data_fitting import fit_routine, Fit_Parameters
        params = Fit_Parameters(names=["Rs"])
        routine_functions = {"comparison_function": lambda fp, s, a: {"error_vector": [0.0], "baseline_vector": [0.0], "measurement_samples": s},
                             "update_function": lambda M, Y, fp, a: None}
        fit_routine([], params, routine_functions, num_of_epochs=1)
        ```
    """
    if aux is None:
        aux = {}
    if parallel:
        parallel_mode_prior = get_parallel_mode()
        set_parallel_mode(False)
    if "initial_guess" in routine_functions:
        routine_functions["initial_guess"](fit_parameters,measurement_samples,aux)
    RMS_errors = []
    this_RMS_errors = []
    record = []
    measurements = measurement_module.collate_device_measurements(measurement_samples)
    if fit_dashboard is not None and num_of_epochs>0:
        if fit_dashboard.RMS_errors is None:
            fit_dashboard.RMS_errors = RMS_errors
            fit_dashboard.measurements = measurements
        else:
            RMS_errors = fit_dashboard.RMS_errors  
    silent_mode = False
    if "silent_mode" in aux:
        silent_mode = aux["silent_mode"]
    if "comparison_function_iterations" not in aux:
        aux["comparison_function_iterations"] = 1
    total=((num_of_epochs-1)*(fit_parameters.num_of_enabled_parameters()+1)+1)*aux["comparison_function_iterations"]
    has_outer_loop = False
    if "pbar" in aux:
        has_outer_loop = True
    if "pbar" not in aux:
        if "f_out" not in aux and enable_pbar:
            aux["pbar"] = tqdm(total=total,desc="Calibrating")
        else:
            aux["pbar"] = SimpleNamespace(
                n=0,
                total=total,
                update=lambda N: setattr(aux["pbar"], "n", aux["pbar"].n + N),
                close=lambda: None,
            )

    for epoch in range(max(1,num_of_epochs)):
        M = []
        iterations = fit_parameters.num_of_enabled_parameters()+1
        if parallel:
            aux_safe = {k: v for k, v in aux.items() if k not in ("pbar", "f_out")}
            n_cpus = os.cpu_count() or 1
            results = Parallel(n_jobs=min(n_cpus,iterations), backend="loky")(
                delayed(construct_M)(iteration,measurement_samples,fit_parameters,
                                routine_functions["comparison_function"],aux_safe)
                for iteration in range(iterations)
            )
            aux["pbar"].update((fit_parameters.num_of_enabled_parameters()+1)*aux["comparison_function_iterations"])
            if "f_out" in aux:
                aux["f_out"].write(
                    f"STATUS:Fitting progress: {aux['pbar'].n} of {aux['pbar'].total}\n"
                )
                aux["f_out"].flush()

        for iteration in range(iterations):
            if parallel:
                output = results[iteration]
            else:
                output = construct_M(iteration,measurement_samples,fit_parameters,
                    routine_functions["comparison_function"],aux) 
            if iteration==0:
                if parallel:
                    if "fit_parameters" in output:
                        fit_parameters_clone = output["fit_parameters"]
                        fit_parameters.set("value", fit_parameters_clone.get("value", enabled_only=False), enabled_only=False)
                        fit_parameters.set("nominal_value", fit_parameters_clone.get("nominal_value", enabled_only=False), enabled_only=False)
                    baseline_vector = output["baseline_vector"]
                    if not isinstance(measurement_samples, list):
                        measurement_samples_list = [measurement_samples]
                        measurement_samples_list_ = [output["measurement_samples"]]
                    else:
                        measurement_samples_list = measurement_samples
                        measurement_samples_list_ = output["measurement_samples"]
                    
                    for i, sample in enumerate(measurement_samples_list):
                        for j, measurement in enumerate(sample.measurements):
                            measurement.simulated_data = measurement_samples_list_[i].measurements[j].simulated_data
                            measurement.simulated_key_parameters = measurement_samples_list_[i].measurements[j].simulated_key_parameters
                            measurement.simulated_key_parameters_baseline = measurement_samples_list_[i].measurements[j].simulated_key_parameters_baseline
                Y = np.array(output["error_vector"])
                this_RMS_errors.append(np.sqrt(np.mean(Y**2)))
                RMS_errors.append(np.sqrt(np.mean(np.array(measurement_module.get_measurements_error_vector(measurements,exclude_tags=None))**2)))
                fit_parameters_clone = fit_parameters.clone()
                fit_parameters_clone.ref_sample = fit_parameters.ref_sample
                record.append({"fit_parameters": fit_parameters_clone,"output": output})
                if fit_dashboard is not None and num_of_epochs>0 and not silent_mode:
                    fit_dashboard.plot()
            else:
                if parallel:
                    output["differential_vector"] = list(np.array(output["differential_vector"]) - np.array(baseline_vector))
                M.append(output["differential_vector"])
            if epoch==num_of_epochs-1:
                if has_outer_loop:
                    if fit_dashboard is not None:
                        fit_dashboard.close()
                    return record[-1]["output"]
                index = np.argmin(np.array(this_RMS_errors))
                fit_parameters_clone = record[index]["fit_parameters"]
                RMS_errors[-1] = RMS_errors[index+len(RMS_errors)-len(this_RMS_errors)]
                this_RMS_errors[-1] = this_RMS_errors[index]
                output = record[index]["output"]
                fit_parameters_clone.set_differential(-1)
                routine_functions["comparison_function"](fit_parameters_clone,measurement_samples,aux)
                if fit_dashboard is not None and num_of_epochs>0 and not silent_mode:
                    fit_dashboard.plot()
                fit_parameters.set("value", fit_parameters_clone.get("value", enabled_only=False), enabled_only=False)
                fit_parameters.set_differential(-1)
                if "pbar" in aux:
                    aux["pbar"].close()
                if fit_dashboard is not None:
                    fit_dashboard.close()
                return output
        M = np.array(M)
        M = M.T
        if num_of_epochs==0: # if num_of_epochs=0, just calculate M, Y but do not try to update
            if fit_dashboard is not None:
                fit_dashboard.close()
            return (M, Y, fit_parameters, aux)
        if epoch==num_of_epochs-2 and not has_outer_loop:
            try:
                resolution, error = uncertainty_analysis(M,Y)
                # scale them back to be in the parameter native units
                d_values = fit_parameters.get("d_value")
                is_logs = fit_parameters.get("is_log")
                values = fit_parameters.get("value")
                for i, is_log in enumerate(is_logs):
                    if is_log:
                        resolution[i] *= 10**(values[i])*(10**(d_values[i])-1)
                        error[i] *= 10**(values[i])*(10**(d_values[i])-1)
                    else:
                        resolution[i] *= d_values[i]
                        error[i] *= d_values[i]
                fit_parameters.set("error",error)
                fit_parameters.set("resolution",resolution)
            except Exception:
                pass
        fit_parameters.set_differential(-1)
        for measurement in measurements:
            measurement.simulated_key_parameters_baseline = {}
        routine_functions["update_function"](M, Y, fit_parameters, aux)
        if "post_update_function" in routine_functions:
            routine_functions["post_update_function"](fit_parameters)

    if parallel:
        set_parallel_mode(parallel_mode_prior)
    
