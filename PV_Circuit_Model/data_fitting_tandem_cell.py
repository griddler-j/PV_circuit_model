import PV_Circuit_Model.utilities as utilities
import PV_Circuit_Model.data_fitting as fitting
import PV_Circuit_Model.circuit_model as circuit
import PV_Circuit_Model.device as device_module
import PV_Circuit_Model.device_analysis as analysis
import PV_Circuit_Model.measurement as measurement_module
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

class Tandem_Cell_Fit_Parameters(fitting.Fit_Parameters):
    """Fit parameters for tandem and single-junction solar cells.
    """
    parameter_names = ["bottom_cell_logJ01","bottom_cell_logJ02","bottom_cell_log_shunt_cond",
                       "top_cell_logJ01","top_cell_logJ02","top_cell_PC_logJ01","top_cell_log_shunt_cond",
                       "log_Rs_cond"]
    def __init__(
        self,
        sample: device_module.Device,
        bottom_cell_Voc: float = 0.7,
        top_cell_Voc: Optional[float] = 1.2,
        disable_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize tandem-cell fit parameters and bounds.

        Args:
            sample: Reference sample (Cell or MultiJunctionCell).
            bottom_cell_Voc (float): Approximate Voc for bottom cell.
            top_cell_Voc (Optional[float]): Approximate Voc for top cell.
            disable_list (Optional[List[str]]): Parameter names to disable.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting_tandem_cell import Tandem_Cell_Fit_Parameters
            from PV_Circuit_Model.device import quick_tandem_cell
            params = Tandem_Cell_Fit_Parameters(quick_tandem_cell(), top_cell_Voc=1.2)
            params.is_tandem
            ```
        """
        if disable_list is None:
            disable_list = ["dshunt_cond"]
        super().__init__(names=self.parameter_names)
        # Jsc and dJsc are not fitting parameters.  They are to be fitted inside inner loop
        self.set("is_log", False, enabled_only=False)
        self.set("is_log", True, names=["bottom_cell_logJ01","bottom_cell_logJ02",
                                        "top_cell_logJ01","top_cell_logJ02","top_cell_PC_logJ01",
                                        "bottom_cell_log_shunt_cond",
                                        "top_cell_log_shunt_cond","log_Rs_cond"])
        
        VT = utilities.VT_at_25C
        max_J01 = 0.01/np.exp(bottom_cell_Voc/VT)
        max_J02 = 0.01/np.exp(bottom_cell_Voc/(2*VT))
        self.set("abs_min", [np.log10(max_J01)-4,np.log10(max_J02)-4], ["bottom_cell_logJ01","bottom_cell_logJ02"])
        # if top_cell_Voc is None, then it is fitting a single junction cell
        self.is_tandem = True
        if isinstance(sample,device_module.MultiJunctionCell):
            self.is_tandem = True
        elif isinstance(sample,device_module.Cell):
            self.is_tandem = False
        else:
            raise NotImplementedError("Sample must be either MultiJunctionCell or Cell")
        if top_cell_Voc is None:
            disable_list.extend(["top_cell_logJ01","top_cell_logJ012","top_cell_PC_logJ01","top_cell_log_shunt_cond"])
            self.set("abs_min", [-6,-2], ["bottom_cell_log_shunt_cond","log_Rs_cond"])
            self.is_tandem = False
        else:
            max_J01 = 0.01/np.exp(top_cell_Voc/VT)
            max_J02 = 0.01/np.exp(top_cell_Voc/(2*VT))
            self.set("abs_min", [np.log10(max_J01)-4,np.log10(max_J02)-4,np.log10(max_J01)-4], ["top_cell_PC_logJ01","top_cell_logJ01","top_cell_logJ02"])
            self.set("abs_min", [-6,-6,-2], ["bottom_cell_log_shunt_cond","top_cell_log_shunt_cond","log_Rs_cond"])
        
        for item in disable_list:
            self.disable_parameter(item)
        self.initialize_from_sample(sample)
        self.set_d_value()
    def initialize_from_sample(self, sample: device_module.Device) -> None:
        """Initialize parameter values from a reference sample.

        Args:
            sample (Any): Reference sample to read initial values from.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting_tandem_cell import Tandem_Cell_Fit_Parameters
            from PV_Circuit_Model.device import quick_tandem_cell
            cell = quick_tandem_cell()
            params = Tandem_Cell_Fit_Parameters(cell)
            params.initialize_from_sample(cell)
            ```
        """
        sample.set_temperature(25.0)
        abs_min = self.get("abs_min")
        if self.is_tandem:
            self.initialize([np.log10(max(10**abs_min[0],sample.cells[0].J01())), np.log10(max(10**abs_min[1],sample.cells[0].J02())), 
                            np.log10(max(10**abs_min[2],sample.cells[0].specific_shunt_cond())), 
                            np.log10(max(10**abs_min[3],sample.cells[1].J01())), np.log10(max(10**abs_min[4],sample.cells[1].J02())), 
                            np.log10(max(10**abs_min[5],sample.cells[1].PC_J01())), 
                            np.log10(max(10**abs_min[6],sample.cells[1].specific_shunt_cond())), 
                            np.log10(max(10**abs_min[7],sample.specific_Rs_cond()))],
                            self.parameter_names)
        else:
            self.initialize([np.log10(max(10**abs_min[0],sample.J01())), np.log10(max(10**abs_min[1],sample.J02())), 
                            np.log10(max(10**abs_min[2],sample.specific_shunt_cond())), 
                            np.log10(max(10**abs_min[7],sample.specific_Rs_cond()))],
                            ["bottom_cell_logJ01","bottom_cell_logJ02","bottom_cell_log_shunt_cond","log_Rs_cond"])
        self.ref_sample = sample
    def apply_to_ref(self, aux_info: Any) -> None:
        """Apply current parameter values to the reference sample.

        Also writes parameter uncertainties into component `aux["error"]`.

        Args:
            aux_info (Any): Auxiliary data (unused).

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.data_fitting_tandem_cell import Tandem_Cell_Fit_Parameters
            from PV_Circuit_Model.device import quick_tandem_cell
            params = Tandem_Cell_Fit_Parameters(quick_tandem_cell())
            params.apply_to_ref(aux_info=None)
            ```
        """
        parameters = self.get_parameters()
        if self.is_tandem:
            self.ref_sample.cells[0].set_J01(parameters["bottom_cell_logJ01"]) # no need to raise power, function returns J01, J02
            self.ref_sample.cells[0].set_J02(parameters["bottom_cell_logJ02"])
            self.ref_sample.cells[0].set_specific_shunt_cond(parameters["bottom_cell_log_shunt_cond"])
            self.ref_sample.cells[1].set_J01(parameters["top_cell_logJ01"]) # no need to raise power, function returns J01, J02
            self.ref_sample.cells[1].set_J02(parameters["top_cell_logJ02"])
            self.ref_sample.cells[1].set_PC_J01(parameters["top_cell_PC_logJ01"])
            self.ref_sample.cells[1].set_specific_shunt_cond(parameters["top_cell_log_shunt_cond"])
            self.ref_sample.set_specific_Rs_cond(parameters["log_Rs_cond"])
            shunt_resistors = self.ref_sample.cells[0].diode_branch.findElementType(circuit.Resistor)
            diodes = self.ref_sample.cells[0].diode_branch.findElementType(circuit.ForwardDiode)
        else:
            self.ref_sample.set_J01(parameters["bottom_cell_logJ01"]) # no need to raise power, function returns J01, J02
            self.ref_sample.set_J02(parameters["bottom_cell_logJ02"])
            self.ref_sample.set_specific_shunt_cond(parameters["bottom_cell_log_shunt_cond"])
            self.ref_sample.set_specific_Rs_cond(parameters["log_Rs_cond"])
            shunt_resistors = self.ref_sample.diode_branch.findElementType(circuit.Resistor)
            diodes = self.ref_sample.diode_branch.findElementType(circuit.ForwardDiode)
    
        errors = self.get("error",enabled_only=False)
        self.ref_sample.series_resistor.aux["error"] = errors[7]
        
        for res in shunt_resistors:
            if res.tag != "defect":
                res.aux["error"] = errors[2]
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,circuit.Intrinsic_Si_diode) and not isinstance(diode,circuit.PhotonCouplingDiode):
                if diode.n==1:
                    diode.aux["error"] = errors[0]
                elif diode.n==2:
                    diode.aux["error"] = errors[1]
        if self.is_tandem:
            shunt_resistors = self.ref_sample.cells[1].diode_branch.findElementType(circuit.Resistor)
            for res in shunt_resistors:
                if res.tag != "defect":
                    res.aux["error"] = errors[6]
            diodes = self.ref_sample.cells[1].diode_branch.findElementType(circuit.ForwardDiode)
            for diode in diodes:
                if diode.tag != "defect" and diode.tag != "intrinsic" and not isinstance(diode,circuit.PhotonCouplingDiode):
                    if isinstance(diode,circuit.PhotonCouplingDiode):
                        diode.aux["error"] = errors[5]
                    elif diode.n==1:
                        diode.aux["error"] = errors[3]
                    elif diode.n==2:
                        diode.aux["error"] = errors[4]
    
def analyze_solar_cell_measurements(
    measurements: Sequence[Any],
    num_of_rounds: int = 20,
    regularization_method: int = 0,
    prefix: Optional[str] = None,
    sample_info: Optional[Dict[str, Any]] = None,
    starting_guess: Optional[Any] = None,
    use_fit_dashboard: bool = True,
    **kwargs: Any,
) -> Union[Tuple[Any, Optional[fitting.Interactive_Fit_Dashboard]], Any]:
    """Analyze and fit solar-cell measurement data.

    Builds an initial cell model, runs fitting, and optionally opens dashboards.

    Args:
        measurements (Sequence[Any]): Measurement objects to fit.
        num_of_rounds (int): Number of fitting rounds (0 for dry run).
        regularization_method (int): Regularization method ID.
        prefix (Optional[str]): Filename prefix for dashboard images.
        sample_info (Optional[Dict[str, Any]]): Sample metadata (area, thickness).
        starting_guess (Optional[Any]): Starting model to clone.
        use_fit_dashboard (bool): If True, create fit dashboards.
        **kwargs (Any): Extra options passed to fitting routines.

    Returns:
        Union[Tuple[Any, Optional[Interactive_Fit_Dashboard]], Any]: Fit result or
        (fitted_sample, interactive_dashboard).

    Example:
        ```python
        from PV_Circuit_Model.data_fitting_tandem_cell import analyze_solar_cell_measurements
        analyze_solar_cell_measurements([], num_of_rounds=0)
        ```
    """
    global pbar, axs
    if sample_info is None:
        sample_info = {}
    aux = {"regularization_method": regularization_method,"limit_order_of_mag": 2.5}
    aux.update(kwargs)

    parallel = False
    if "parallel" in kwargs and kwargs["parallel"]:
        parallel = True

    is_tandem = True
    if "is_tandem" in kwargs and not kwargs["is_tandem"]:
        is_tandem = False

    test_cell_area = 1.0
    if "area" in sample_info:
        test_cell_area = sample_info["area"]
    kwargs_to_pass = {}
    if "bottom_cell_thickness" in sample_info:
        kwargs_to_pass["thickness"] = sample_info["bottom_cell_thickness"]
    if "base_type" in sample_info:
        kwargs_to_pass["base_type"] = sample_info["base_type"]
    if "base_doping" in sample_info:
        kwargs_to_pass["base_doping"] = sample_info["base_doping"]
    enable_Auger = True
    if "enable_Auger" in sample_info:
        enable_Auger = sample_info["enable_Auger"]

    if starting_guess is not None:
        cell = circuit.circuit_deepcopy(starting_guess)
    elif is_tandem:
        bottom_cell = None
        top_cell = None
        for measurement in measurements:
            if isinstance(measurement,measurement_module.Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[0,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc < bottom_cell_Isc*1e-3: # this is red spectrum Suns-Voc
                    bottom_cell_Voc = measurement.measurement_data[0,arg_max]
                    Jsc = bottom_cell_Isc/test_cell_area
                    J01, J02 = analysis.estimate_cell_J01_J02(Jsc=Jsc,Voc=bottom_cell_Voc,Si_intrinsic_limit=enable_Auger,**kwargs_to_pass)
                    bottom_cell = device_module.make_solar_cell(Jsc, J01, J02, area=test_cell_area,**kwargs_to_pass)
                    bottom_cell.set_Suns(1.0)
                    break
        for measurement in measurements:
            if isinstance(measurement,measurement_module.Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[1,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc > bottom_cell_Isc*1e-2 and bottom_cell_Isc > top_cell_Isc*1e-2: # this is white spectrum Suns-Voc
                    tandem_cell_Voc = measurement.measurement_data[0,arg_max]
                    bottom_cell.set_IL(bottom_cell_Isc)
                    bottom_cell.build_IV()
                    top_cell_Voc = tandem_cell_Voc - bottom_cell.get_Voc()
                    Jsc = top_cell_Isc/test_cell_area
                    J01, J02 = analysis.estimate_cell_J01_J02(Jsc=Jsc,Voc=top_cell_Voc,Si_intrinsic_limit=False)
                    # just arbitrarily guess the PC
                    J01_PC = J01*0.2
                    J01 = J01*0.8
                    top_cell = device_module.make_solar_cell(Jsc, J01, J02, area=test_cell_area, 
                            Si_intrinsic_limit=False,J01_photon_coupling=J01_PC)
                    top_cell.set_Suns(1.0)
                    break
        if bottom_cell is not None and top_cell is not None:
            cell = device_module.MultiJunctionCell([bottom_cell,top_cell])
        else:
            cell = analysis.quick_tandem_cell()
    else:
        for measurement in measurements:
            if isinstance(measurement,measurement_module.Dark_IV_measurement):
                measurement.set_unit_error("I_bias",value=1.0) # unimportant for single junction

        # survey for 2 lights at 25C
        Suns = []
        light_IV_measurements = []
        temperatures = []
        for measurement in measurements:
            if isinstance(measurement,measurement_module.Light_IV_measurement):
                Suns.append(measurement.measurement_condition["Suns"])
                temperatures.append(measurement.measurement_condition["temperature"])
                light_IV_measurements.append(measurement)
        Suns = np.array(Suns)
        temperatures = np.array(temperatures)
        find_ = np.where(np.abs(temperatures-25)<1)[0]
        index1Sun = None
        Isc_one_Sun = None
        if len(find_)>0:
            index1Sun = find_[np.argmin(np.abs(Suns[find_]-1.0))]
            Isc_one_Sun = light_IV_measurements[index1Sun].key_parameters["Isc"]/Suns[index1Sun]
            if abs(Suns[index1Sun]-1.0)>0.1:
                index1Sun = None
        indexhalfSun = None
        if len(find_)>0:
            indexhalfSun = find_[np.argmin(np.abs(Suns[find_]-0.5))]
            if Suns[indexhalfSun]<0 or Suns[indexhalfSun]>0.7:
                indexhalfSun = None
        
        light_IV_1Sun = None
        light_IV_halfSun = None
        if index1Sun is not None:
            light_IV_1Sun = light_IV_measurements[index1Sun]
        if indexhalfSun is not None:
            light_IV_halfSun = light_IV_measurements[indexhalfSun]

        dark_IV = None
        for measurement in measurements:
            if isinstance(measurement,measurement_module.Dark_IV_measurement):
                if abs(measurement.measurement_condition["temperature"]-25)<1:
                    dark_IV = measurement
                    break
        
        Rshunt_ = 1e6
        if dark_IV is not None:
            IV_curve = dark_IV.measurement_data
            Rshunt_ = analysis.Rshunt_extraction(IV_curve,base_point=0)*test_cell_area

        Rs_ = 0
        if light_IV_1Sun is not None and light_IV_halfSun is not None:
            IV_curve1 = light_IV_1Sun.measurement_data
            IV_curve2 = light_IV_halfSun.measurement_data
            Rs_ = analysis.Rs_extraction_two_light_IVs([IV_curve1,IV_curve2])*test_cell_area

        J01 = 10e-15
        J02 = 1e-9
        if light_IV_1Sun is not None:
            Voc_ = light_IV_1Sun.key_parameters["Voc"]
            Jsc_ = light_IV_1Sun.key_parameters["Isc"]/test_cell_area
            Pmax_ = light_IV_1Sun.key_parameters["Pmax"]
            J01, J02 = analysis.estimate_cell_J01_J02(Jsc=Jsc_,Voc=Voc_,Pmax=Pmax_,Rs=Rs_,Rshunt=Rshunt_,
                    temperature=25,Sun=1.0,Si_intrinsic_limit=enable_Auger,**kwargs_to_pass)

        cell = device_module.make_solar_cell(Jsc_, J01, J02, Rshunt_, Rs_, test_cell_area, shape=None, **kwargs_to_pass)

    measurement_module.assign_measurements(cell,measurements)

    fit_parameters = Tandem_Cell_Fit_Parameters(cell)

    fit_dashboard = None
    if use_fit_dashboard:
        fit_dashboard = fitting.Fit_Dashboard(3,3,save_file_name=prefix)
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=measurement_module.Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"white"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"black"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=measurement_module.Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"blue"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"blue"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=measurement_module.Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"red"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"red"})
        if is_tandem:
            fit_dashboard.define_plot_what(which_axs=2, measurement_type=measurement_module.Dark_IV_measurement, 
                                        measurement_condition={'spectrum':"red"}, 
                                        plot_type="overlap_curves",
                                            title="Dark IV", plot_style_parameters={"color":"red"})
            fit_dashboard.define_plot_what(which_axs=2, measurement_type=measurement_module.Dark_IV_measurement, 
                                        measurement_condition={'spectrum':"blue"}, 
                                        plot_type="overlap_curves",
                                            title="Dark IV", plot_style_parameters={"color":"blue"})
        else:
            fit_dashboard.define_plot_what(which_axs=2, measurement_type=measurement_module.Dark_IV_measurement, 
                                        plot_type="overlap_curves",
                                            title="Dark IV", plot_style_parameters={"color":"black"})
        for i, key in enumerate(measurement_module.Light_IV_measurement.keys):
            for j, Suns in enumerate([1.0, 0.5]):
                fit_dashboard.define_plot_what(which_axs=3*(j+1)+i, measurement_type=measurement_module.Light_IV_measurement, 
                                            key_parameter=key, 
                                            measurement_condition={'Suns':Suns},
                                            plot_type="overlap_key_parameter",
                                            x_axis="I_imbalance",
                                            title=str(Suns)+" Suns IV", plot_style_parameters={"color":"blue"})
                
    result = fitting.fit_routine(cell,fit_parameters,
                routine_functions={
                    "update_function":fitting.linear_regression,
                    "comparison_function":fitting.compare_experiments_to_simulations,
                },
                fit_dashboard=fit_dashboard,
                aux=aux,num_of_epochs=num_of_rounds, parallel=parallel)

    if num_of_rounds==0:
        return result
    fit_parameters.apply_to_ref(aux)
    interactive_fit_dashboard = None
    if use_fit_dashboard:
        interactive_fit_dashboard = fitting.Interactive_Fit_Dashboard(cell,fit_parameters,ref_fit_dashboard=fit_dashboard)

    if not is_tandem and Isc_one_Sun is not None:
        fit_parameters.ref_sample.set_IL(Isc_one_Sun)

    return fit_parameters.ref_sample, interactive_fit_dashboard

def generate_differentials(measurements: Sequence[Any], cell: Any) -> Any:
    is_tandem = False
    if isinstance(cell,device_module.MultiJunctionCell):
        is_tandem = True
    return analyze_solar_cell_measurements(measurements,num_of_rounds=0,starting_guess=cell,is_tandem=is_tandem)
