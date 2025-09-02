from PV_Circuit_Model.circuit_model import *
import numpy as np
from PV_Circuit_Model.measurement import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
from PV_Circuit_Model.data_fitting import *

class Tandem_Cell_Fit_Parameters(Fit_Parameters):
    parameter_names = ["bottom_cell_logJ01","bottom_cell_logJ02","bottom_cell_log_shunt_cond",
                       "top_cell_logJ01","top_cell_logJ02","top_cell_PC_logJ01","top_cell_log_shunt_cond",
                       "log_Rs_cond"]
    def __init__(self, sample, bottom_cell_Voc=0.7, top_cell_Voc=1.2, disable_list=["dshunt_cond"]):
        super().__init__(names=self.parameter_names)
        # Jsc and dJsc are not fitting parameters.  They are to be fitted inside inner loop
        self.set("is_log", False, enabled_only=False)
        self.set("is_log", True, names=["bottom_cell_logJ01","bottom_cell_logJ02",
                                        "top_cell_logJ01","top_cell_logJ02","top_cell_PC_logJ01",
                                        "bottom_cell_log_shunt_cond",
                                        "top_cell_log_shunt_cond","log_Rs_cond"])
        VT = get_VT(25.0)
        self.approx_bottom_cell_Voc = bottom_cell_Voc
        self.approx_top_cell_Voc = top_cell_Voc
        max_J01 = 0.01/np.exp(bottom_cell_Voc/VT)
        max_J02 = 0.01/np.exp(bottom_cell_Voc/(2*VT))
        self.set("abs_min", [np.log10(max_J01)-4,np.log10(max_J02)-4], ["bottom_cell_logJ01","bottom_cell_logJ02"])
        max_J01 = 0.01/np.exp(top_cell_Voc/VT)
        max_J02 = 0.01/np.exp(top_cell_Voc/(2*VT))
        self.set("abs_min", [np.log10(max_J01)-4,np.log10(max_J02)-4,np.log10(max_J01)-4], ["top_cell_PC_logJ01","top_cell_logJ01","top_cell_logJ02"])
        self.set("abs_min", [-6,-6,-2], ["bottom_cell_log_shunt_cond","top_cell_log_shunt_cond","log_Rs_cond"])
        for item in disable_list:
            self.disable_parameter(item)
        self.initialize_from_sample(sample)
        self.set_d_value()
    def initialize_from_sample(self,sample):
        sample.set_temperature(25.0)
        abs_min = self.get("abs_min")
        self.initialize([np.log10(max(abs_min[0],sample.cells[0].J01())), np.log10(max(abs_min[1],sample.cells[0].J02())), 
                          np.log10(max(abs_min[2],sample.cells[0].specific_shunt_cond())), 
                          np.log10(max(abs_min[3],sample.cells[1].J01())), np.log10(max(abs_min[4],sample.cells[1].J02())), 
                          np.log10(max(abs_min[5],sample.cells[1].PC_J01())), 
                          np.log10(max(abs_min[6],sample.cells[1].specific_shunt_cond())), 
                          np.log10(max(abs_min[7],sample.specific_Rs_cond()))],
                          self.parameter_names)
        self.ref_sample = sample
    def apply_to_ref(self, aux_info):
        parameters = self.get_parameters()
        self.ref_sample.cells[0].set_J01(parameters["bottom_cell_logJ01"]) # no need to raise power, function returns J01, J02
        self.ref_sample.cells[0].set_J02(parameters["bottom_cell_logJ02"])
        self.ref_sample.cells[0].set_specific_shunt_cond(parameters["bottom_cell_log_shunt_cond"])
        self.ref_sample.cells[1].set_J01(parameters["top_cell_logJ01"]) # no need to raise power, function returns J01, J02
        self.ref_sample.cells[1].set_J02(parameters["top_cell_logJ02"])
        self.ref_sample.cells[1].set_PC_J01(parameters["top_cell_PC_logJ01"])
        self.ref_sample.cells[1].set_specific_shunt_cond(parameters["top_cell_log_shunt_cond"])
        self.ref_sample.set_specific_Rs_cond(parameters["log_Rs_cond"])
        errors = self.get("error")
        self.ref_sample.series_resistor.aux["error"] = errors[7]
        shunt_resistors = self.ref_sample.cells[0].diode_branch.findElementType(Resistor)
        for res in shunt_resistors:
            if res.tag != "defect":
                res.aux["error"] = errors[2]
        diodes = self.ref_sample.cells[0].diode_branch.findElementType(ForwardDiode)
        for diode in diodes:
            if diode.tag != "defect" and diode.tag != "intrinsic" and not isinstance(diode,PhotonCouplingDiode):
                if diode.n==1:
                    diode.aux["error"] = errors[0]
                elif diode.n==2:
                    diode.aux["error"] = errors[1]
        shunt_resistors = self.ref_sample.cells[1].diode_branch.findElementType(Resistor)
        for res in shunt_resistors:
            if res.tag != "defect":
                res.aux["error"] = errors[6]
        diodes = self.ref_sample.cells[1].diode_branch.findElementType(ForwardDiode)
        for diode in diodes:
            if diode.tag != "defect" and diode.tag != "intrinsic" and not isinstance(diode,PhotonCouplingDiode):
                if isinstance(diode,PhotonCouplingDiode):
                    diode.aux["error"] = errors[5]
                elif diode.n==1:
                    diode.aux["error"] = errors[3]
                elif diode.n==2:
                    diode.aux["error"] = errors[4]
    
def analyze_tandem_cell_measurements(measurements,num_of_rounds=40,regularization_method=0,prefix=None,sample_info={},tandem_cell_starting_guess=None,use_fit_dashboard=True,**kwargs):
    global pbar, axs
    aux = {"regularization_method": regularization_method,"limit_order_of_mag": 1}
    aux.update(kwargs)

    test_cell_area = 1.0
    bottom_cell_thickness = 160e-4
    if "area" in sample_info:
        test_cell_area = sample_info["area"]
    if "bottom_cell_thickness" in sample_info:
        bottom_cell_thickness = sample_info["bottom_cell_thickness"]

    if tandem_cell_starting_guess is not None:
        tandem_cell = circuit_deepcopy(tandem_cell_starting_guess)
    else:
        tandem_cell = quick_tandem_cell()
        bottom_cell = None
        top_cell = None
        for measurement in measurements:
            if isinstance(measurement,Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[0,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc < bottom_cell_Isc*1e-3: # this is red spectrum Suns-Voc
                    bottom_cell_Voc = measurement.measurement_data[0,arg_max]
                    Jsc = bottom_cell_Isc/test_cell_area
                    J01, J02 = estimate_cell_J01_J02(Jsc=Jsc,Voc=bottom_cell_Voc,thickness=bottom_cell_thickness)
                    bottom_cell = make_solar_cell(Jsc, J01, J02, thickness=bottom_cell_thickness, area=test_cell_area)
                    bottom_cell.set_Suns(1.0)
                    break
        for measurement in measurements:
            if isinstance(measurement,Suns_Voc_measurement):
                num_row = measurement.measurement_data.shape[0]
                num_subcells = int((num_row-1)/2)
                Iscs = measurement.measurement_data[num_subcells+1:,:]
                arg_max = np.argmax(Iscs[1,:])
                bottom_cell_Isc = Iscs[0,arg_max]
                top_cell_Isc = Iscs[1,arg_max]
                if top_cell_Isc > bottom_cell_Isc*1e-2: # this is white spectrum Suns-Voc
                    tandem_cell_Voc = measurement.measurement_data[0,arg_max]
                    bottom_cell.set_IL(bottom_cell_Isc)
                    bottom_cell.build_IV()
                    top_cell_Voc = tandem_cell_Voc - bottom_cell.get_Voc()
                    Jsc = top_cell_Isc/test_cell_area
                    J01, J02 = estimate_cell_J01_J02(Jsc=Jsc,Voc=top_cell_Voc,Si_intrinsic_limit=False)
                    # just arbitrarily guess the PC
                    J01_PC = J01*0.2
                    J01 = J01*0.8
                    top_cell = make_solar_cell(Jsc, J01, J02, area=test_cell_area, 
                            Si_intrinsic_limit=False,J01_photon_coupling=J01_PC)
                    top_cell.set_Suns(1.0)
                    break
        if bottom_cell is not None and top_cell is not None:
            tandem_cell = MultiJunctionCell([bottom_cell,top_cell])

    tandem_cell.assign_measurements(measurements)

    fit_parameters = Tandem_Cell_Fit_Parameters(tandem_cell)

    fit_dashboard = None
    if use_fit_dashboard:
        fit_dashboard = Fit_Dashboard(3,3,save_file_name=prefix)
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"white"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"black"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"blue"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"blue"})
        fit_dashboard.define_plot_what(which_axs=1, measurement_type=Suns_Voc_measurement, 
                                    measurement_condition={'spectrum':"red"}, 
                                    plot_type="overlap_curves",
                                        title="Suns-Voc", plot_style_parameters={"color":"red"})
        fit_dashboard.define_plot_what(which_axs=2, measurement_type=Dark_IV_measurement, 
                                    measurement_condition={'spectrum':"red"}, 
                                    plot_type="overlap_curves",
                                        title="Dark IV", plot_style_parameters={"color":"red"})
        fit_dashboard.define_plot_what(which_axs=2, measurement_type=Dark_IV_measurement, 
                                    measurement_condition={'spectrum':"blue"}, 
                                    plot_type="overlap_curves",
                                        title="Dark IV", plot_style_parameters={"color":"blue"})
        for i, key in enumerate(Light_IV_measurement.keys):
            for j, Suns in enumerate([1.0, 0.5]):
                fit_dashboard.define_plot_what(which_axs=3*(j+1)+i, measurement_type=Light_IV_measurement, 
                                            key_parameter=key, 
                                            measurement_condition={'Suns':Suns},
                                            plot_type="overlap_key_parameter",
                                            x_axis="I_imbalance",
                                            title=str(Suns)+" Suns IV", plot_style_parameters={"color":"blue"})
                
    result = fit_routine(tandem_cell,fit_parameters,
                routine_functions={
                    "update_function":linear_regression,
                    "comparison_function":compare_experiments_to_simulations,
                },
                fit_dashboard=fit_dashboard,
                aux=aux,num_of_epochs=num_of_rounds)
    if num_of_rounds==0:
        return result
    fit_parameters.apply_to_ref(aux)
    interactive_fit_dashboard = None
    if use_fit_dashboard:
        interactive_fit_dashboard = Interactive_Fit_Dashboard(tandem_cell,fit_parameters,ref_fit_dashboard=fit_dashboard)
    return fit_parameters.ref_sample, interactive_fit_dashboard

def generate_differentials(measurements,tandem_cell):
    return analyze_tandem_cell_measurements(measurements,num_of_rounds=0,tandem_cell_starting_guess=tandem_cell)
