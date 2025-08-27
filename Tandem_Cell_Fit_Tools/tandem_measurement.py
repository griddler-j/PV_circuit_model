from PV_Circuit_Model.measurement import *
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.data_fitting import *
import numpy as np

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
        self.initialize([np.log10(sample.cells[0].J01()), np.log10(sample.cells[0].J02()), 
                          np.log10(sample.cells[0].specific_shunt_cond()), 
                          np.log10(sample.cells[1].J01()), np.log10(sample.cells[1].J02()), 
                          np.log10(sample.cells[1].PC_J01()), 
                          np.log10(sample.cells[1].specific_shunt_cond()), 
                          np.log10(sample.specific_Rs_cond())],
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


