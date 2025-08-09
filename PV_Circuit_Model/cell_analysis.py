import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.multi_junction_cell import *
from matplotlib import pyplot as plt

def get_Voc(argument):
    if isinstance(argument,CircuitGroup) and hasattr(argument,"IV_parameters") and "Voc" in argument.IV_parameters:
        return argument.IV_parameters["Voc"]
    if isinstance(argument,CircuitGroup):
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    Voc = interp_(0,IV_curve[1,:],IV_curve[0,:])
    if isinstance(argument,CircuitGroup):
        if not hasattr(argument,"IV_parameters"):
            argument.IV_parameters = {}
        argument.IV_parameters["Voc"] = Voc
    return Voc
CircuitGroup.get_Voc = get_Voc

def get_Isc(argument):
    if isinstance(argument,CircuitGroup) and hasattr(argument,"IV_parameters") and "Isc" in argument.IV_parameters:
        return argument.IV_parameters["Isc"]
    if isinstance(argument,CircuitGroup):
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    Isc = -interp_(0,IV_curve[0,:],IV_curve[1,:])
    if isinstance(argument,CircuitGroup):
        if not hasattr(argument,"IV_parameters"):
            argument.IV_parameters = {}
        argument.IV_parameters["Isc"] = Isc
    return Isc
CircuitGroup.get_Isc = get_Isc

def get_Jsc(argument):
    Jsc = argument.get_Isc()
    if hasattr(argument,"area"):
        Jsc /= argument.area
    return Jsc
CircuitGroup.get_Jsc = get_Jsc

def get_Pmax(argument, return_op_point=False):
    if isinstance(argument,CircuitGroup):
        if hasattr(argument,"IV_parameters") and "Pmax" in argument.IV_parameters:
            Pmax = argument.IV_parameters["Pmax"]
            Vmp = argument.IV_parameters["Vmp"]
            Imp = argument.IV_parameters["Imp"]
            if return_op_point:
                return Pmax, Vmp, Imp
            return Pmax
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
        return_op_point = True
    V = IV_curve[0,:]
    I = IV_curve[1,:]
    power = -V*I
    index = np.argmax(power)
    V = np.linspace(IV_curve[0,index-1],IV_curve[0,index+1],1000)
    I = interp_(V,IV_curve[0,:],IV_curve[1,:])
    power = -V*I
    index = np.argmax(power)
    Vmp = V[index]
    Imp = I[index]
    if isinstance(argument,CircuitGroup):
        argument.set_operating_point(V=Vmp, refine_IV=True)
        if argument.IV_table is None:
            argument.build_IV()
            IV_curve = argument.IV_table
            V = IV_curve[0,:]
            I = IV_curve[1,:]
            power = -V*I
            index = np.argmax(power)
            V = np.linspace(IV_curve[0,index-1],IV_curve[0,index+1],1000)
            I = interp_(V,IV_curve[0,:],IV_curve[1,:])
            power = -V*I
            index = np.argmax(power)
            Vmp = V[index]
            Imp = I[index]
    Pmax = power[index]
    if isinstance(argument,CircuitGroup):
        if not hasattr(argument,"IV_parameters"):
            argument.IV_parameters = {}
        argument.IV_parameters["Pmax"] = Pmax
        argument.IV_parameters["Vmp"] = Vmp
        argument.IV_parameters["Imp"] = Imp
    if return_op_point:
        return Pmax, Vmp, Imp
    return Pmax
CircuitGroup.get_Pmax = get_Pmax

def get_Eff(argument):
    Eff = argument.get_Pmax()
    if hasattr(argument,"area"):
        Eff *= 10.0/argument.area
    return Eff
CircuitGroup.get_Eff = get_Eff

def get_FF(argument):
    if isinstance(argument,CircuitGroup):
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    Voc = get_Voc(IV_curve)
    Isc = get_Isc(IV_curve)
    Pmax, _, _ = get_Pmax(IV_curve)
    return Pmax/(Isc*Voc)
CircuitGroup.get_FF = get_FF

def Rs_extraction_two_light_IVs(IV_curves):
    Isc0 = -1*get_Isc(IV_curves[0])
    Isc1 = -1*get_Isc(IV_curves[1])
    _, Vmp0, Imp0 = get_Pmax(IV_curves[0])
    delta_I = -Isc0+Imp0
    delta_Is_halfSun = -Isc1+IV_curves[1][1,:]
    V_point = np.interp(delta_I,delta_Is_halfSun,IV_curves[1][0,:])
    Rs = (Vmp0-V_point)/(Isc0-Isc1)
    return Rs

def Rshunt_extraction(IV_curve,base_point=0):
    base_point = max(base_point,np.min(IV_curve[0,:]))
    indices = np.where((IV_curve[0,:]>=base_point) & (IV_curve[0,:]<=base_point+0.1))[0]
    indices = list(indices)
    if len(indices)<2:
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

def estimate_cell_J01_J02(Jsc,Voc,Pmax=None,FF=1.0,Rs=0.0,Rshunt=1e6,
                          temperature=25,Sun=1.0,thickness=180e-4,Si_intrinsic_limit=True):
    if Pmax is None:
        Pmax = Jsc*Voc*FF          
    VT = get_VT(temperature)
    max_J01 = Jsc/np.exp(Voc/VT)
    for inner_k in range(100):
        trial_cell = make_solar_cell(Jsc, max_J01, 0.0, Rshunt, 
                                     Rs, thickness=thickness,Si_intrinsic_limit=Si_intrinsic_limit)
        trial_cell.set_temperature(temperature,rebuild_IV=False)
        trial_cell.set_Suns(Sun)
        Voc_ = trial_cell.get_Voc()
        if abs(Voc_-Voc) < 1e-10:
            break 
        max_J01 *= np.exp((Voc_-Voc)/VT)
    max_J02 = Jsc/np.exp(Voc/(2*VT))
    for inner_k in range(100):
        trial_cell = make_solar_cell(Jsc, 0.0, max_J02, Rshunt, Rs, 
                                     thickness=thickness,Si_intrinsic_limit=Si_intrinsic_limit)
        trial_cell.set_temperature(temperature,rebuild_IV=False)
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
            trial_J01 = interp_(Pmax, outer_record_[:,1], outer_record_[:,0])
            trial_J01 = max(trial_J01, 0.0)
            trial_J01 = min(trial_J01, max_J01)
        inner_record = []
        for inner_k in range(100):
            if inner_k==0:
                trial_J02 = 0.0
            elif inner_k==1:
                trial_J02 = max_J02
            else:
                inner_record_ = np.array(inner_record)
                indices = np.argsort(inner_record_[:,0])
                inner_record_ = inner_record_[indices,:]
                trial_J02 = interp_(Voc, inner_record_[:,1], inner_record_[:,0])
                trial_J02 = max(trial_J02, 0.0)
                trial_J02 = min(trial_J02, max_J02)
            trial_cell = make_solar_cell(Jsc, trial_J01, trial_J02, Rshunt, Rs,thickness=thickness,
                                         Si_intrinsic_limit=Si_intrinsic_limit)
            trial_cell.set_temperature(temperature,rebuild_IV=False)
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

def plot(self, fourth_quadrant=True, show_IV_parameters=True, title="I-V Curve"):
    if fourth_quadrant and isinstance(self,CircuitGroup):
        self.get_Pmax()
        Voc = self.get_Voc()
        Isc = self.get_Isc()
        plt.plot(self.IV_table[0,:],-self.IV_table[1,:])
        if self.operating_point is not None:
            plt.plot(self.operating_point[0],-self.operating_point[1],marker='o')
            if len(self.operating_point)==3:
                plt.plot(self.operating_point[2],-self.operating_point[1],marker='o')
        plt.xlim((0,Voc*1.1))
        plt.ylim((0,Isc*1.1))
    else:
        plt.plot(self.IV_table[0,:],self.IV_table[1,:])
        if self.operating_point is not None:
            plt.plot(self.operating_point[0],self.operating_point[1],marker='o')
            if len(self.operating_point)==3:
                plt.plot(self.operating_point[2],self.operating_point[1],marker='o')
    if show_IV_parameters and fourth_quadrant and (isinstance(self,Cell) or isinstance(self,Module) or isinstance(self,MultiJunctionCell) or self.__class__.__name__=="Module" or self.__class__.__name__=="Cell" or self.__class__.__name__=="MultiJunctionCell"):
        max_power, Vmp, Imp = self.get_Pmax(return_op_point=True)
        Voc = self.get_Voc()
        Isc = self.get_Isc()
        FF = self.get_FF()
        y_space = 0.07
        plt.plot(Voc,0,marker='o',color="blue")
        plt.plot(0,Isc,marker='o',color="blue")
        if fourth_quadrant:
            Imp *= -1
        plt.plot(Vmp,Imp,marker='o',color="blue")
        if (isinstance(self,Cell) or isinstance(self,MultiJunctionCell) or self.__class__.__name__=="Cell" or self.__class__.__name__=="MultiJunctionCell"):
            plt.text(Voc*0.05, Isc*(0.8-0*y_space), f"Isc = {Isc:.3f} A")
            plt.text(Voc*0.05, Isc*(0.8-1*y_space), f"Jsc = {Isc/self.area*1000:.3f} mA/cm2")
            plt.text(Voc*0.05, Isc*(0.8-2*y_space), f"Voc = {Voc:.4f} V")
            plt.text(Voc*0.05, Isc*(0.8-3*y_space), f"FF = {FF*100:.3f} %")
            plt.text(Voc*0.05, Isc*(0.8-4*y_space), f"Pmax = {max_power:.3f} W")
            plt.text(Voc*0.05, Isc*(0.8-5*y_space), f"Eff = {max_power/self.area*1000:.3f} %")
            plt.text(Voc*0.05, Isc*(0.8-6*y_space), f"Area = {self.area:.3f} cm2")
        else:
            plt.text(Voc*0.05, Isc*(0.8-0*y_space), f"Isc = {Isc:.3f} A")
            plt.text(Voc*0.05, Isc*(0.8-1*y_space), f"Voc = {Voc:.2f} V")
            plt.text(Voc*0.05, Isc*(0.8-2*y_space), f"FF = {FF*100:.3f} %")
            plt.text(Voc*0.05, Isc*(0.8-3*y_space), f"Pmax = {max_power:.2f} W")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.gcf().canvas.manager.set_window_title(title)
CircuitGroup.plot = plot
CircuitElement.plot = plot

def show(self):
    plt.show()
CircuitGroup.show = show
CircuitElement.show = show

def quick_solar_cell(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, thickness=160e-4, wafer_format="M10",half_cut=True):
    shape, area = wafer_shape(format=wafer_format)
    J01, J02 = estimate_cell_J01_J02(Jsc,Voc,FF=FF,Rs=Rs,Rshunt=Rshunt,thickness=thickness)
    return make_solar_cell(Jsc, J01, J02, Rshunt, Rs, area, shape, thickness)

def quick_butterfly_module(Isc=None, Voc=None, FF=None, Pmax=None, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24):
    shape, area = wafer_shape(format=wafer_format)
    Jsc = 0.042
    if Isc is not None:
        Jsc = Isc / area /2
    else:
        Isc = Jsc * area * 2 
    cell_Voc = 0.735
    if Voc is not None:
        cell_Voc = Voc / (num_strings*num_cells_per_halfstring)
    target_Pmax = 0.8*cell_Voc*Isc
    if Pmax is not None:
        target_Pmax = Pmax
    elif FF is not None:
        target_Pmax = Voc*Isc*FF
    try_FF = target_Pmax/Isc/Voc
    record = []
    for _ in tqdm(range(20),desc="Tweaking module cell parameters..."):
        cell = quick_solar_cell(Jsc=Jsc, Voc=cell_Voc, FF=try_FF, wafer_format=wafer_format,half_cut=True)
        cells = [circuit_deepcopy(cell) for _ in range(2*num_strings*num_cells_per_halfstring)]
        module = make_butterfly_module(cells, num_strings=num_strings, num_cells_per_halfstring=num_cells_per_halfstring)
        module.set_Suns(1.0,rebuild_IV=False)
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
