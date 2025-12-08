import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.multi_junction_cell import *
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        from IPython.display import display
        return get_ipython() is not None and hasattr(get_ipython(), "kernel")
    except Exception:
        return False

IN_NOTEBOOK = _in_notebook()
if not IN_NOTEBOOK:
    matplotlib.use("TkAgg")

BASE_UNITS = {
    "Pmax": ("W",  "W"),
    "Vmp":  ("V",  "V"),
    "Imp":  ("A",  "A"),
    "Voc":  ("V",  "V"),
    "Isc":  ("A",  "A"),
    "FF":   ("%",  r"\%"),
    "Eff":   ("%",  r"\%"),
    "Area": ("cm²", r"cm$^2$"),
    "Jsc":  ("mA/cm²", r"mA/cm$^2$"),
    "Jmp":  ("mA/cm²", r"mA/cm$^2$"),
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

solver_env_variables = ParameterSet.get_set("solver_env_variables")
REFINE_V_HALF_WIDTH = solver_env_variables["REFINE_V_HALF_WIDTH"]


def get_Voc(argument):
    if isinstance(argument,CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    Voc = interp_(0,IV_curve[1,:],IV_curve[0,:])
    return Voc
CircuitGroup.get_Voc = get_Voc

def get_Isc(argument):
    if isinstance(argument,CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    Isc = -interp_(0,IV_curve[0,:],IV_curve[1,:])
    return Isc
CircuitGroup.get_Isc = get_Isc

def get_Jsc(argument):
    Jsc = argument.get_Isc()
    if hasattr(argument,"area"):
        Jsc /= argument.area
    return Jsc
CircuitGroup.get_Jsc = get_Jsc


def get_Pmax(argument, return_op_point=False, refine_IV=True):
    if isinstance(argument,CircuitGroup):
        if argument.IV_V is None:
            argument.build_IV()
        IV_curve = argument.IV_table
    else:
        IV_curve = argument
    V = IV_curve[0,:]
    I = IV_curve[1,:]
    power = -V*I
    index = np.argmax(power)
    V = np.linspace(IV_curve[0,index-1],IV_curve[0,index+1],1000)
    I = interp_(V,IV_curve[0,:],IV_curve[1,:])
    power = -V*I
    index = np.argmax(power)
    Vmp = V[index]
    
    if isinstance(argument,CircuitGroup) and refine_IV and not argument.refined_IV:
        if not hasattr(argument,"job_heap"):
            argument.build_IV()
        argument.set_operating_point(V=Vmp, refine_IV=refine_IV)
        IV_V = argument.IV_V
        IV_I = argument.IV_I
        power = -IV_V*IV_I
        index = np.argmax(power)
        V = np.linspace(IV_V[index-1],IV_I[index+1],1000)
        I = interp_(V,IV_V,IV_I)
        power = -V*I
        index = np.argmax(power)
    Vmp = V[index]
    Imp = I[index]
    Pmax = power[index]
    if isinstance(argument,CircuitGroup):
        argument.operating_point = [Vmp,Imp]

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
    Voc = get_Voc(argument)
    Isc = get_Isc(argument)
    Pmax = get_Pmax(argument)
    FF = Pmax/(Isc*Voc)
    return FF
CircuitGroup.get_FF = get_FF

def Rs_extraction_two_light_IVs(IV_curves):
    Isc0 = -1*get_Isc(IV_curves[0])
    Isc1 = -1*get_Isc(IV_curves[1])
    _, Vmp0, Imp0 = get_Pmax(IV_curves[0],return_op_point=True)
    delta_I = -Isc0+Imp0
    delta_Is_halfSun = -Isc1+IV_curves[1][1,:]
    V_point = np.interp(delta_I,delta_Is_halfSun,IV_curves[1][0,:])
    Rs = (Vmp0-V_point)/(Isc0-Isc1)
    return Rs

def Rshunt_extraction(IV_curve,base_point=0):
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

def estimate_cell_J01_J02(Jsc,Voc,Pmax=None,FF=1.0,Rs=0.0,Rshunt=1e6,
                          temperature=25,Sun=1.0,Si_intrinsic_limit=True,**kwargs):
    if Pmax is None:
        Pmax = Jsc*Voc*FF          
    VT = get_VT(temperature,VT_at_25C)
    max_J01 = Jsc/np.exp(Voc/VT)
    for inner_k in range(100):
        trial_cell = make_solar_cell(Jsc, max_J01, 0.0, Rshunt, 
                                     Rs, Si_intrinsic_limit=Si_intrinsic_limit, **kwargs)
        trial_cell.set_temperature(temperature)
        trial_cell.set_Suns(Sun)
        Voc_ = trial_cell.get_Voc()
        if abs(Voc_-Voc) < 1e-10:
            break 
        max_J01 *= np.exp((Voc_-Voc)/VT)
    max_J02 = Jsc/np.exp(Voc/(2*VT))
    for inner_k in range(100):
        trial_cell = make_solar_cell(Jsc, 0.0, max_J02, Rshunt, Rs, 
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
            trial_cell = make_solar_cell(Jsc, trial_J01, trial_J02, Rshunt, Rs,
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

def get_IV_parameter_words(self, display_or_latex=0, cell_or_module=0, cap_decimals=True, include_bounds=False):
    words = {}
    curves = {"normal":np.array([self.IV_V,self.IV_I])}
    if hasattr(self,"IV_V_lower"):
        curves["lower"] = np.array([self.IV_V_lower,self.IV_I_lower])
        curves["upper"] = np.array([self.IV_V_upper,self.IV_I_upper])
    all_parameters = {}
    for key, _ in curves.items():
        all_parameters[key] = {}
        parameters = all_parameters[key]
        parameters["Pmax"], parameters["Vmp"], parameters["Imp"] = get_Pmax(curves[key],return_op_point=True)
        parameters["Imp"] *= -1
        parameters["Voc"] = get_Voc(curves[key])
        parameters["Isc"] = get_Isc(curves[key])
        parameters["FF"] = get_FF(curves[key])*100
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
            decimals = 6
        words[key] = f"{key} = {value:.{decimals}f}{error_word} {BASE_UNITS[key][display_or_latex]}"
        if hasattr(self,"IV_V_lower") and include_bounds:
            words[key] += f" (from lower bound curve: {all_parameters['lower'][key]:.{decimals}f} {BASE_UNITS[key][display_or_latex]}"
            words[key] += f", from upper bound curve: {all_parameters['upper'][key]:.{decimals}f} {BASE_UNITS[key][display_or_latex]})"
    return words, parameters

def plot(self, fourth_quadrant=True, show_IV_parameters=True, title="I-V Curve", show_solver_summary=True):
    if self.IV_V is None:
        self.build_IV()
    if (fourth_quadrant or show_IV_parameters) and isinstance(self,CircuitGroup):
        _, Vmp, Imp = self.get_Pmax(return_op_point=True)
        Voc = self.get_Voc()
        Isc = self.get_Isc()
        FF = self.get_FF()
    bottom_up_operating_point = getattr(self,"bottom_up_operating_point",None)
    normalized_operating_point = getattr(self,"bottom_up_operating_point",None)
    find_near_op = None
    if bottom_up_operating_point:
        operating_point_V = self.operating_point[0]
        bottom_up_operating_point_V = bottom_up_operating_point[0]
        normalized_op_pt_V = normalized_operating_point[0]
        left_V = min(operating_point_V,bottom_up_operating_point_V)
        right_V = max(operating_point_V,bottom_up_operating_point_V)
        left_V -= normalized_op_pt_V*REFINE_V_HALF_WIDTH
        right_V += normalized_op_pt_V*REFINE_V_HALF_WIDTH
        find_near_op = np.where((self.IV_V >= left_V) & (self.IV_V <= right_V))[0]
    if fourth_quadrant and isinstance(self,CircuitGroup):
        fig, ax1 = plt.subplots()

        # Left Y-axis
        # if bottom_up_operating_point:
        #     ax1.plot(self.IV_V_lower,-self.IV_I_lower,color="green")
        #     ax1.plot(self.IV_V_upper,-self.IV_I_upper,color="gray")
        ax1.plot(self.IV_V,-self.IV_I)
        if find_near_op is not None:
            ax1.plot(self.IV_V[find_near_op],-self.IV_I[find_near_op],color="red")
        if self.operating_point is not None:
            ax1.plot(self.operating_point[0],-self.operating_point[1],marker='o',color="blue")
        ax1.set_xlim((0,Voc*1.1))
        ax1.set_ylim((0,Isc*1.1))
        ax1.set_xlabel("Voltage (V)")
        ax1.set_ylabel("Current (A)")
        ax2 = ax1.twinx()

        P = -self.IV_V*self.IV_I
        # Right Y-axis (shares same X)
        ax2.plot(self.IV_V,P,color="orange")
        if find_near_op is not None:
            ax2.plot(self.IV_V[find_near_op],P[find_near_op],color="red")
        if self.operating_point is not None:
            ax2.plot(self.operating_point[0],-self.operating_point[0]*self.operating_point[1],marker='o',color="orange")
        ax2.set_ylim((0,np.max(P)*1.1))
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, pos: f"{x:.0e}")
        )
        ax2.set_ylabel("Power (W)")
        if show_IV_parameters and fourth_quadrant and isinstance(self,CircuitGroup):
            cell_or_module=1
            params = ["Isc","Voc","FF","Pmax"]
            if self._type_number==6 or self._type_number==7: # cell or MJ cell
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
        plt.tight_layout()

        if show_solver_summary:
            self.show_solver_summary(fig=fig)

    else:
        plt.plot(self.IV_V,self.IV_I)
        if find_near_op is not None:
            plt.plot(self.IV_V[find_near_op],self.IV_I[find_near_op],color="red")
        if self.operating_point is not None:
            plt.plot(self.operating_point[0],self.operating_point[1],marker='o')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        if show_IV_parameters and fourth_quadrant and isinstance(self,CircuitGroup):
            cell_or_module=1
            params = ["Isc","Voc","FF","Pmax"]
            if self._type_number==6 or self._type_number==7: # cell or MJ cell
                cell_or_module=0
                params = ["Isc","Jsc","Voc","FF","Pmax","Eff","Area"]
            words, _ = get_IV_parameter_words(self, display_or_latex=0, cell_or_module=cell_or_module, cap_decimals=True)
        
            y_space = 0.07
            plt.plot(Voc,0,marker='o',color="blue")
            plt.plot(0,Isc,marker='o',color="blue")
            if fourth_quadrant:
                Imp *= -1
            plt.plot(Vmp,Imp,marker='o',color="blue")
            for i, param in enumerate(params):
                plt.text(Voc*0.05, Isc*(0.8-i*y_space), words[param])
        
    
    plt.gcf().canvas.manager.set_window_title(title)
CircuitComponent.plot = plot

def show(self):
    # In notebooks, figures are auto-shown; don't block
    if IN_NOTEBOOK:
        plt.show(block=False)   # or even just `return`
    else:
        plt.show()
CircuitComponent.show = show

def quick_solar_cell(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, wafer_format="M10",half_cut=True, **kwargs):
    shape, area = wafer_shape(format=wafer_format,half_cut=half_cut)
    J01, J02 = estimate_cell_J01_J02(Jsc,Voc,FF=FF,Rs=Rs,Rshunt=Rshunt,**kwargs)
    return make_solar_cell(Jsc, J01, J02, Rshunt, Rs, area, shape, **kwargs)

def quick_module(Isc=None, Voc=None, FF=None, Pmax=None, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, special_conditions=None, half_cut=False, butterfly=False,**kwargs):
    force_n1 = False
    if special_conditions is not None:
        if "force_n1" in special_conditions:
            force_n1 = special_conditions["force_n1"]
    shape, area = wafer_shape(format=wafer_format, half_cut=half_cut)
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
        cells = [circuit_deepcopy(cell) for _ in range(cell_num_factor*num_strings*num_cells_per_halfstring)]
        try_R = 0.02
        record = []
        for _ in tqdm(range(20),desc="Tweaking module cell parameters..."):
            module = make_module(cells, num_strings=num_strings, num_cells_per_halfstring=num_cells_per_halfstring, halfstring_resistor = try_R, butterfly=butterfly)
            module.set_Suns(1.0)
            module.build_IV()
            Pmax = module.get_Pmax()
            record.append([try_R, Pmax, cell.get_Pmax()])
            if np.abs(Pmax-target_Pmax) < 1e-6:
                break
            record_ = np.array(record)
            record_ = record_[record_[:, 0].argsort()]
            if np.max(record_[:,1])>=target_Pmax and np.min(record_[:,1])<=target_Pmax:
                try_R = interp_(target_Pmax, record_[:,1], record_[:,0])
            elif np.max(record_[:,1])<target_Pmax:
                try_R /= 10
            else:
                try_R *= 10
    else:
        try_FF = target_Pmax/Isc/Voc
        record = []
        for _ in tqdm(range(20),desc="Tweaking module cell parameters..."):
            cell = quick_solar_cell(Jsc=Jsc, Voc=cell_Voc, FF=try_FF, wafer_format=wafer_format,half_cut=half_cut,**kwargs)
            cells = [circuit_deepcopy(cell) for _ in range(cell_num_factor*num_strings*num_cells_per_halfstring)]
            module = make_module(cells, num_strings=num_strings, num_cells_per_halfstring=num_cells_per_halfstring, butterfly=butterfly)
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

def quick_butterfly_module(Isc=None, Voc=None, FF=None, Pmax=None, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, special_conditions=None, half_cut=True,**kwargs):
    return quick_module(Isc, Voc, FF, Pmax, wafer_format, num_strings, num_cells_per_halfstring, special_conditions, half_cut, butterfly=True,**kwargs)

def quick_tandem_cell(Jscs=[0.019,0.020], Vocs=[0.710,1.2], FFs=[0.8,0.78], Rss=[0.3333,0.5], Rshunts=[1e6,5e4], thicknesses=[160e-4,1e-6], wafer_format="M10",half_cut=True):
    shape, area = wafer_shape(format=wafer_format)
    cells = []
    for i in range(len(Jscs)):
        Si_intrinsic_limit = True
        if i > 0:
            Si_intrinsic_limit = False
        J01, J02 = estimate_cell_J01_J02(Jscs[i],Vocs[i],FF=FFs[i],Rs=Rss[i],Rshunt=Rshunts[i],Si_intrinsic_limit=Si_intrinsic_limit,thickness=thicknesses[i])
        cells.append(make_solar_cell(Jscs[i], J01, J02, Rshunts[i], Rss[i], area, shape, thickness=thicknesses[i]))
    return MultiJunctionCell(cells)

def solver_summary_heap(job_heap): 
    build_time = job_heap.timers["build"]
    IV_time = job_heap.timers["IV"]
    refine_time = job_heap.timers["refine"]
    bounds_time = job_heap.timers["bounds"]
    component = job_heap.components[0]
    if component.refined_IV:
        paragraph = "I-V Parameters:\n"
    else:
        paragraph = "I-V Parameters (coarse - run device.get_Pmax() to get refinement!):\n"
    cell_or_module=1
    params = ["Isc","Imp","Voc","Vmp","FF","Pmax"]
    if component._type_number==6 or component._type_number==7: # cell or MJ cell
        cell_or_module=0
        params = ["Isc","Jsc","Imp","Jmp","Voc","Vmp","FF","Pmax","Eff","Area"]
    words, _ = get_IV_parameter_words(component, display_or_latex=0, cell_or_module=cell_or_module, cap_decimals=False, include_bounds=True)
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
        paragraph += f"Kirchhoff’s Voltage Law deviation: V error <= {worst_V_error:.3e} V\n"
        paragraph += f"Kirchhoff’s Current Law deviation: I error <= {worst_I_error:.3e} A\n"
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

def solver_summary(self):
    paragraph = "----------------------------------------------------------------------------\n"
    paragraph += "I-V Solver Summary for "
    if getattr(self,"name",None) is not None and self.name != "":
        paragraph += f"{self.name} of "
    paragraph += f" type {type(self).__name__}:\n"
    paragraph += "----------------------------------------------------------------------------\n"
    if hasattr(self,"job_heap") and self.IV_V is not None:
        paragraph += solver_summary_heap(self.job_heap)
    else:
        paragraph += "I-V Curve has not been calculated\n"
    paragraph += "----------------------------------------------------------------------------\n"
    paragraph += f"CircuitGroup Information:\n"
    paragraph += f"Circuit Depth: {self.circuit_depth}\n"
    paragraph += f"Number of Circuit Elements: {self.num_circuit_elements}\n"
    paragraph += "----------------------------------------------------------------------------\n"
    paragraph += "Solver Environment Variables:\n"
    solver_env_variables_dict = ParameterSet.get_set("solver_env_variables")()
    for key, value in solver_env_variables_dict.items():
        paragraph += f"{key}: {value}\n"
    paragraph += "----------------------------------------------------------------------------\n"

    return paragraph

def show_solver_summary(self, fig=None):
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

CircuitGroup.solver_summary = solver_summary
CircuitGroup.show_solver_summary = show_solver_summary