import numpy as np
from matplotlib import pyplot as plt
from PV_Circuit_Model.utilities import *
from PV_Circuit_Model.iterative_solver import assign_nodes, iterative_solve
import copy
from tqdm import tqdm
from PV_Circuit_Model.IV_jobs import *

pbar = None
x_spacing = 1.5
y_spacing = 0.2

class const():
    VT = 0.02568 # thermal voltage at 25C

def get_VT(temperature):
    return const.VT*(temperature + 273.15)/(25 + 273.15)

def get_ni(temperature):
    return 9.15e19*((temperature+273.15)/300)**2*np.exp(-6880/(temperature+273.15))
    
class CircuitComponent:
    def __init__(self,tag=None):
        self.IV_table = None
        self.operating_point = None #V,I
        self.circuit_diagram_extent = [0, 0.8]
        self.parent = None
        self.aux = {}
    def null_IV(self, keep_dark=False):
        self.refined_IV = False
        if hasattr(self,"IV_parameters"):
            del self.IV_parameters
        self.IV_table = None
        if keep_dark==False and hasattr(self,"dark_IV_table"):
            self.dark_IV_table = None
        if self.parent is not None:
            self.parent.null_IV(keep_dark=keep_dark)

class CircuitElement(CircuitComponent):
    def __init__(self,tag=None):
        self.IV_table = None
        self.tag = tag
        self.operating_point = None #V,I
        self.circuit_diagram_extent = [0, 0.8]
        self.parent = None
        self.aux = {}
    def set_operating_point(self,V=None,I=None,refine_IV=False,top_level=True):
        if V is not None:
            I = interp_(V,self.IV_table[0,:],self.IV_table[1,:])
        elif I is not None:
            V = interp_(I,self.IV_table[1,:],self.IV_table[0,:])
        self.operating_point = [V,I]
        if refine_IV and isinstance(self,Diode):
            findright_ = np.where(self.IV_table[0,:] > V)[0]
            findleft_ = np.where(self.IV_table[0,:] < V)[0]
            if len(findright_)>2 and self.IV_table[0,findright_[1]]-V < 0.001/100*5 and len(findleft_)>2 and V-self.IV_table[0,findleft_[-2]] < 0.001/100*5:
                return # already refined IV before, good
            Vs = self.get_V_range()
            if isinstance(self,ReverseDiode):
                V_range = np.sort(np.concatenate([Vs, np.linspace(-V - 0.001, -V + 0.001, 100)]))
            else:
                V_range = np.sort(np.concatenate([Vs, np.linspace(V - 0.001, V + 0.001, 100)]))
            self.build_IV(V=V_range)
            if top_level:
                if V is not None:
                    I = interp_(V,self.IV_table[0,:],self.IV_table[1,:])
                elif I is not None:
                    V = interp_(I,self.IV_table[1,:],self.IV_table[0,:])
                self.operating_point = [V,I]
            if self.parent is not None:
                self.parent.null_IV(keep_dark=False)
    def get_value_text(self):
        pass
    def get_draw_func(self):
        pass
    def draw(self, ax=None, x=0, y=0, color="black", display_value=False):
        text = None
        if display_value:
            text = self.get_value_text()
        draw_symbol(self.get_draw_func(),ax=ax,x=x,y=y,color=color,text=text)
        if "pos_node" in self.aux:
            ax.text(x,y-0.5,str(self.aux["neg_node"]), va='center', fontsize=6)
            ax.text(x,y+0.5,str(self.aux["pos_node"]), va='center', fontsize=6)
    def calc_I(self,V):
        pass
    def calc_dI_dV(self,V):
        pass

class CurrentSource(CircuitElement):
    def __init__(self, IL, Suns=1.0, temperature=25, temp_coeff=0.0, tag=None):
        super().__init__(tag=tag)
        if np.isnan(IL):
            assert(1==0)
        self.IL = IL
        self.refSuns = Suns
        self.Suns = Suns
        self.refIL = IL
        self.refT = temperature
        self.T = temperature
        self.temp_coeff = temp_coeff

    def calc_I(self,V):
        if isinstance(V,numbers.Number):
            return -self.IL
        else:
            return -self.IL*np.ones_like(V)
    
    def calc_dI_dV(self,V):
        if isinstance(V,numbers.Number):
            return 0.0
        else:
            return np.zeros_like(V)

    def set_IL(self,IL):
        self.IL = IL
        keep_dark = True
        if IL > 0 and self.parent is not None:
            forward_diodes = self.parent.findElementType(ForwardDiode)
            for diode in forward_diodes:
                while diode.max_I < 2*IL:
                    keep_dark = False
                    diode.max_I *= 2
            if not keep_dark:
                for diode in forward_diodes:
                    diode.build_IV()
        self.null_IV(keep_dark=keep_dark)
        self.build_IV()

    def copy(self,source):
        self.refSuns = source.refSuns
        self.Suns = source.Suns
        self.refIL = source.refIL
        self.refT = source.refT
        self.T = source.T
        self.temp_coeff = source.temp_coeff
        self.set_IL(source.IL)

    def changeTemperatureAndSuns(self,temperature=None,Suns=None,rebuild_IV=True):
        if Suns is not None:
            self.Suns = Suns
        if temperature is not None:
            self.T = temperature
        self.set_IL(self.Suns*(self.refIL / self.refSuns + self.temp_coeff * (self.T - self.refT)))
        if rebuild_IV:
            self.build_IV()

    def build_IV(self, V=np.array([-0.1,0.1]), *args, **kwargs):
        # job_heap = IV_Job_Heap(self)
        # job_heap.run_jobs()
        # return
        self.IV_table = np.array([V, self.calc_I(V)])

    def __str__(self):
        return "Current Source: IL = " + self.get_value_text()
    
    def get_value_text(self):
        return f"{self.IL:.4f} A"
    def get_draw_func(self):
        return draw_CC_symbol

class Resistor(CircuitElement):
    def __init__(self, cond=1, tag=None):
        super().__init__(tag=tag)
        self.cond = cond
    def calc_I(self,V):
        return V*self.cond
    def calc_dI_dV(self,V):
        if isinstance(V,numbers.Number):
            return self.cond
        else:
            return self.cond*np.ones_like(V)
    def build_IV(self, V=np.array([-0.1,-0.05,0,0.05,0.1]), *args, **kwargs):
        # job_heap = IV_Job_Heap(self)
        # job_heap.run_jobs()
        # return
        self.IV_table = np.array([V, self.calc_I(V)])
    def set_cond(self,cond):
        self.cond = cond
        self.null_IV()
    def copy(self,source):
        self.set_cond(source.cond)
    def __str__(self):
        return "Resistor: R = " + self.get_value_text()
    def get_value_text(self):
        R = 1/self.cond
        if "area" in self.aux:
            R *= self.aux["area"]
        word = f"{R:.3f}"
        if "error" in self.aux and not np.isnan(self.aux["error"]):
            R_cond_error = self.aux["error"]
            R_error = R**2*R_cond_error
            word += f"\n\u00B1{R_error:.3f}"
        word += " ohm"
        return word
    def get_draw_func(self):
        return draw_resistor_symbol

class Diode(CircuitElement):
    def __init__(self,I0=1e-15,n=1,V_shift=0,tag=None,temperature=25): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(tag=tag)
        self.I0 = I0
        self.n = n
        self.V_shift = V_shift
        self.VT = get_VT(temperature)
        self.refI0 = I0
        self.refT = temperature

    def set_I0(self,I0):
        self.I0 = I0
        self.null_IV()

    def copy(self,source):
        self.n = source.n
        self.V_shift = source.V_shift
        self.VT = source.VT
        self.refI0 = source.refI0
        self.refT = source.refT
        self.set_I0(source.I0)

    def changeTemperature(self,temperature,rebuild_IV=True):
        self.VT = get_VT(temperature)
        old_ni  = get_ni(self.refT)
        new_ni  = get_ni(temperature)
        scale_factor = (new_ni/old_ni)**(2/self.n)
        self.set_I0(self.refI0*scale_factor)
        if rebuild_IV:
            self.build_IV()

    def get_V_range(self,max_num_points=100):
        if max_num_points is None:
            max_num_points = 100
        max_I = 0.2
        if hasattr(self,"max_I"):
            max_I = self.max_I
            max_num_points *= max_I/0.2
        # assume that 0.2 A/cm2 is max you'll need
        if self.I0==0:
            Voc = 10
        else:
            Voc = self.n*self.VT*np.log(max_I/self.I0)
        V = [self.V_shift-1.1,self.V_shift-1.0,self.V_shift,self.V_shift+0.02,self.V_shift+.08]+list(self.V_shift + Voc*np.log(np.arange(1,max_num_points))/np.log(max_num_points-1))
        V = np.array(V)
        return V

    def calc_I(self,V):
        return self.I0*(np.exp((V-self.V_shift)/(self.n*self.VT))-1)
    def calc_dI_dV(self,V):
        I = self.calc_I(V)
        return I/(self.n*self.VT)
    def build_IV(self, V=None, max_num_points=100, *args, **kwargs):
        # job_heap = IV_Job_Heap(self)
        # job_heap.run_jobs()
        # return
        if V is None:
            V = self.get_V_range(max_num_points)
        self.IV_table = np.array([V,self.calc_I(V)])
    
class ForwardDiode(Diode):
    def __init__(self,I0=1e-15,n=1,tag=None): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(I0, n, V_shift=0,tag=tag)
        self.max_I = 0.2
    def build_IV(self, V=None, max_num_points=100, *args, **kwargs):
        # job_heap = IV_Job_Heap(self)
        # job_heap.run_jobs()
        # return
        super().build_IV(V,max_num_points)
    def __str__(self):
        return "Forward Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n)
    def get_value_text(self):
        word = f"I0 = {self.I0:.3e}"
        if "error" in self.aux and not np.isnan(self.aux["error"]):
            word += f"\n\u00B1{self.aux['error']:.3e}"
        word += f" A\nn = {self.n:.2f}"
        return word
    def get_draw_func(self):
        return draw_forward_diode_symbol
    
class PhotonCouplingDiode(ForwardDiode):
    def get_draw_func(self):
        return draw_LED_diode_symbol
    def __str__(self):
        return "Photon Coupling Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n)

class ReverseDiode(Diode):
    def __init__(self,I0=1e-15,n=1, V_shift=0,tag=None): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(I0, n, V_shift, tag=tag)
    def calc_I(self,V):
        return -self.I0*np.exp((-V-self.V_shift)/(self.n*self.VT))
    def calc_dI_dV(self,V):
        I = self.calc_I(V)
        return -I/(self.n*self.VT)
    def build_IV(self, V=None, max_num_points=100, *args, **kwargs):
        # job_heap = IV_Job_Heap(self)
        # job_heap.run_jobs()
        # return
        if V is None:
            V = self.get_V_range(max_num_points)
        # I = self.I0*(np.exp((V-self.V_shift)/(self.n*self.VT))-1)
        self.IV_table = np.array([-V,self.calc_I(-V)])
        # self.IV_table[1,:] += self.I0
        # self.IV_table *= -1
        self.IV_table = self.IV_table[:,::-1]
    def __str__(self):
        return "Reverse Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n) + ", breakdown V = " + str(self.V_shift)
    def get_value_text(self):
        return f"I0 = {self.I0:.3e}A\nn = {self.n:.2f}\nbreakdown V = {self.V_shift:.2f}"
    def get_draw_func(self):
        return draw_reverse_diode_symbol

class CircuitGroup(CircuitComponent):
    def __init__(self,subgroups,connection="series",name=None,location=None,
                 rotation=0,x_mirror=1,y_mirror=1,extent=None):
        self.connection = connection
        self.subgroups = subgroups
        for element in self.subgroups:
            element.parent = self
        self.parent = None
        self.IV_table = None
        self.dark_IV_table = None
        self.name = name
        if location is None:
            self.location = np.array([0,0])
        else:
            self.location = location
        self.rotation = rotation
        self.x_mirror = x_mirror
        self.y_mirror = y_mirror
        if extent is not None:
            self.extent = extent
        else:
            self.extent = get_extent(subgroups)
        self.circuit_diagram_extent = get_circuit_diagram_extent(subgroups,connection)
        self.operating_point = None #V,I
        self.aux = {}
        self.is_circuit_group = True

    def add_element(self,element):
        self.subgroups.append(element)
        element.parent = self

    def null_all_IV(self):
        self.IV_table = None
        if hasattr(self,"dark_IV_table"):
            self.dark_IV_table = None
        for element in self.subgroups:
            if isinstance(element,CircuitElement):
                element.IV_table = None
            else:
                element.null_all_IV()

    def reassign_parents(self):
        for element in self.subgroups:
            element.parent = self
            if isinstance(element,CircuitGroup):
                element.reassign_parents()

    def set_operating_point(self,V=None,I=None,refine_IV=False,top_level=True, refine_op_point=True):
        refine_op_point_ = top_level and not refine_IV and refine_op_point
        refine_IV_ = refine_IV
        if hasattr(self,"refined_IV") and self.refined_IV:
            refine_IV_ = False
        if self.IV_table is None:
            self.build_IV()
        if V is not None:
            I_ = interp_(V,self.IV_table[0,:],self.IV_table[1,:])
            V_ = V
        elif I is not None:
            V_ = interp_(I,self.IV_table[1,:],self.IV_table[0,:])
            I_ = I
        for element in self.subgroups:
            if self.connection == "series": # then all elements have same current
                target_I = I_
                # solar cell needs to scale IV table by area
                if hasattr(self,"shape") and self.area is not None:
                    target_I /= self.area
                element.set_operating_point(V=None,I=target_I,refine_IV=refine_IV_,top_level=False)
            else: # then all elements have same voltage
                element.set_operating_point(V=V_,I=None,refine_IV=refine_IV_,top_level=False)
        if refine_IV_ and top_level:
            self.refined_IV = True
            if self.IV_table is None:
                self.build_IV()
            if V is not None:
                I_ = interp_(V,self.IV_table[0,:],self.IV_table[1,:])
                V_ = V
            elif I is not None:
                V_ = interp_(I,self.IV_table[1,:],self.IV_table[0,:])
                I_ = I
        if refine_op_point_:
            assign_nodes(self)
            op_point = iterative_solve(self,V=V,I=I)
            V_ = op_point[0]
            I_ = op_point[1]
            
        self.operating_point = [V_,I_]
        # cells also store Vint
        if hasattr(self,"shape"):
            self.operating_point.append(self.diode_branch.operating_point[0])

    def removeElementOfTag(self,tag):
        for element in self.subgroups[:]:
            if isinstance(element,CircuitElement):
                if element.tag==tag:
                    self.subgroups.remove(element)
            elif isinstance(element,CircuitGroup):
                element.removeElementOfTag(tag)
        self.null_IV()

    def set_temperature(self,temperature,rebuild_IV=True):
        diodes = self.findElementType(Diode)
        for diode in diodes:
            diode.changeTemperature(temperature,rebuild_IV=False)
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(temperature=temperature,rebuild_IV=False)
        if rebuild_IV:
            self.build_IV()

    def findElementType(self,type,serialize=False,path=[]):
        list_ = []
        if isinstance(type,str):
            type = eval(type)
        for i, element in enumerate(self.subgroups):
            if isinstance(element,type):
                list_.append(element)
            elif isinstance(element,CircuitGroup):
                list_.extend(element.findElementType(type,serialize=serialize))
        if serialize:
            for i, element in enumerate(list_):
                element.name = str(i)
        return list_
    
    def build_IV(self, max_num_points=None, cap_current=None):
        if hasattr(self,"IV_parameters"):
            del self.IV_parameters

        job_heap = IV_Job_Heap(self, max_num_points=max_num_points, cap_current=cap_current)
        job_heap.run_jobs()
        return

        # if solar cell, then express in current density
        Vints = None
        if hasattr(self,"shape") and self.area is not None and cap_current is not None:
            cap_current /= self.area
        all_circuit_element_children = True
        for element in self.subgroups:
            if isinstance(element,CircuitGroup):
                all_circuit_element_children = False
            if element.IV_table is None:
                element.build_IV(max_num_points=max_num_points,cap_current=cap_current)
        shift_IV_only = False 
        total_IL = 0.0       
        if self.connection=="series":
            extra_Is = []
            for iteration in [0,1]: # goes to 1 only if there is PC diode
                # add voltage
                Is = []
                for element in self.subgroups:
                    Is.extend(list(element.IV_table[1,:]))
                if iteration==1:
                    Is.extend(extra_Is)
                Is = np.sort(np.array(Is))
                if max_num_points is None:
                    Is = np.unique(Is)
                else:
                    tol = (Is[-1]-Is[0])/(max_num_points*1000)
                    quantized = np.round(Is / tol)
                    _, idx = np.unique(quantized, return_index=True)
                    Is = Is[idx]
                Vs = np.zeros_like(Is)
                Vints = np.zeros_like(Is)
                # do reverse order to allow for photon coupling
                pc_IVs = []
                for element in reversed(self.subgroups):
                    IV_table = element.IV_table
                    if len(pc_IVs)>0:
                        prev_subcell_V = interp_(Is,prev_IV[1,:],prev_IV[0,:])
                        added_I = np.zeros_like(prev_subcell_V)
                        for pc_IV in pc_IVs:
                            added_I -= interp_(prev_subcell_V, pc_IV[0,:], pc_IV[1,:])
                        V = interp_(Is-added_I,IV_table[1,:],IV_table[0,:])
                        extra_Is.extend(Is+added_I)
                    else:
                        V = interp_(Is,IV_table[1,:],IV_table[0,:])
                    Vs += V
                    if hasattr(self,"shape") and isinstance(element,CircuitGroup):
                        Vints += V
                    pc_IVs = []
                    prev_IV = []
                    if hasattr(element,"photon_coupling_diodes"):
                        prev_IV = element.IV_table
                        for pc in element.photon_coupling_diodes:
                            pc_IVs.append(pc.IV_table.copy())
                            pc_IVs[-1][1,:] *= element.area
                    if element.IV_table.shape[0]==3 and np.max(element.IV_table[2,:])>0:
                        Vint = interp_(Is,IV_table[2,:],IV_table[0,:])
                        Vints += Vint
                if len(extra_Is)==0:
                    break
        else:
            # add current
            for element in self.subgroups:
                if isinstance(element,CurrentSource):
                    element.set_IL(element.IL)
                    total_IL -= element.IL
            if self.dark_IV_table is not None:
                shift_IV_only = True
                self.IV_table = self.dark_IV_table.copy()
                if hasattr(self,"shape") and self.area is not None:
                    total_IL *= self.area
                self.IV_table[1,:] += total_IL
            else:
                Vs = []
                left_limit = None
                right_limit = None
                for element in self.subgroups:
                    if not isinstance(element,CurrentSource):
                        Vs.extend(list(element.IV_table[0,:]))
                        if isinstance(element,ForwardDiode):
                            if right_limit is None:
                                right_limit = element.IV_table[0,-1]
                            else:
                                right_limit = min(element.IV_table[0,-1],right_limit)
                        elif isinstance(element,ReverseDiode):
                            if left_limit is None:
                                left_limit = element.IV_table[0,0]
                            else:
                                left_limit = min(element.IV_table[0,0],left_limit)
                Vs = np.sort(np.array(Vs))
                if left_limit is not None:
                    find_ = np.where(Vs >= left_limit)[0]
                    Vs = Vs[find_]
                if right_limit is not None:
                    find_ = np.where(Vs <= right_limit)[0]
                    Vs = Vs[find_]
                if max_num_points is None:
                    Vs = np.unique(Vs)
                else:
                    tol = (Vs[-1]-Vs[0])/(max_num_points*1000)
                    quantized = np.round(Vs / tol)
                    _, idx = np.unique(quantized, return_index=True)
                    Vs = Vs[idx]
                Is = np.zeros_like(Vs)
                for element in self.subgroups:
                    if not isinstance(element,CurrentSource):
                        Is += interp_(Vs,element.IV_table[0,:],element.IV_table[1,:])
                Is += total_IL
                if cap_current is not None:
                    find_ = np.where(np.abs(Is) < cap_current)[0]
                    Vs = Vs[find_]
                    Is = Is[find_]
                # plt.plot(Vs,Is)
                # plt.show()

        if shift_IV_only==False:
            self.IV_table = np.array([Vs,Is])
            if max_num_points is None:
                pass
            else:
                V_range = np.max(Vs)-np.min(Vs)
                I_range = np.max(Is)-np.min(Is)
                V_segments = Vs[1:]-Vs[:-1]
                I_segments = Is[1:]-Is[:-1]
                segment_lengths = np.sqrt((V_segments/V_range)**2+(I_segments/I_range)**2)
                total_length = np.sum(segment_lengths)
                ideal_segment_length = total_length / max_num_points
                short_segments = np.where(segment_lengths < ideal_segment_length)[0]
                long_segments = np.where(segment_lengths >= ideal_segment_length)[0]
                if len(short_segments)>0:
                    short_segment_lengths = segment_lengths[short_segments]
                    short_segment_lengths_cum = np.cumsum(short_segment_lengths)
                    short_segment_left_V = Vs[short_segments]
                    long_segment_left_V = Vs[long_segments]
                    ideal_Vs = np.linspace(0,short_segment_lengths_cum[-1],max_num_points-len(long_segments))
                    short_segment_lengths_cum -= short_segment_lengths_cum[0]
                    index = np.searchsorted(short_segment_lengths_cum, ideal_Vs, side='right')-1
                    new_Vs = short_segment_left_V[index] + ideal_Vs - short_segment_lengths_cum[index]
                    new_Vs = list(new_Vs)
                    new_Vs.extend(long_segment_left_V)
                    new_Vs = np.sort(np.array(new_Vs))
                    if new_Vs[0] > Vs[0]:
                        new_Vs = np.hstack((Vs[0],new_Vs))
                    if new_Vs[-1] < Vs[-1]:
                        new_Vs = np.hstack((new_Vs,Vs[-1]))
                    new_Is = interp_(new_Vs,Vs,Is)
                    if Vints is not None:
                        new_Vints = interp_(new_Is,Is,Vints)
                        self.IV_table = np.array([new_Vs, new_Is, new_Vints])
                    else:
                        self.IV_table = np.array([new_Vs, new_Is])
                else:
                    if Vints is not None:
                        self.IV_table = np.array([Vs,Is,Vints])
                    else:
                        self.IV_table = np.array([Vs,Is])
            if all_circuit_element_children:
                self.dark_IV_table = self.IV_table.copy()
                self.dark_IV_table[1,:] -= total_IL
            # solar cell needs to scale IV table by area
            if hasattr(self,"shape") and self.area is not None:
                self.IV_table[1,:] *= self.area
                if self.dark_IV_table is not None:
                    self.dark_IV_table[1,:] *= self.area
    
    def __str__(self):
        word = self.connection + " connection:\n"
        for i, element in enumerate(self.subgroups):
            if isinstance(element,CircuitGroup):
                word += "Subgroup " + str(i) + ":\n"
            word += str(element) + "\n"
        return word    
    
    def draw(self, ax=None, x=0, y=0, display_value=False, title="Model", linewidth=1.5):
        global pbar
        draw_immediately = False
        if ax is None:
            num_of_elements = len(self.findElementType(CircuitElement))
            pbar = tqdm(total=num_of_elements)
            fig, ax = plt.subplots()
            draw_immediately = True
        
        current_x = x - self.circuit_diagram_extent[0]/2
        current_y = y - self.circuit_diagram_extent[1]/2
        if self.connection != "series":
            current_y += 0.1
        for i, element in enumerate(self.subgroups):
            if isinstance(element,CircuitElement):
                pbar.update(1)
            center_x = current_x+element.circuit_diagram_extent[0]/2
            center_y = current_y+element.circuit_diagram_extent[1]/2
            if self.connection == "series":
                center_x = x
            else:
                center_y = y
            element.draw(ax=ax, x=center_x, y=center_y, display_value=display_value)
            if self.connection=="series":
                if i > 0:
                    line = plt.Line2D([x,x],[current_y-y_spacing, current_y], color="black", linewidth=linewidth)
                    ax.add_line(line)
                current_y += element.circuit_diagram_extent[1]+y_spacing
            else:
                line = plt.Line2D([center_x,center_x], [center_y+element.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                ax.add_line(line)
                line = plt.Line2D([center_x,center_x], [center_y-element.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                ax.add_line(line)
                if i > 0:
                    line = plt.Line2D([center_x,current_x-x_spacing-self.subgroups[i-1].circuit_diagram_extent[0]/2], [y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                    ax.add_line(line)
                    line = plt.Line2D([center_x,current_x-x_spacing-self.subgroups[i-1].circuit_diagram_extent[0]/2], [y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                    ax.add_line(line)
                current_x += element.circuit_diagram_extent[0]+x_spacing
        if draw_immediately:
            pbar.close()
            line = plt.Line2D([x,x], [y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2-0.2], color="black", linewidth=linewidth)
            ax.add_line(line)
            line = plt.Line2D([x,x], [y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2+0.2], color="black", linewidth=linewidth)
            ax.add_line(line)
            draw_symbol(draw_earth_symbol, ax=ax,  x=x, y=y-self.circuit_diagram_extent[1]/2-0.3)
            draw_symbol(draw_pos_terminal_symbol, ax=ax,  x=x, y=y+self.circuit_diagram_extent[1]/2+0.25)
            ax.set_aspect('equal')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            fig.canvas.manager.set_window_title(title)
            plt.show()

def get_extent(elements, center=True):
    x_bounds = [None,None]
    y_bounds = [None,None]
    for element in elements:
        if hasattr(element,"extent") and element.extent is not None:
            xs = [element.location[0]-element.extent[0]/2,element.location[0]+element.extent[0]/2]
            ys = [element.location[1]-element.extent[1]/2,element.location[1]+element.extent[1]/2]
            if x_bounds[0] is None:
                x_bounds[0] = xs[0]
            else:
                x_bounds[0] = min(x_bounds[0],xs[0])
            if x_bounds[1] is None:
                x_bounds[1] = xs[1]
            else:
                x_bounds[1] = max(x_bounds[1],xs[1])
            if y_bounds[0] is None:
                y_bounds[0] = ys[0]
            else:
                y_bounds[0] = min(y_bounds[0],ys[0])
            if y_bounds[1] is None:
                y_bounds[1] = ys[1]
            else:
                y_bounds[1] = max(y_bounds[1],ys[1])
    if (x_bounds[0] is not None) and (x_bounds[1] is not None) and (y_bounds[0] is not None) and (y_bounds[1] is not None):
        if center:
            center = [0.5*(x_bounds[0]+x_bounds[1]),0.5*(y_bounds[0]+y_bounds[1])]
            for element in elements:
                if hasattr(element,"extent") and element.extent is not None:
                    element.location[0] -= center[0]
                    element.location[1] -= center[1]
        return [x_bounds[1]-x_bounds[0],y_bounds[1]-y_bounds[0]]
    else:
        return None

def get_circuit_diagram_extent(elements,connection):
    total_extent = [0.0,0.0]
    for i, element in enumerate(elements):
        extent_ = element.circuit_diagram_extent
        if connection=="series":
            total_extent[0] = max(total_extent[0], extent_[0])
            total_extent[1] += extent_[1]
            if i > 0:
                total_extent[1] += y_spacing
        else:
            total_extent[1] = max(total_extent[1], extent_[1])
            total_extent[0] += extent_[0]
            if i > 0:
                total_extent[0] += x_spacing
    if connection!="series":
        total_extent[1] += 0.2 # the connectors
    return total_extent

def tile_elements(elements, rows=None, cols=None, x_gap = 0.0, y_gap = 0.0, turn=True, col_wise_ordering=True):
    assert((rows is not None) or (cols is not None))
    if rows is None:
        rows = int(np.ceil(float(len(elements))/float(cols)))
    if cols is None:
        cols = int(np.ceil(float(len(elements))/float(rows)))
    row = 0
    col = 0
    rotation = 0
    pos = np.array([0,0]).astype(float)
    max_x_extent = 0.0
    max_y_extent = 0.0
    for element in elements:
        if hasattr(element,"extent"):
            x_extent = element.extent[0]
            max_x_extent = max(max_x_extent,x_extent)
            y_extent = element.extent[1]
            max_y_extent = max(max_y_extent,y_extent)
            element.location = pos.copy()
            element.rotation = rotation
            if col_wise_ordering:
                row += 1
                if row < rows:
                    if rotation==0:
                        pos[1] += y_extent + y_gap
                    else:
                        pos[1] -= (y_extent + y_gap)
                else:
                    row = 0
                    col += 1
                    pos[0] += max_x_extent + x_gap
                    max_x_extent = 0.0
                    if turn:
                        rotation = 180 - rotation
                    else:
                        pos[1] = 0
            else:
                col += 1
                if col < cols:
                    if rotation==0:
                        pos[0] += x_extent + x_gap
                    else:
                        pos[0] -= (x_extent + x_gap)
                else:
                    col = 0
                    row += 1
                    pos[1] += max_y_extent + y_gap
                    max_y_extent = 0.0
                    if turn:
                        rotation = 180 - rotation
                    else:
                        pos[0] = 0
            

def circuit_deepcopy(circuit_group):
    circuit_group2 = copy.deepcopy(circuit_group)
    circuit_group2.reassign_parents()
    return circuit_group2

def find_subgroups_by_name(circuit_group, target_name):
    result = []
    for element in circuit_group.subgroups:
        if hasattr(element, 'name') and element.name == target_name:
            result.append(element)
        if isinstance(element, CircuitGroup):
            result.extend(find_subgroups_by_name(element, target_name))
    return result

def find_subgroups_by_tag(circuit_group, tag):
    result = []
    for element in circuit_group.subgroups:
        if hasattr(element, 'tag') and element.tag == tag:
            result.append(element)
        if isinstance(element, CircuitGroup):
            result.extend(find_subgroups_by_name(element, tag))
    return result