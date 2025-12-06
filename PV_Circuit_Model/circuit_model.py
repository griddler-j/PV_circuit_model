import numpy as np
from matplotlib import pyplot as plt
from PV_Circuit_Model.utilities import *
from PV_Circuit_Model.utilities_silicon import *
from tqdm import tqdm
from PV_Circuit_Model.IV_jobs import *
import gc, numbers
      
class CircuitComponent(ParamSerializable):
    _critical_fields = ("max_I","max_num_points")
    max_I = None
    max_num_points = None
    def __init__(self,tag=None):
        self.IV_V = None  # 1D float64
        self.IV_I = None  # 1D float64
        self.operating_point = None #V,I
        self.circuit_diagram_extent = [0, 0.8]
        self.parent = None
        self.aux = {}
        self.tag = tag
        self.num_circuit_elements = 1
        self.circuit_depth = 1

    @property
    def IV_table(self):
        if self.IV_V is None or self.IV_I is None:
            return None
        # This allocates a fresh 2xN array for user-land / plotting.
        return np.stack([self.IV_V, self.IV_I], axis=0)

    @IV_table.setter
    def IV_table(self, value):
        # Allow clearing with None
        if value is None:
            self.IV_V = None
            self.IV_I = None
            return

        value = np.ascontiguousarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != 2:
            raise ValueError("IV_table must be shape (2, N)")

        # Copy rows into 1D contiguous arrays
        self.IV_V = value[0, :].copy()
        self.IV_I = value[1, :].copy()

    def null_IV(self):
        self.refined_IV = False
        self.IV_V = None
        self.IV_I = None
        if self.parent is not None:
            self.parent.null_IV()
    def build_IV(self):
        gc.disable()
        self.job_heap = IV_Job_Heap(self)
        self.job_heap.run_IV()
        gc.enable()
    def refine_IV(self):
        if not hasattr(self,"job_heap"):
            self.job_heap = IV_Job_Heap(self)
        self.job_heap.refine_IV()

class CircuitElement(CircuitComponent):
    def set_operating_point(self,V=None,I=None):
        if V is not None:
            I = interp_(V,self.IV_V,self.IV_I)
        elif I is not None:
            V = interp_(I,self.IV_I,self.IV_V)
        self.operating_point = [V,I]
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
    _critical_fields = CircuitComponent._critical_fields + ("IL",)
    _type_number = 0
    def __init__(self, IL, Suns=1.0, temperature=25, temp_coeff=0.0, tag=None):
        super().__init__(tag=tag)
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
        self.null_IV()

    def copy(self,source):
        self.refSuns = source.refSuns
        self.Suns = source.Suns
        self.refIL = source.refIL
        self.refT = source.refT
        self.T = source.T
        self.temp_coeff = source.temp_coeff
        self.set_IL(source.IL)

    def changeTemperatureAndSuns(self,temperature=None,Suns=None):
        if Suns is not None:
            self.Suns = Suns
        if temperature is not None:
            self.T = temperature
        self.set_IL(self.Suns*(self.refIL / self.refSuns + self.temp_coeff * (self.T - self.refT)))

    def __str__(self):
        return "Current Source: IL = " + self.get_value_text()
    
    def get_value_text(self):
        return f"{self.IL:.4f} A"
    def get_draw_func(self):
        return draw_CC_symbol

class Resistor(CircuitElement):
    _critical_fields = CircuitComponent._critical_fields + ("cond",)
    _type_number = 1
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
    _critical_fields = CircuitComponent._critical_fields + ("I0","n","V_shift","VT")
    _type_number = 2
    def __init__(self,I0=1e-15,n=1,V_shift=0,tag=None,temperature=25): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(tag=tag)
        self.I0 = I0
        self.n = n
        self.V_shift = V_shift
        self.VT = get_VT(temperature,VT_at_25C)
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

    def changeTemperature(self,temperature):
        self.VT = get_VT(temperature,VT_at_25C)
        old_ni  = get_ni(self.refT)
        new_ni  = get_ni(temperature)
        scale_factor = (new_ni/old_ni)**(2/self.n)
        self.set_I0(self.refI0*scale_factor)

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
    
class ForwardDiode(Diode):
    def __init__(self,I0=1e-15,n=1,tag=None): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(I0, n, V_shift=0,tag=tag)
        self.max_I = 0.2
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
    _type_number = 3
    def __init__(self,I0=1e-15,n=1, V_shift=0,tag=None): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(I0, n, V_shift, tag=tag)
    def calc_I(self,V):
        return -self.I0*np.exp((-V-self.V_shift)/(self.n*self.VT))
    def calc_dI_dV(self,V):
        I = self.calc_I(V)
        return -I/(self.n*self.VT)
    def __str__(self):
        return "Reverse Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n) + ", breakdown V = " + str(self.V_shift)
    def get_value_text(self):
        return f"I0 = {self.I0:.3e}A\nn = {self.n:.2f}\nbreakdown V = {self.V_shift:.2f}"
    def get_draw_func(self):
        return draw_reverse_diode_symbol
    
class Intrinsic_Si_diode(ForwardDiode):
    _type_number = 4
    bandgap_narrowing_RT = bandgap_narrowing_RT
    # area is 1 is OK because the cell subgroup has normalized area of 1
    def __init__(self,base_thickness=180e-4,base_type="n",base_doping=1e+15,area=1.0,temperature=25,tag=None):
        CircuitElement.__init__(self, tag)
        self.base_thickness = base_thickness
        self.base_type = base_type
        self.base_doping = base_doping
        self.max_I = 0.2
        self.temperature = temperature
        self.I0 = 0.0
        self.n = 0.0
        self.V_shift = 0.0
        self.area = area
        self.VT = get_VT(self.temperature,VT_at_25C)
        self.ni = get_ni(self.temperature)
    def __str__(self):
        return "Si Intrinsic Diode"
    def get_value_text(self):
        word = f"intrinsic:\nt={self.base_thickness:.2e}\n{self.base_type} type\n{self.base_doping:.2e} cm-3"
        return word
    def set_I0(self,I0):
        pass # does nothing
    def copy(self,source):
        self.base_thickness = source.base_thickness
        self.base_type = source.base_type
        self.base_doping = source.base_doping
        self.temperature = source.temperature
    def changeTemperature(self,temperature):
        self.temperature = temperature
        self.VT = get_VT(self.temperature,VT_at_25C)
        self.ni = get_ni(self.temperature)
        self.null_IV()
    def calc_I(self,V,get_dI_dV=False):
        ni = get_ni(self.temperature)
        VT = get_VT(self.temperature,VT_at_25C)
        N_doping = self.base_doping
        pn = ni**2*np.exp(V/VT)
        delta_n = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2*np.exp(V/VT)))
        if get_dI_dV:
            d_pn_dV = pn/VT
            d_delta_n_dV = ni**2*np.exp(V/VT)/VT/np.sqrt(N_doping**2 + 4*ni**2*np.exp(V/VT))
        if self.base_type == "p":
            n0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
            p0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        else:
            p0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
            n0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        BGN = interp_(delta_n,self.bandgap_narrowing_RT[:,0],self.bandgap_narrowing_RT[:,1])
        ni_eff = ni*np.exp(BGN/2/VT)

        q = 1.602e-19
        geeh = 1 + 13*(1-np.tanh((n0/3.3e17)**0.66))
        gehh = 1 + 7.5*(1-np.tanh((p0/7e17)**0.63))
        Brel = 1
        Blow = 4.73e-15
        intrinsic_recomb = (pn - ni_eff**2)*(2.5e-31*geeh*n0+8.5e-32*gehh*p0+3e-29*delta_n**0.92+Brel*Blow) # in units of 1/s/cm3
        if get_dI_dV:
            d_intrinsic_recomb_dV = d_pn_dV*(2.5e-31*geeh*n0+8.5e-32*gehh*p0+3e-29*delta_n**0.92+Brel*Blow) + pn*3e-29*d_delta_n_dV**0.92
            return q*d_intrinsic_recomb_dV*self.base_thickness*self.area
        return q*intrinsic_recomb*self.base_thickness*self.area
    
    def calc_dI_dV(self,V):
        return self.calc_I(V,get_dI_dV=True)
    
    def get_V_range(self,max_num_points=100):
        if max_num_points is None:
            max_num_points = 100
        max_I = 0.2
        if hasattr(self,"max_I"):
            max_I = self.max_I
            max_num_points *= max_I/0.2
        # assume that 0.2 A/cm2 is max you'll need
        if self.base_thickness==0:
            Voc = 10
        else:
            VT = get_VT(self.temperature,VT_at_25C)
            Voc = 0.7
            for _ in range(10):
                I = self.calc_I(Voc)
                if I >= max_I and I <= max_I*1.1:
                    break
                Voc += VT*np.log(max_I/I)
        V = [self.V_shift-1.1,self.V_shift-1.0,self.V_shift,self.V_shift+0.02,self.V_shift+.08]+list(self.V_shift + Voc*np.log(np.arange(1,max_num_points))/np.log(max_num_points-1))
        V = np.array(V)
        return V

class CircuitGroup(CircuitComponent):
    _critical_fields = CircuitComponent._critical_fields + ("connection","subgroups")
    _type_number = 5
    def __init__(self,subgroups,connection="series",name=None,location=None,
                 rotation=0,x_mirror=1,y_mirror=1,extent=None):
        super().__init__()
        self.connection = connection
        self.subgroups = subgroups
        self.num_circuit_elements = 0
        for element in self.subgroups:
            element.parent = self
            self.num_circuit_elements += element.num_circuit_elements
            self.circuit_depth = max(self.circuit_depth,element.circuit_depth+1)
        if self.num_circuit_elements > 50:
            self.max_num_points = int(500*np.sqrt(self.num_circuit_elements))
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
        self.is_circuit_group = True

    def add_element(self,element):
        self.subgroups.append(element)
        element.parent = self

    def null_all_IV(self):
        self.IV_V = None
        self.IV_I = None
        if hasattr(self,"refined_IV") and self.refined_IV:
            self.refined_IV = False
        for element in self.subgroups:
            if isinstance(element,CircuitElement):
                element.IV_V = None
                element.IV_I = None
            else:
                element.null_all_IV()

    def reassign_parents(self):
        for element in self.subgroups:
            element.parent = self
            if isinstance(element,CircuitGroup):
                element.reassign_parents()

    def set_operating_point(self,V=None,I=None,refine_IV=False):
        if self.IV_V is None or not hasattr(self,"job_heap"):
            self.job_heap.run_IV()
        gc.disable()
        self.job_heap.set_operating_point(V,I)
        if refine_IV:
            self.job_heap.refine_IV()
        self.refined_IV = True
        gc.enable()

    def removeElementOfTag(self,tag):
        for element in self.subgroups[:]:
            if isinstance(element,CircuitElement):
                if element.tag==tag:
                    self.subgroups.remove(element)
            elif isinstance(element,CircuitGroup):
                element.removeElementOfTag(tag)
        self.null_IV()

    def set_temperature(self,temperature):
        diodes = self.findElementType(Diode)
        for diode in diodes:
            diode.changeTemperature(temperature)
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(temperature=temperature)

    def findElementType(self,type_,serialize=False):
        list_ = []
        for i, element in enumerate(self.subgroups):
            if (not isinstance(type_,str) and isinstance(element,type_))  or (isinstance(type_,str) and type(element).__name__==type_):
                list_.append(element)
            elif isinstance(element,CircuitGroup):
                list_.extend(element.findElementType(type_,serialize=serialize))
        if serialize:
            for i, element in enumerate(list_):
                element.name = str(i)
        return list_
    
    def __str__(self):
        if self.num_circuit_elements > 2000:
            print(f"There are too many elements to draw ({self.num_circuit_elements}).  I give up!")
            return
        word = self.connection + " connection:\n"
        for i, element in enumerate(self.subgroups):
            if isinstance(element,CircuitGroup):
                word += "Subgroup " + str(i) + ":\n"
            word += str(element) + "\n"
        return word    
    
    def draw(self, ax=None, x=0, y=0, display_value=False, title="Model", linewidth=1.5):
        if self.num_circuit_elements > 2000:
            print(f"There are too many elements to draw ({self.num_circuit_elements}).  I give up!")
            return
        
        global pbar
        draw_immediately = False
        if ax is None:
            num_of_elements = self.num_circuit_elements
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
    return circuit_group.clone()

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
