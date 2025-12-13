import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from PV_Circuit_Model.utilities import *
from PV_Circuit_Model.utilities_silicon import *
from tqdm import tqdm
from PV_Circuit_Model.IV_jobs import *
import gc, inspect

solver_env_variables = ParameterSet.get_set("solver_env_variables")
REMESH_POINTS_DENSITY = solver_env_variables["REMESH_POINTS_DENSITY"]
REMESH_NUM_ELEMENTS_THRESHOLD = solver_env_variables["REMESH_NUM_ELEMENTS_THRESHOLD"]
      
class CircuitComponent(ParamSerializable):
    _critical_fields = ("max_I","max_num_points")
    _artifacts = ("IV_V", "IV_I", "IV_V_lower", "IV_I_lower", "IV_V_upper", "IV_I_upper","extrapolation_allowed", "extrapolation_dI_dV",
                  "has_I_domain_limit","job_heap", "refined_IV","operating_point","bottom_up_operating_point")
    _dont_serialize = ("circuit_depth", "num_circuit_elements")
    max_I = None
    max_num_points = None
    IV_V = None  
    IV_I = None  
    extrapolation_allowed = [False,False]
    extrapolation_dI_dV = [0,0]
    has_I_domain_limit = [False,False]
    refined_IV = False
    operating_point = None
    num_circuit_elements = 1
    circuit_depth = 1

    def __init__(self,tag=None):
        self.circuit_diagram_extent = [0, 0.8]
        self.parent = None
        self.aux = {}
        self.tag = tag
        self.extrapolation_allowed = [False,False]
        self.extrapolation_dI_dV = [0,0]
        self.has_I_domain_limit = [False,False]

    def __len__(self):
        return self.num_circuit_elements

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
        self.clear_artifacts()
        if self.parent is not None:
            self.parent.null_IV()

    def null_all_IV(self):
        self.clear_artifacts()
        if self._type_number >= 5: # is CircuitGroup
            for element in self.subgroups:
                element.null_all_IV()

    def build_IV(self):
        gc.disable()
        self.job_heap = IV_Job_Heap(self)
        self.job_heap.run_IV()
        gc.enable()

    def refine_IV(self):
        if hasattr(self,"job_heap") and getattr(self,"operating_point",None) is not None:
            gc.disable()
            self.job_heap.refine_IV()
            self.refined_IV = True
            gc.enable()

    def calc_uncertainty(self):
        if hasattr(self,"job_heap"):
            if hasattr(self.job_heap,"calc_uncertainty"): # python version does not have this
                self.job_heap.calc_uncertainty()

    def __call__(self, *, atomic=True):
        clone = self.clone()
        clone._is_atomic = atomic
        return clone

    def __add__(self, other):
        if not isinstance(other, CircuitComponent):
            return NotImplemented
        return series(self, other, flatten_connection_=True)
    
    def __or__(self, other):
        if not isinstance(other, CircuitComponent):
            return NotImplemented
        return parallel(self, other, flatten_connection_=True)
    
    def __mul__(self, other):
        # component * scalar
        if isinstance(other, (int, float)):
            return series(*[circuit_deepcopy(self)() for _ in range(int(other))])
        return NotImplemented

    def __rmul__(self, other):
        # scalar * component
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        # component ** scalar
        if isinstance(other, (int, float)):
            return parallel(*[circuit_deepcopy(self)() for _ in range(int(other))])
        return NotImplemented

    def __rpow__(self, other):
        # scalar * component
        return self.__pow__(other)

    def __ipow__(self, other):
        return self.__pow__(other)
    
    def structure(self):
        children = getattr(self, "subgroups", None)
        return (
            type(self),
            getattr(self, "connection", None),
            tuple(c.structure() for c in children) if children else (),
        )
    
    def copy(self,other): # weak copy, only critical fields
        for field in self._critical_fields:
            if hasattr(self,field) and hasattr(other,field):
                setattr(self,field,getattr(other,field))

def flatten_connection(parts_list,connection):
    flat_list = []
    for part in parts_list:
        # _type_number > 5 means already a special grouping like cell, module, so cannot break apart
        if isinstance(part,CircuitGroup) and part.connection==connection and not getattr(part,"_is_atomic",False) and part._type_number==5:
            flat_list.extend(part.subgroups)
        else:
            flat_list.append(part)
    return flat_list

def connect(*args,connection="series",flatten_connection_=False,**kwargs):
    flat_list = []
    for arg in args:
        if isinstance(arg,(list, tuple)):
            flat_list.extend(arg)
        else:
            flat_list.append(arg)
    if flatten_connection_:
        flat_list = flatten_connection(flat_list,connection=connection)
    all_items_have_extent = True
    for item in flat_list:
        if hasattr(item,"_is_atomic"):
            del item._is_atomic
        if not hasattr(item,"extent"):
            all_items_have_extent = False
    if all_items_have_extent:
        safe_kwargs = filter_kwargs(tile_elements, kwargs)
        tile_elements(flat_list, **safe_kwargs)
    safe_kwargs = filter_kwargs(CircuitGroup.__init__, kwargs)
    return CircuitGroup(subgroups=flat_list,connection=connection,**safe_kwargs)

def series(*args,flatten_connection_=False,**kwargs):
    kwargs.pop("connection", None)
    return connect(*args,connection="series",flatten_connection_=flatten_connection_,**kwargs)

def parallel(*args,flatten_connection_=False,**kwargs):
    kwargs.pop("connection", None)
    return connect(*args,connection="parallel",flatten_connection_=flatten_connection_,**kwargs)



class CircuitElement(CircuitComponent):
    def set_operating_point(self,V=None,I=None):
        pass
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
    def set_operating_point(self,V=None,I=None):
        if I is not None:
            raise NotImplementedError
        self.operating_point = [V,-self.IL]
    def set_IL(self,IL):
        self.IL = IL
        self.null_IV()
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
    def __init__(self, cond=1.0, tag=None):
        super().__init__(tag=tag)
        self.cond = cond
    def set_cond(self,cond):
        self.cond = cond
        self.null_IV()
    def set_operating_point(self,V=None,I=None):
        if I is not None:
            self.operating_point = [I/self.cond,I]
        if V is not None:
            self.operating_point = [V,V*self.cond]
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
    max_I = 0.2
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
    def set_operating_point(self,V=None,I=None):
        pass
    def changeTemperature(self,temperature):
        self.VT = get_VT(temperature,VT_at_25C)
        old_ni  = get_ni(self.refT)
        new_ni  = get_ni(temperature)
        scale_factor = (new_ni/old_ni)**(2/self.n)
        self.set_I0(self.refI0*scale_factor)
    
class ForwardDiode(Diode):
    def __init__(self,I0=1e-15,n=1,tag=None): #V_shift is to shift the starting voltage, e.g. to define breakdown
        super().__init__(I0, n, V_shift=0,tag=tag)
    def __str__(self):
        return "Forward Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n)
    def get_value_text(self):
        word = f"I0 = {self.I0:.3e}"
        if "error" in self.aux and not np.isnan(self.aux["error"]):
            word += f"\n\u00B1{self.aux['error']:.3e}"
        word += f" A\nn = {self.n:.2f}"
        return word
    def set_operating_point(self,V=None,I=None):
        if V is not None:
            self.operating_point = [V, self.I0*(np.exp((V-self.V_shift)/(self.n*self.VT))-1)]
        else:
            self.operating_point = [interp_(I,self.IV_I,self.IV_V),I]
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
    def __str__(self):
        return "Reverse Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n) + ", breakdown V = " + str(self.V_shift)
    def get_value_text(self):
        return f"I0 = {self.I0:.3e}A\nn = {self.n:.2f}\nbreakdown V = {self.V_shift:.2f}"
    def set_operating_point(self,V=None,I=None):
        if V is not None:
            self.operating_point = [V, -self.I0*np.exp((-V-self.V_shift)/(self.n*self.VT))]
        else:
            self.operating_point = [interp_(I,self.IV_I,self.IV_V),I]
    def get_draw_func(self):
        return draw_reverse_diode_symbol
    
class Intrinsic_Si_diode(ForwardDiode):
    _type_number = 4
    bandgap_narrowing_RT = np.array(bandgap_narrowing_RT)
    # area is 1 is OK because the cell subgroup has normalized area of 1
    def __init__(self,base_thickness=180e-4,base_type="n",base_doping=1e+15,area=1.0,temperature=25,tag=None):
        CircuitElement.__init__(self, tag)
        self.base_thickness = base_thickness
        self.base_type = base_type
        self.base_doping = base_doping
        self.temperature = temperature
        self.I0 = 0.0
        self.n = 0.0
        self.V_shift = 0.0
        self.area = area
        self.VT = get_VT(self.temperature,VT_at_25C)
        self.ni = get_ni(self.temperature)
    def __str__(self):
        return "Si Intrinsic Diode"
    
    def calc_intrinsic_Si_I(self, V):
        ni = self.ni
        VT = self.VT
        N_doping = self.base_doping
        pn = ni**2*np.exp(V/VT)
        delta_n = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2*np.exp(V/VT)))
        if self.base_type == "p":
            n0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
            p0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        else:
            p0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
            n0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        BGN = interp_(delta_n,self.bandgap_narrowing_RT[:,0],self.bandgap_narrowing_RT[:,1])
        ni_eff = ni*np.exp(BGN/2/VT)
        geeh = 1 + 13*(1-np.tanh((n0/3.3e17)**0.66))
        gehh = 1 + 7.5*(1-np.tanh((p0/7e17)**0.63))
        Brel = 1
        Blow = 4.73e-15
        intrinsic_recomb = (pn - ni_eff**2)*(2.5e-31*geeh*n0+8.5e-32*gehh*p0+3e-29*delta_n**0.92+Brel*Blow) # in units of 1/s/cm3
        return q*intrinsic_recomb*self.base_thickness*self.area

    def set_operating_point(self,V=None,I=None):
        if V is not None:
            self.operating_point = [V, self.calc_intrinsic_Si_I(V)]
        else:
            self.operating_point = [interp_(I,self.IV_I,self.IV_V),I]

    def get_value_text(self):
        word = f"intrinsic:\nt={self.base_thickness:.2e}\n{self.base_type} type\n{self.base_doping:.2e} cm-3"
        return word
    def set_I0(self,I0):
        pass # does nothing
    def changeTemperature(self,temperature):
        self.temperature = temperature
        self.VT = get_VT(self.temperature,VT_at_25C)
        self.ni = get_ni(self.temperature)
        self.null_IV()

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
        if self.num_circuit_elements > REMESH_NUM_ELEMENTS_THRESHOLD:
            self.max_num_points = int(REMESH_POINTS_DENSITY*np.sqrt(self.num_circuit_elements))
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

    def set_operating_point(self,V=None,I=None,refine_IV=False,shallow=False):
        if not hasattr(self,"job_heap"):
            self.build_IV()
        elif self.IV_V is None:
            self.job_heap.run_IV()

        # autorange
        if V is not None:
            if (not self.extrapolation_allowed[1] and V > self.IV_V[-1]) or (not self.extrapolation_allowed[0] and V < self.IV_V[0]): # out of reach of IV curve
                diodes = self.findElementType(Diode)
                while (not self.extrapolation_allowed[1] and V > self.IV_V[-1]) or (not self.extrapolation_allowed[0] and V < self.IV_V[0]): 
                    for diode in diodes:
                        diode.max_I *= 10
                    self.null_all_IV()
                    self.build_IV()
        elif I is not None:
            if (not self.extrapolation_allowed[1] and I > self.IV_I[-1]) or (not self.extrapolation_allowed[0] and I < self.IV_I[0]): # out of reach of IV curve
                diodes = self.findElementType(Diode)
                while (not self.extrapolation_allowed[1] and I > self.IV_I[-1]) or (not self.extrapolation_allowed[0] and I < self.IV_I[0]):
                    for diode in diodes:
                        diode.max_I *= 10
                    self.null_all_IV()
                    self.build_IV()

        if shallow or self.num_circuit_elements < 10000: # no need to go c++ overkill
            V_ = V
            I_ = I
            if V is not None:
                I_ = np.interp(V,self.IV_V,self.IV_I)
            if I is not None:
                V_ = np.interp(I,self.IV_I,self.IV_V)
            self.operating_point = [V_,I_]
            if self._type_number==6: # cell, has area
                I_ /= self.area
            if shallow:   
                return
            for item in self.subgroups:
                if self.connection=="series":
                    item.set_operating_point(I=I_)
                else:
                    item.set_operating_point(V=V_)
        else:
            self.job_heap.set_operating_point(V,I)

        gc.disable()
        if refine_IV:
            self.job_heap.refine_IV()
        self.refined_IV = True
        gc.enable()

    def removeElementOfTag(self,tag):
        for j in range(len(self.subgroups) - 1, -1, -1):
            element = self.subgroups[j]
            if isinstance(element, CircuitElement):
                if element.tag == tag:
                    self.subgroups.pop(j)
            elif isinstance(element, CircuitGroup):
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
    
    def __getitem__(self,type_):
        return self.findElementType(type_)
    
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
    
    def as_type(self, cls, **kwargs):
        if not issubclass(cls, CircuitComponent):
            raise TypeError(...)
        if hasattr(cls, "from_circuitgroup"):
            return cls.from_circuitgroup(self, **kwargs)
        raise TypeError(f"{cls.__name__} does not support conversion")

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
    if rows is None and cols is None:
        rows = 1
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
            

def circuit_deepcopy(circuit_component):
    return circuit_component.clone()

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



