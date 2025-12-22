import numpy as np
from PV_Circuit_Model.circuit_model import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, Point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import numbers
from functools import partial

# Device is a logical wrapper around CircuitGroup that preserves internal structure.
# When Devices are connected in series or parallel via + or |, their internal subgroups are NOT flattened.
# This allows complex components (cells, modules, strings, etc.) to behave as atomic units.
# Subclasses of Device typically add device-specific parameters, behavior, and analysis methods.
class Device(CircuitGroup):
    _is_atomic = True
    @classmethod
    def from_circuitgroup(cls, comp, **kwargs):
        return cls(comp.subgroups, comp.connection, **kwargs)

def set_Suns(circuit_group, suns):
    currentSources = circuit_group.findElementType(CurrentSource)
    for currentSource in currentSources:
        currentSource.changeTemperatureAndSuns(Suns=suns)
CircuitGroup.set_Suns = set_Suns

class Cell(Device,_type_number=6):
    photon_coupling_diodes = None
    _critical_fields = CircuitGroup._critical_fields + ("area",)
    def __init__(self,subgroups,connection="series",area=1,location=None,
                 rotation=0,shape=None,name=None,temperature=25,Suns=1.0):
        x_extent = 0.0
        y_extent = 0.0
        if shape is not None:
            x_extent = np.max(shape[:,0])-np.min(shape[:,0])
            y_extent = np.max(shape[:,1])-np.min(shape[:,1])
        super().__init__(subgroups, connection,location=location,rotation=rotation,
                         name=name,extent=np.array([x_extent,y_extent]).astype(float))
        self.area = area
        if self.max_I is not None:
            self.max_I *= area
        self.is_cell = True
        self.shape = shape
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        self.get_branches()
        self.photon_coupling_diodes = self.findElementType(PhotonCouplingDiode)

    def get_branches(self):
        if self.connection=="series":
            for branch in self.subgroups:
                if isinstance(branch,Resistor):
                    self.series_resistor = branch
                else:
                    self.diode_branch = branch
        else:
            self.series_resistor = None
            self.diode_branch = self

    # a weak copy, only the parameters
    def copy_values(self,cell2):
        self.temperature = cell2.temperature
        self.Suns = cell2.Suns
        if self.series_resistor is not None:
            self.series_resistor.copy_values(cell2.series_resistor)
        for i, element in enumerate(self.diode_branch.subgroups):
            if i < len(cell2.diode_branch.subgroups) and type(element)==type(cell2.diode_branch.subgroups[i]):
                element.copy_values(cell2.diode_branch.subgroups[i])

    def set_Suns(self,Suns):
        self.Suns = Suns
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(Suns=Suns)
    
    def set_temperature(self,temperature):
        super().set_temperature(temperature)
        self.temperature = temperature

    def JL(self):
        JL = 0.0
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            JL += currentSource.IL
        return JL

    def IL(self):
        return self.JL()*self.area     
    
    def set_JL(self,JL,Suns=1.0,temperature=25):
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            if currentSource.tag != "defect":
                currentSource.refSuns = Suns
                currentSource.refIL = JL
                currentSource.refT = temperature
                currentSource.changeTemperatureAndSuns(
                    temperature=self.temperature,Suns=self.Suns)
                break

    def set_IL(self,IL,Suns=1.0,temperature=25):
        self.set_JL(IL/self.area,Suns=Suns,temperature=temperature)
    
    def J0(self,n):
        J0 = 0.0
        diodes = self.findElementType(ForwardDiode)
        for diode in diodes:
            if not isinstance(diode,Intrinsic_Si_diode) and not isinstance(diode,PhotonCouplingDiode):
                if diode.n==n:
                    J0 += diode.I0
        return J0
    def J01(self):
        return self.J0(n=1)
    def J02(self):
        return self.J0(n=2)     
    
    def PC_J0(self,n):
        J0 = 0.0
        diodes = self.findElementType(PhotonCouplingDiode)
        for diode in diodes:
            if diode.n==n:
                J0 += diode.I0
        return J0
    def PC_J01(self):
        return self.PC_J0(n=1)
    def PC_I0(self,n):
        return self.PC_J0(n)*self.area
    def PC_I01(self):
        return self.PC_I0(n=1)
    
    def I0(self,n):
        return self.J0(n)*self.area
    def I01(self):
        return self.I0(n=1)
    def I02(self):
        return self.I0(n=2)    
    
    def set_J0(self,J0,n,temperature=25):
        diodes = self.findElementType(ForwardDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,Intrinsic_Si_diode) and diode.n==n and not isinstance(diode,PhotonCouplingDiode):
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature)
                break
    def set_J01(self,J0,temperature=25):
        self.set_J0(J0,n=1,temperature=temperature)
    def set_J02(self,J0,temperature=25):
        self.set_J0(J0,n=2,temperature=temperature)

    def set_I0(self,I0,n,temperature=25):
        self.set_J0(I0/self.area, n=n, temperature=temperature)
    def set_I01(self,I0,temperature=25):
        self.set_I0(I0,n=1,temperature=temperature)
    def set_I02(self,I0,temperature=25):
        self.set_I0(I0,n=2,temperature=temperature)

    def set_PC_J0(self,J0,n,temperature=25):
        diodes = self.findElementType(PhotonCouplingDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,Intrinsic_Si_diode) and diode.n==n:
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature)
                break
    def set_PC_J01(self,J0,temperature=25):
        self.set_PC_J0(J0,n=1,temperature=temperature)
    def set_PC_I0(self,I0,n,temperature=25):
        self.set_PC_J0(I0/self.area, n=n, temperature=temperature)
    def set_PC_I01(self,I0,temperature=25):
        self.set_PC_I0(I0,n=1,temperature=temperature)
    
    def specific_Rs_cond(self):
        if self.series_resistor is None:
            return np.inf
        return self.series_resistor.cond
    def Rs_cond(self):
        return self.specific_Rs_cond()/self.area
    def specific_Rs(self):
        return 1/self.specific_Rs_cond()
    def Rs(self):
        return 1/self.Rs_cond()
    
    def set_rev_breakdown_V(self,V):
        reverse_diode = self.diode_branch.findElementType(ReverseDiode)[0]
        reverse_diode.V_shift = V
        reverse_diode.null_IV()

    def set_specific_Rs_cond(self,cond):
        if self.series_resistor is not None:
            self.series_resistor.set_cond(cond)
    def set_Rs_cond(self,cond):
        self.set_specific_Rs_cond(cond/self.area)
    def set_specific_Rs(self,Rs):
        self.set_specific_Rs_cond(1/Rs)
    def set_Rs(self,Rs):
        self.set_specific_Rs(Rs*self.area)
    
    def specific_shunt_cond(self):
        Rsh_cond = 0.0
        shunt_resistors = self.diode_branch.findElementType(Resistor)
        for res in shunt_resistors:
            Rsh_cond += res.cond
        return Rsh_cond
    def shunt_cond(self):
        return self.specific_shunt_cond()/self.area
    def specific_shunt_res(self):
        return 1/self.specific_shunt_cond()
    def shunt_res(self):
        return 1/self.shunt_cond()
    
    def set_specific_shunt_cond(self,cond):
        shunt_resistors = self.diode_branch.findElementType(Resistor)
        for res in shunt_resistors:
            if res.tag != "defect":
                res.set_cond(cond)
                break
    def set_shunt_cond(self,cond):
        self.set_specific_shunt_cond(cond/self.area)
    def set_specific_shunt_res(self,Rsh):
        self.set_specific_shunt_cond(1/Rsh)
    def set_shunt_res(self,Rsh):
        self.set_specific_shunt_res(Rsh*self.area)
    def set_shape(self,wafer_format="M10",half_cut=True):
        shape_area = wafer_shape(format=wafer_format,half_cut=half_cut)
        self.shape, self.area = shape_area["shape"], shape_area["area"]

    @classmethod
    def from_circuitgroup(cls, comp, **kwargs):
        return cls(comp.subgroups,comp.connection, **kwargs)
    

class Module(Device):
    def __init__(self,subgroups,connection="series",location=None,
                 rotation=0,name=None,temperature=25,Suns=1.0):
        super().__init__(subgroups, connection,location=location,rotation=rotation,name=name)
        cells = self.findElementType(Cell)
        self.cells = cells
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns 
    def set_Suns(self,Suns):
        for cell in self.cells:
            cell.set_Suns(Suns=Suns)
    def set_temperature(self,temperature):
        super().set_temperature(temperature)
        self.temperature = temperature

    @classmethod
    def from_circuitgroup(cls, comp, **kwargs):
        return cls(comp.subgroups,comp.connection, **kwargs)


class ByPassDiode(ReverseDiode):
    max_I = None

Dbypass = partial(ByPassDiode)

class MultiJunctionCell(Device):
    def __init__(self,subcells=None,subgroups=None,Rs=0.1,location=None,
                 rotation=0,name=None,temperature=25,Suns=1.0):
        if subgroups is not None:
            components = subgroups
        else:
            components = subcells
            components.append(Resistor(cond=subcells[0].area/Rs))
        super().__init__(components, connection="series",location=location,rotation=rotation,
                         name=name,extent=components[0].extent)
        self.cells = []
        self.series_resistor = None
        for item in self.subgroups:
            if isinstance(item,Cell):
                self.cells.append(item)
            elif isinstance(item,Resistor):
                self.series_resistor = item
        self.area = self.cells[0].area
        if self.series_resistor is not None:
            self.series_resistor.aux["area"] = self.area
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        self.set_Suns(Suns)     
    def set_Suns(self,Suns):
        if isinstance(Suns,numbers.Number):
            Suns = [Suns]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_Suns(Suns=Suns[i])
    def set_JL(self,JL,Suns=1.0,temperature=25):
        if isinstance(JL,numbers.Number):
            JL = [JL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_JL(JL[i], Suns=Suns, temperature=temperature)
    def set_IL(self,IL,Suns=1.0,temperature=25):
        if isinstance(IL,numbers.Number):
            IL = [IL]*len(self.cells)
        for i, cell in enumerate(self.cells):
            cell.set_IL(IL[i], Suns=Suns, temperature=temperature)
    def set_temperature(self,temperature):
        super().set_temperature(temperature)
        self.temperature = temperature
    def specific_Rs_cond(self):
        if self.series_resistor is None:
            return np.inf
        return self.series_resistor.cond/self.area
    def Rs_cond(self):
        return self.specific_Rs_cond()*self.area
    def specific_Rs(self):
        return 1/self.specific_Rs_cond()
    def Rs(self):
        return 1/self.Rs_cond()
    def set_specific_Rs_cond(self,cond):
        if self.series_resistor is not None:
            self.series_resistor.set_cond(cond*self.area)
    def set_Rs_cond(self,cond):
        self.set_specific_Rs_cond(cond)
    def set_specific_Rs(self,Rs):
        self.set_specific_Rs_cond(1/Rs)
    def set_Rs(self,Rs):
        self.set_Rs_cond(1/Rs)

    @classmethod
    def from_circuitgroup(cls, comp, **kwargs):
        total_Rs = 0
        cell_area = -1
        subcells = []
        if comp.connection != "series":
            raise NotImplementedError
        for item in comp.subgroups:
            if isinstance(item,Cell):
                if cell_area < 0:
                        cell_area = item.area
                subcells.append(item)
            elif isinstance(item,Resistor):
                total_Rs += 1/item.cond
            else:
                raise NotImplementedError
        total_Rs *= cell_area # actually input a specific Rs
        if "Rs" not in kwargs and total_Rs > 0:
            kwargs["Rs"] = total_Rs
        return cls(subcells=subcells,**kwargs)

    # colormap: choose between cm.magma, inferno, plasma, cividis, viridis, turbo, gray
def draw_cells(self,display=True,show_names=False,colour_bar=False,colour_what="Vint",show_module_names=False,fontsize=9,min_value=None,max_value=None,title="Cells Layout",colormap=cm.plasma):
    if display:
        fig, ax = plt.subplots()
    shapes = []
    names = []
    Vints = []
    EL_Vints = []
    Is = []
    if isinstance(self,list):
        for element in self:
            if hasattr(element,"extent") and element.extent is not None:
                shapes_, names_, Vints_, EL_Vints_, Is_ = element.draw_cells(display=False)
                shapes.extend(shapes_)
                names.extend(names_)
                Vints.extend(Vints_)
                EL_Vints.extend(EL_Vints_)
                Is.extend(Is_)
                if show_module_names and element.name is not None and display:
                    ax.text(element.location[0], element.location[1]+element.extent[1]/2*1.05, element.name, fontsize=fontsize, color='black', ha="center", va="center")
    elif hasattr(self,"shape"): # a solar cell
        shapes.append(self.shape.copy())
        names.append(self.name)
        if self.operating_point is not None:
            Vints.append(self.diode_branch.operating_point[0])
            Is.append(self.operating_point[1])
        if self.aux is not None and "EL_Vint" in self.aux:
            EL_Vints.append(self.aux["EL_Vint"])
    else:
        for element in self.subgroups:
            if hasattr(element,"extent") and element.extent is not None:
                shapes_, names_, Vints_, EL_Vints_, Is_ = element.draw_cells(display=False)
                shapes.extend(shapes_)
                names.extend(names_)
                Vints.extend(Vints_)
                EL_Vints.extend(EL_Vints_)
                Is.extend(Is_)
    has_Vint = False
    has_EL_Vint = False
    has_power = False
    has_aux = False
    norm = None
    vmin = None
    vmax = None
    if len(EL_Vints)==len(shapes) and colour_what=="EL_Vint": # every cell has a EL_Vint
        has_EL_Vint = True
        vmin = min(EL_Vints)
        vmax=max(EL_Vints)
    elif len(Vints)==len(shapes) and colour_what=="Vint": # every cell has a Vint
        has_Vint = True
        vmin=min(Vints)
        vmax=max(Vints)
    elif len(Is)==len(shapes) and colour_what=="power": # every cell has a power
        has_power = True
        powers = np.array(Vints)*np.array(Is)
        vmin=np.min(powers)
        vmax=np.max(powers)
    elif colour_what is not None and len(colour_what)>0 and hasattr(self,"aux") and colour_what in self.aux:
        has_aux = True
        all_aux = self.aux[colour_what]
        vmin=min(all_aux)
        vmax=max(all_aux)
    elif colour_what is not None and len(colour_what)>0 and hasattr(self,"cells") and hasattr(self.cells[0],"aux") and colour_what in self.cells[0].aux:
        has_aux = True
        all_aux = []
        for cell in self.cells:
            all_aux.append(cell.aux[colour_what])
        vmin=min(all_aux)
        vmax=max(all_aux)
    if min_value is not None:
        vmin = max(min_value, vmin)
    if max_value is not None:
        vmax = min(max_value, vmax)
    if vmin is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormap
    
    rotation_ = 0
    x_mirror_ = 1
    y_mirror_ = 1
    location_ = [0,0]
    if not isinstance(self,list):
        rotation_ = self.rotation
        x_mirror_ = self.x_mirror
        y_mirror_ = self.y_mirror
        location_ = self.location
    for i, shape in enumerate(shapes):
        cos = np.cos(np.pi/180*rotation_)
        sin = np.sin(np.pi/180*rotation_)
        new_shape = shape.copy()
        new_shape[:,0] = shape[:,0]*cos + shape[:,1]*sin
        new_shape[:,1] = shape[:,1]*cos - shape[:,0]*sin
        if x_mirror_ == -1:
            new_shape[:,0] *= -1
        if y_mirror_ == -1:
            new_shape[:,1] *= -1
        new_shape[:,0] += location_[0]
        new_shape[:,1] += location_[1]

        shapes[i] = new_shape
    if display:
        for i, shape in enumerate(shapes):
            color = 'gray'
            if has_EL_Vint:
                color = cmap(norm(EL_Vints[i]))
            elif has_Vint:
                color = cmap(norm(Vints[i]))
            elif has_power:
                color = cmap(norm(powers[i]))
            elif has_aux:
                color = cmap(norm(all_aux[i]))
            polygon = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
            x = 0.5*(np.max(shape[:,0])+np.min(shape[:,0]))
            y = 0.5*(np.max(shape[:,1])+np.min(shape[:,1]))
            if show_names:
                ax.text(x, y, names[i], fontsize=fontsize, color='black')
            ax.add_patch(polygon)

        # ---- Tight axes from the actual polygons (fixes big blank space) ----
        xs = np.concatenate([s[:,0] for s in shapes])
        ys = np.concatenate([s[:,1] for s in shapes])
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        w, h = xmax - xmin, ymax - ymin
        pad = 0.05 * max(w, h)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect('equal')
        plt.gcf().canvas.manager.set_window_title(title)

        # 4) Inset colorbar (doesn't shrink the main axes)

        if colour_bar and norm is not None:
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            # place the bar just outside the right edge of the axes
            cax = inset_axes(
                ax, width="6%", height="90%", loc="center left",
                bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),  # x offset just to the right
                bbox_transform=ax.transAxes, borderpad=0
            )
            cbar = fig.colorbar(sm, cax=cax)
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            cbar.ax.yaxis.set_major_formatter(fmt)
            cbar.set_label(colour_what)
        else:
            fig.tight_layout()

        plt.show()
    return shapes, names, Vints, EL_Vints, Is
CircuitGroup.draw_cells = draw_cells

draw_modules = draw_cells

def wafer_shape(L=1, W=1, ingot_center=None, ingot_diameter=None, format=None, half_cut=True):
    if format is not None and format in wafer_formats:
        size = wafer_formats[format]["size"]
        L = size
        W = size
        ingot_diameter = wafer_formats[format]["diagonal"]
        ingot_center = [0,0]
        if half_cut:
            L = size/2
            ingot_center[1] = -L/2
    rect = np.array([[-W/2,-L/2],[W/2,-L/2],[W/2,L/2],[-W/2,L/2]]) # CCW
    if ingot_center is not None and ingot_diameter is not None:
        ingot_radius = ingot_diameter/2
        circle = Point(ingot_center[0], ingot_center[1]).buffer(ingot_radius, resolution=180)
        rect_poly = Polygon(rect)
        intersection = rect_poly.intersection(circle)
        if intersection.is_empty:
            if ingot_radius < W/2 and ingot_radius < L/2:
                intersection = circle
            else:
                intersection = rect
        elif intersection.geom_type == "Polygon":
            intersection = np.array(intersection.exterior.coords)
        else:
            assert(1==0)
    else:
        intersection = rect
    
    x = intersection[:,0]
    y = intersection[:,1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return {"shape": intersection, "area": area}

# note: always made at 25C 1 Sun
def make_solar_cell(Jsc=0.042, J01=10e-15, J02=2e-9, Rshunt=1e6, Rs=0.0, area=1.0, 
                    shape=None, breakdown_V=-10, J0_rev=100e-15,
                    J01_photon_coupling=0.0, Si_intrinsic_limit=True, **kwargs):
    elements = [CurrentSource(IL=Jsc, temp_coeff = Jsc_fractional_temp_coeff*Jsc),
                ForwardDiode(I0=J01,n=1),
                ForwardDiode(I0=J02,n=2)]
    if J01_photon_coupling > 0:
        elements.append(PhotonCouplingDiode(I0=J01_photon_coupling,n=1))
    if Si_intrinsic_limit:
        kwargs_to_pass = {}
        if "thickness" in kwargs:
            kwargs_to_pass["base_thickness"] = kwargs["thickness"]
        if "base_type" in kwargs:
            kwargs_to_pass["base_type"] = kwargs["base_type"]
        if "base_doping" in kwargs:
            kwargs_to_pass["base_doping"] = kwargs["base_doping"]
        elements.append(Intrinsic_Si_diode(**kwargs_to_pass))
    elements.extend([ReverseDiode(I0=J0_rev, n=1, V_shift = -breakdown_V),
                Resistor(cond=1/Rshunt)])
    if Rs == 0.0:
        cell = Cell(elements,"parallel",area=area,location=np.array([0.0,0.0]).astype(float),shape=shape,name="cell")
    else:
        group = CircuitGroup(elements,"parallel")
        cell = Cell([group,Resistor(cond=1/Rs)],"series",area=area,location=np.array([0.0,0.0]).astype(float),shape=shape,name="cell")
    return cell

# colormap: choose between cm.magma, inferno, plasma, cividis, viridis, turbo, gray        
draw_modules = draw_cells
    
def make_module(cells, num_strings=3, num_cells_per_halfstring=24, 
                         halfstring_resistor = 0.02, I0_rev = 1000e-15, butterfly=False):
    count = 0
    cell_strings = []
    num_half_strings = 1
    if butterfly:
        num_half_strings = 2
    else:
        halfstring_resistor /= 2
    for _ in range(num_strings):
        cell_halfstrings = []
        for _ in range(num_half_strings):
            cells_ = cells[count:count+num_cells_per_halfstring]
            count += num_cells_per_halfstring
            tile_elements(cells_,cols=2, x_gap = 0.1, y_gap = 0.1, turn=True)
            components = cells_
            if halfstring_resistor > 0:
                components += [Resistor(cond=1/halfstring_resistor)]
            halfstring = CircuitGroup(components,
                                        "series",name="cell_halfstring")
            cell_halfstrings.append(halfstring)
        if butterfly:
            tile_elements(cell_halfstrings, cols = 1, y_gap = 1, yflip=True)

        bypass_diode = ByPassDiode(I0=I0_rev, n=1, V_shift = 0)
        bypass_diode.max_I = 0.2*cells[0].area
        cell_strings.append(CircuitGroup(cell_halfstrings+[bypass_diode],
                                "parallel",name="cell_string"))

    tile_elements(cell_strings, rows=1, x_gap = 1, y_gap = 0.0)
    module = Module(cell_strings,"series")
    module.aux["halfstring_resistor"] = halfstring_resistor
    return module

def make_butterfly_module(cells, num_strings=3, num_cells_per_halfstring=24, 
                         halfstring_resistor = 0.02, I0_rev = 1000e-15):
    return make_module(cells, num_strings, num_cells_per_halfstring, 
                         halfstring_resistor, I0_rev, butterfly=True)

def reset_half_string_resistors(self:CircuitGroup, halfstring_resistor=None):
    if halfstring_resistor is None:
        if self.aux is not None and "halfstring_resistor" in self.aux:
            halfstring_resistor = self.aux["halfstring_resistor"]
    for element in self.subgroups:
        if isinstance(element,Cell):
            pass
        elif isinstance(element,CircuitGroup):
            element.reset_half_string_resistors(halfstring_resistor=halfstring_resistor)
        elif isinstance(element,Resistor):
            element.set_cond(1/halfstring_resistor)
CircuitGroup.reset_half_string_resistors = reset_half_string_resistors

def get_cell_col_row(self: CircuitGroup, fuzz_distance=0.2):
    shapes, _, _, _, _ = self.draw_cells(display=False)
    xs = []
    ys = []
    indices = []
    for i, shape in enumerate(shapes):
        xs.append(int(np.round(0.5*(np.max(shape[:,0])+np.min(shape[:,0]))/fuzz_distance)))
        ys.append(int(np.round(0.5*(np.max(shape[:,1])+np.min(shape[:,1]))/fuzz_distance)))
        indices.append(i)
    xs = np.array(xs)
    ys = np.array(ys)
    indices = np.array(indices)
    unique_xs = np.unique(xs)
    unique_ys = np.unique(ys)
    unique_ys = unique_ys[::-1] # reverse y such that y increases downwards
    cell_col_row = np.zeros((len(indices),2),dtype=int)
    map = np.zeros((len(indices)),dtype=int)
    inverse_map = np.zeros((len(indices)),dtype=int)
    count = 0
    for i, x in enumerate(unique_xs):
        for j, y in enumerate(unique_ys):
            find_ = np.where((xs==x) & (ys==y))[0]
            cell_col_row[indices[find_],0] = i
            cell_col_row[indices[find_],1] = j
            map[indices[find_]] = count
            inverse_map[count] = indices[find_]
            count += 1
    return cell_col_row, map, inverse_map
CircuitGroup.get_cell_col_row = get_cell_col_row

