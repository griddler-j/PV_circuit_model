import numpy as np
from PV_Circuit_Model.circuit_model import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, Point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter

class wafer_formats():
    formats = {
    "M0":   {"size": 15.6,   "diagonal": 20.52},
    "M2":   {"size": 15.675, "diagonal": 20.98},
    "G1":   {"size": 15.875, "diagonal": 22.3},
    "M4":   {"size": 16.170, "diagonal": 21.05},
    "M6":   {"size": 16.6,   "diagonal": 22.28},
    "M10":  {"size": 18.2,   "diagonal": 25.0},
    "M12":  {"size": 21.0,   "diagonal": 29.5},
    "M12+": {"size": 21.7,   "diagonal": 29.5}
    }

class intrinsic_Si():
    Jsc_fractional_temp_coeff = 0.0004

class Intrinsic_Si_diode(ForwardDiode):
    bandgap_narrowing_RT = np.array([[1.00E+10,	1.41E-03],
        [1.00E+14,	0.00145608],
        [3.00E+14,	0.00155279],
        [1.00E+15,	0.00187385],
        [3.00E+15,	0.00258644],
        [1.00E+16,	0.00414601],
        [3.00E+16,	0.00664397],
        [1.00E+17,	0.0112257],
        [3.00E+17,	0.018247],
        [1.00E+18,	0.0295337],
        [3.00E+18,	0.0421825],
        [1.00E+19,	0.0597645],
        [3.00E+19,	0.0811658],
        [1.00E+20,	0.113245]])
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
        self.VT = get_VT(self.temperature)
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
    def changeTemperature(self,temperature,rebuild_IV=True):
        self.temperature = temperature
        self.VT = get_VT(self.temperature)
        self.ni = get_ni(self.temperature)
        self.null_IV()
        if rebuild_IV:
            self.build_IV()
    def calc_I(self,V,get_dI_dV=False):
        ni = get_ni(self.temperature)
        VT = get_VT(self.temperature)
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
            VT = get_VT(self.temperature)
            Voc = 0.7
            for _ in range(10):
                I = self.calc_I(Voc)
                if I >= max_I and I <= max_I*1.1:
                    break
                Voc += VT*np.log(max_I/I)
        V = [self.V_shift-1.1,self.V_shift-1.0,self.V_shift,self.V_shift+0.02,self.V_shift+.08]+list(self.V_shift + Voc*np.log(np.arange(1,max_num_points))/np.log(max_num_points-1))
        V = np.array(V)
        return V
    
    def build_IV(self, V=None, max_num_points=100, *args, **kwargs):
        if V is None:
            V = self.get_V_range(max_num_points=max_num_points)
        I = self.calc_I(V)
        self.IV_table = np.array([V,I])

class Cell(CircuitGroup):
    def __init__(self,components,connection="series",area=None,location=None,
                 rotation=0,shape=None,name=None,temperature=25,Suns=1.0):
        x_extent = 0.0
        y_extent = 0.0
        if shape is not None:
            x_extent = np.max(shape[:,0])-np.min(shape[:,0])
            y_extent = np.max(shape[:,1])-np.min(shape[:,1])
        super().__init__(components, connection,location=location,rotation=rotation,
                         name=name,extent=np.array([x_extent,y_extent]).astype(float))
        self.area = area
        self.is_cell = True
        self.shape = shape
        self.build_IV()
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
    def copy(self,cell2):
        self.temperature = cell2.temperature
        self.Suns = cell2.Suns
        if self.series_resistor is not None:
            self.series_resistor.copy(cell2.series_resistor)
        for i, element in enumerate(self.diode_branch.subgroups):
            if i < len(cell2.diode_branch.subgroups):
                element.copy(cell2.diode_branch.subgroups[i])

    def set_Suns(self,Suns,rebuild_IV=True):
        self.Suns = Suns
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(Suns=Suns)
        if rebuild_IV:
            self.build_IV()
    
    def set_temperature(self,temperature,rebuild_IV=True):
        super().set_temperature(temperature)
        self.temperature = temperature
        if rebuild_IV:
            self.build_IV()

    def JL(self):
        JL = 0.0
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            JL += currentSource.IL
        return JL

    def IL(self):
        return self.JL()*self.area     
    
    def set_JL(self,JL,Suns=1.0,temperature=25,rebuild_IV=True):
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            if currentSource.tag != "defect":
                currentSource.refSuns = Suns
                currentSource.refIL = JL
                currentSource.refT = temperature
                currentSource.changeTemperatureAndSuns(
                    temperature=self.temperature,Suns=self.Suns,rebuild_IV=False)
                break
        if rebuild_IV:
            self.build_IV()

    def set_IL(self,IL,Suns=1.0,temperature=25,rebuild_IV=True):
        self.set_JL(IL/self.area,Suns=Suns,temperature=temperature,rebuild_IV=rebuild_IV)
    
    def J0(self,n):
        J0 = 0.0
        diodes = self.findElementType(ForwardDiode)
        for diode in diodes:
            if diode.n==n:
                if not isinstance(diode,Intrinsic_Si_diode) and not isinstance(diode,PhotonCouplingDiode):
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
    
    def set_J0(self,J0,n,temperature=25,rebuild_IV=True):
        diodes = self.findElementType(ForwardDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,Intrinsic_Si_diode) and diode.n==n and not isinstance(diode,PhotonCouplingDiode):
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature,rebuild_IV=False)
                break
        if rebuild_IV:
            self.build_IV()
    def set_J01(self,J0,temperature=25,rebuild_IV=True):
        self.set_J0(J0,n=1,temperature=temperature,rebuild_IV=rebuild_IV)
    def set_J02(self,J0,temperature=25,rebuild_IV=True):
        self.set_J0(J0,n=2,temperature=temperature,rebuild_IV=rebuild_IV)

    def set_I0(self,I0,n,temperature=25,rebuild_IV=True):
        self.set_J0(I0/self.area, n=n, temperature=temperature,rebuild_IV=rebuild_IV)
    def set_I01(self,I0,temperature=25,rebuild_IV=True):
        self.set_I0(I0,n=1,temperature=temperature,rebuild_IV=rebuild_IV)
    def set_I02(self,I0,temperature=25,rebuild_IV=True):
        self.set_I0(I0,n=2,temperature=temperature,rebuild_IV=rebuild_IV)

    def set_PC_J0(self,J0,n,temperature=25,rebuild_IV=True):
        diodes = self.findElementType(PhotonCouplingDiode)
        for diode in diodes:
            if diode.tag != "defect" and not isinstance(diode,Intrinsic_Si_diode) and diode.n==n:
                diode.refI0 = J0
                diode.refT = temperature
                diode.changeTemperature(temperature=self.temperature,rebuild_IV=False)
                break
        if rebuild_IV:
            self.build_IV()
    def set_PC_J01(self,J0,temperature=25,rebuild_IV=True):
        self.set_PC_J0(J0,n=1,temperature=temperature,rebuild_IV=rebuild_IV)
    def set_PC_I0(self,I0,n,temperature=25,rebuild_IV=True):
        self.set_PC_J0(I0/self.area, n=n, temperature=temperature,rebuild_IV=rebuild_IV)
    def set_PC_I01(self,I0,temperature=25,rebuild_IV=True):
        self.set_PC_I0(I0,n=1,temperature=temperature,rebuild_IV=rebuild_IV)
    
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

class Module(CircuitGroup):
    def __init__(self,subgroups,connection="series",location=np.array([0,0]),
                 rotation=0,cap_current=None,name=None,temperature=25,Suns=1.0):
        super().__init__(subgroups, connection,location=location,rotation=rotation,name=name)
        if self.location is None:
            self.location = np.array([0,0])
        self.cap_current = cap_current
        cells = self.findElementType(Cell,serialize=True)
        self.cells = cells
        self.temperature = temperature
        self.set_temperature(temperature)
        self.Suns = Suns
        self.set_Suns(Suns)     
    def set_Suns(self,Suns, rebuild_IV=True):
        for cell in self.cells:
            cell.set_Suns(Suns=Suns, rebuild_IV=False)
        if rebuild_IV:
            self.build_IV()
    def set_temperature(self,temperature, rebuild_IV=True):
        super().set_temperature(temperature,rebuild_IV=False)
        self.temperature = temperature
        if rebuild_IV:
            self.build_IV()
    def build_IV(self, max_num_points=500):
        super().build_IV(max_num_points=max_num_points,
                         cap_current=self.cap_current)
    
# colormap: choose between cm.magma, inferno, plasma, cividis, viridis, turbo, gray
def draw_cells(self: CircuitGroup,display=True,show_names=False,colour_bar=False,colour_what="Vint",min_value=None,max_value=None,title="Cells Layout",colormap=cm.plasma):
    shapes = []
    names = []
    Vints = []
    EL_Vints = []
    Is = []
    if hasattr(self,"shape"): # a solar cell
        shapes.append(self.shape.copy())
        names.append(self.name)
        if self.operating_point is not None and len(self.operating_point)==3:
            Vints.append(self.operating_point[2])
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
            
    for i, shape in enumerate(shapes):
        cos = np.cos(np.pi/180*self.rotation)
        sin = np.sin(np.pi/180*self.rotation)
        new_shape = shape.copy()
        new_shape[:,0] = shape[:,0]*cos + shape[:,1]*sin
        new_shape[:,1] = shape[:,1]*cos - shape[:,0]*sin
        if self.x_mirror == -1:
            new_shape[:,0] *= -1
        if self.y_mirror == -1:
            new_shape[:,1] *= -1
        new_shape[:,0] += self.location[0]
        new_shape[:,1] += self.location[1]

        shapes[i] = new_shape
    if display:
        fig, ax = plt.subplots()
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
                ax.text(x, y, names[i], fontsize=8, color='black')
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
        fig.tight_layout()
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

        plt.show()
    return shapes, names, Vints, EL_Vints, Is
CircuitGroup.draw_cells = draw_cells

def wafer_shape(L=1, W=1, ingot_center=None, ingot_diameter=None, format=None, half_cut=True):
    if format is not None and format in wafer_formats.formats:
        size = wafer_formats.formats[format]["size"]
        L = size
        W = size
        ingot_diameter = wafer_formats.formats[format]["diagonal"]
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
    return intersection, area

# note: always made at 25C 1 Sun
def make_solar_cell(Jsc=0.042, J01=10e-15, J02=2e-9, Rshunt=1e6, Rs=0.0, area=1.0, 
                    shape=None, breakdown_V=-10, J0_rev=100e-15,
                    J01_photon_coupling=0.0, Si_intrinsic_limit=True, **kwargs):
    elements = [CurrentSource(IL=Jsc, temp_coeff = intrinsic_Si.Jsc_fractional_temp_coeff*Jsc),
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



