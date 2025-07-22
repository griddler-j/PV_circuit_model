import numpy as np
from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class Module(CircuitGroup):
    def __init__(self,subgroups,connection="series",location=None,
                 rotation=0,cap_current=None,name=None,temperature=25,Suns=1.0):
        super().__init__(subgroups, connection,location=location,rotation=rotation,name=name)
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
def draw_modules(modules,show_names=False,colour_what="EL_Vint",show_module_names=False):
    all_shapes = []
    all_names = []
    all_Vints = []
    all_EL_Vints = []
    xlim_ = [modules[0].location[0]-modules[0].extent[0]/2*1.1, modules[0].location[0]+modules[0].extent[0]/2*1.1]
    ylim_ = [modules[0].location[1]-modules[0].extent[1]/2*1.1, modules[0].location[1]+modules[0].extent[1]/2*1.1]
    fig, ax = plt.subplots()
    for module in modules:
        shapes, names, Vints, EL_Vints = module.draw_cells(display=False)
        all_shapes.extend(shapes)
        all_names.extend(names)
        all_Vints.extend(Vints)
        all_EL_Vints.extend(EL_Vints)
        xlim_[0] = min(xlim_[0], module.location[0]-module.extent[0]/2*1.1)
        xlim_[1] = max(xlim_[1], module.location[0]+module.extent[0]/2*1.1)
        ylim_[0] = min(ylim_[0], module.location[1]-module.extent[1]/2*1.1)
        ylim_[1] = max(ylim_[1], module.location[1]+module.extent[1]/2*1.1)
        if show_module_names and module.name is not None:
            ax.text(module.location[0], module.location[1]+module.extent[1]/2*1.05, module.name, fontsize=9, color='black', ha="center", va="center")

    has_Vint = False
    has_EL_Vint = False
    has_aux = False
    if len(all_EL_Vints)==len(all_shapes) and colour_what=="EL_Vint":
        has_EL_Vint = True
        norm = mcolors.Normalize(vmin=min(all_EL_Vints), vmax=max(all_EL_Vints))
        cmap = cm.viridis  # You can use other colormaps like 'plasma', 'coolwarm', etc.
    elif len(all_Vints)==len(all_shapes) and colour_what=="Vint": # every cell has a Vint
        has_Vint = True
        norm = mcolors.Normalize(vmin=min(all_Vints), vmax=max(all_Vints))
        cmap = cm.viridis  # You can use other colormaps like 'plasma', 'coolwarm', etc.
    elif len(colour_what)>0 and hasattr(modules[0],"aux") and colour_what in modules[0].aux:
        has_aux = True
        all_aux = []
        for module in modules:
            all_aux.extend(module.aux[colour_what])
        norm = mcolors.Normalize(vmin=min(all_aux), vmax=max(all_aux))
        cmap = cm.viridis
    elif len(colour_what)>0 and hasattr(modules[0].cells[0],"aux") and colour_what in modules[0].cells[0].aux:
        has_aux = True
        all_aux = []
        for module in modules:
            for cell in module.cells:
                all_aux.append(cell.aux[colour_what])
        norm = mcolors.Normalize(vmin=min(all_aux), vmax=max(all_aux))
        cmap = cm.viridis

    for i, shape in enumerate(all_shapes):
        color = 'skyblue'
        if has_EL_Vint:
            color = cmap(norm(all_EL_Vints[i]))
        elif has_Vint:
            color = cmap(norm(all_Vints[i]))
        elif has_aux:
            color = cmap(norm(all_aux[i]))
        polygon = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
        x = 0.5*(np.max(shape[:,0])+np.min(shape[:,0]))
        y = 0.5*(np.max(shape[:,1])+np.min(shape[:,1]))
        if show_names:
            ax.text(x, y, all_names[i], fontsize=9, color='black', ha="center", va="center")
        ax.add_patch(polygon)
    ax.set_xlim(xlim_[0],xlim_[1])
    ax.set_ylim(ylim_[0],ylim_[1])
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()
    
def make_butterly_module(cells, num_strings=3, num_cells_per_halfstring=24, 
                         halfstring_resistor = 0.02, I0_rev = 1000e-15):
    count = 0
    cell_strings = []
    for _ in range(num_strings):
        cell_halfstrings = []
        for _ in range(2):
            cells_ = cells[count:count+num_cells_per_halfstring]
            count += num_cells_per_halfstring
            tile_elements(cells_, 
                        rows=num_cells_per_halfstring // 2, cols=2, 
                        x_gap = 0.1, y_gap = 0.1, turn=True)
            halfstring = CircuitGroup(cells_ + [Resistor(cond=1/halfstring_resistor)],
                                        "series",name="cell_halfstring")
            cell_halfstrings.append(halfstring)
        cell_halfstrings[1].y_mirror = -1
        cell_halfstrings[1].location[1] -= cell_halfstrings[1].extent[1] + 1

        cell_strings.append(CircuitGroup(cell_halfstrings+[ReverseDiode(I0=I0_rev, n=1, V_shift = 0)],
                                "parallel",name="cell_string"))
    tile_elements(cell_strings, rows=1, x_gap = 1, y_gap = 0.0, turn=False)
    module = Module(cell_strings,"series",cap_current=cells[0].IL()*3)
    module.aux["halfstring_resistor"] = halfstring_resistor
    return module

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
    shapes, names, _, _ = self.draw_cells(display=False)
    xs = []
    ys = []
    indices = []
    for i, shape in enumerate(shapes):
        xs.append(int(np.round(0.5*(np.max(shape[:,0])+np.min(shape[:,0]))/fuzz_distance)))
        ys.append(int(np.round(0.5*(np.max(shape[:,1])+np.min(shape[:,1]))/fuzz_distance)))
        indices.append(int(names[i]))
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