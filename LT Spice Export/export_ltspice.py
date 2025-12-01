"""
export_ltspice.py

Export a CircuitGroup (from PV_Circuit_Model.circuit_model) as an LTspice netlist.

Usage
-----
from PV_Circuit_Model.circuit_model import CircuitGroup
from export_ltspice import export_ltspice_netlist

# Suppose `top_group` is your top-level CircuitGroup
export_ltspice_netlist(
    top_group,
    filename="circuit.net",
    top_node="N001",  # node at "top" of group
    gnd_node="0",     # LTspice ground
    add_vsource=False,
    vsource_name="V1",
    vsource_value=0.0
)
"""

from typing import Dict, Tuple, Set

from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
import time

class _ExportState:
    """
    Holds counters, node allocator and diode-model registry.
    """

    def __init__(self) -> None:
        self.used_nodes: Set[str] = set()
        self._next_node_id: int = 1

        self.res_counter: int = 1
        self.isrc_counter: int = 1
        self.diode_counter: int = 1

        # key: (cls_name, I0, n, V_shift) -> (model_name, model_line)
        self.diode_models: Dict[Tuple[str, float, float, float], Tuple[str, str]] = {}

    # ---------- node naming ----------

    def register_node(self, name: str) -> None:
        self.used_nodes.add(str(name))

    def new_node(self) -> str:
        """Return a fresh node name like N001, N002, ... avoiding collisions."""
        while True:
            candidate = f"N{self._next_node_id:03d}"
            self._next_node_id += 1
            if candidate not in self.used_nodes:
                self.used_nodes.add(candidate)
                return candidate

    # ---------- element naming ----------

    def next_res_name(self, element: Resistor) -> str:
        if getattr(element, "tag", None):
            return f"R{element.tag}"
        name = f"R{self.res_counter}"
        self.res_counter += 1
        return name

    def next_isrc_name(self, element: CurrentSource) -> str:
        if getattr(element, "tag", None):
            return f"I{element.tag}"
        name = f"I{self.isrc_counter}"
        self.isrc_counter += 1
        return name

    def next_diode_name(self, element: Diode) -> str:
        if getattr(element, "tag", None):
            return f"D{element.tag}"
        name = f"D{self.diode_counter}"
        self.diode_counter += 1
        return name

    # ---------- diode models ----------

    def register_diode_model(self, d: Diode, area) -> str:
        """
        Return model name for this diode instance, creating it if needed.
        We dedupe by (class, I0, n, V_shift).
        """
        cls_name = type(d).__name__
        I0 = float(getattr(d, "I0", 1e-15))*area
        n = float(getattr(d, "n", 1.0))
        V_shift = float(getattr(d, "V_shift", 0.0))

        key = (cls_name, I0, n, V_shift)
        if key in self.diode_models:
            return self.diode_models[key][0]

        model_name = f"DMOD{len(self.diode_models) + 1}"
        model_line = f".model {model_name} D(IS={I0:.3e} N={n:.3g})"

        self.diode_models[key] = (model_name, model_line)
        return model_name


def _emit_component(
    obj,
    node_plus: str,
    node_minus: str,
    state: _ExportState,
    out_lines: list,
    area = 1
) -> None:
    """
    Recursively emit LTspice lines for obj between node_plus and node_minus.
    CircuitGroup is treated as a 2-terminal subcircuit, expanded inline.
    """
    # Groups: series or parallel
    if isinstance(obj, CircuitGroup):
        subs = obj.subgroups
        if not subs:
            return
        
        if area==1 and isinstance(obj,Cell):
            area = obj.area

        if obj.connection == "series":
            # chain subs: node_plus -- sub0 -- n1 -- sub1 -- ... -- node_minus
            current_plus = node_plus
            for i, sub in enumerate(subs):
                current_minus = node_minus if i == len(subs) - 1 else state.new_node()
                _emit_component(sub, current_plus, current_minus, state, out_lines,area)
                current_plus = current_minus
        else:
            # treat anything non-"series" as parallel: each subgroup gets same terminals
            for sub in subs:
                _emit_component(sub, node_plus, node_minus, state, out_lines,area)
        return

    # Leaves: CircuitElement subclasses
    if not isinstance(obj, CircuitElement):
        raise NotImplementedError(
            f"LTspice export does not know how to handle object of type {type(obj).__name__}"
        )

    # Resistor
    if isinstance(obj, Resistor):
        name = state.next_res_name(obj)
        cond = float(obj.cond*area)
        # Avoid 0-ohm (SPICE hates it); use large resistor if cond == 0
        if cond == 0.0:
            R = 1e12
        else:
            R = 1.0 / cond
        out_lines.append(f"{name} {node_plus} {node_minus} {R:.6g}")
        return

    # Current source
    if isinstance(obj, CurrentSource):
        name = state.next_isrc_name(obj)
        IL = float(obj.IL*area)
        # In your original netlists you used negative DC to represent PV current
        # source delivering current into the node. Keep that convention:
        value = -IL
        out_lines.append(f"{name} {node_plus} {node_minus} DC {value:.6g}")
        return

    # Diodes (forward, reverse, photon-coupling, or generic Diode)
    if isinstance(obj, (ForwardDiode, ReverseDiode, Diode)):
        if isinstance(obj,ReverseDiode) and obj.V_shift != 0:
            pass
        else:
            name = state.next_diode_name(obj)
            model_name = state.register_diode_model(obj,area)
            if isinstance(obj, ReverseDiode):
                out_lines.append(f"{name} {node_minus} {node_plus} {model_name}")
            else:
                out_lines.append(f"{name} {node_plus} {node_minus} {model_name}")
            return
    
    # don't do PhotonCouplingDiode, 

    # If we hit an element type we don't know how to translate, fail loudly
    # raise NotImplementedError(
    #     f"LTspice export does not yet support element type {type(obj).__name__}"
    # )


def _build_netlist_text(
    group: CircuitGroup,
    top_node: str,
    gnd_node: str,
    add_vsource: bool,
    vsource_name: str,
    vsource_value: float,
) -> str:
    """
    Internal: walk the CircuitGroup and produce a netlist string.
    """
    state = _ExportState()
    state.register_node(top_node)
    state.register_node(gnd_node)

    lines = []

    # Header
    title = getattr(group, "name", None) or "CircuitGroup"
    lines.append(f"* LTspice netlist generated from {title}")
    lines.append("* Node 0 is global ground")
    lines.append("")

    # Optional supply (you can also add this in LTspice manually)
    if add_vsource:
        lines.append(
            f"{vsource_name} {top_node} {gnd_node} DC {float(vsource_value):.6g}"
        )
        lines.append("")

    # Flatten the group
    _emit_component(group, top_node, gnd_node, state, lines)
    lines.append("")

    # Diode models
    if state.diode_models:
        lines.append("* Diode models")
        for _, (_, model_line) in sorted(
            state.diode_models.items(), key=lambda kv: kv[1][0]
        ):
            lines.append(model_line)
        lines.append("")

    # Simple sweep by default
    lines.append(f".dc {vsource_name} {gnd_node} 0.6 0.01")
    lines.append(".end")
    lines.append("")

    return "\n".join(lines)


def export_ltspice_netlist(
    group: CircuitGroup,
    filename: str,
    top_node: str = "N001",
    gnd_node: str = "0",
    add_vsource: bool = False,
    vsource_name: str = "V1",
    vsource_value: float = 0.0,
) -> None:
    """
    Export a CircuitGroup as an LTspice netlist.

    Parameters
    ----------
    group : CircuitGroup
        The top-level circuit group to export. Treated as a 2-terminal element
        between `top_node` and `gnd_node`.
    filename : str
        Path of the netlist file to write.
    top_node : str, default "N001"
        Node name at the "top" of the group (LTspice node label).
    gnd_node : str, default "0"
        Node name for the bottom terminal (typically "0" in LTspice).
    add_vsource : bool, default False
        If True, add a DC voltage source between top_node and gnd_node.
    vsource_name : str, default "V1"
        Element name for the added voltage source.
    vsource_value : float, default 0.0
        DC value of the added voltage source (Volts).

    Returns
    -------
    None
        Writes `filename` to disk.
    """
    netlist_text = _build_netlist_text(
        group=group,
        top_node=top_node,
        gnd_node=gnd_node,
        add_vsource=add_vsource,
        vsource_name=vsource_name,
        vsource_value=vsource_value,
    )
    with open(filename, "w") as f:
        f.write(netlist_text)


if __name__ == "__main__": 

    # build some toy circuit as a CircuitGroup
    R = Resistor(cond=1/100.0, tag="LOAD")      # 100 Ω
    Iph = CurrentSource(IL=0.042, tag="PHOT")   # 42 mA
    D = ForwardDiode(I0=1e-14, n=1.0, tag="D1")
    D = ForwardDiode(I0=1e-9, n=2.0, tag="D2")

    # say: (Iph || D) in series with R
    parallel_branch = CircuitGroup([Iph, D], connection="parallel", name="Branch")
    top_group = CircuitGroup([parallel_branch, R], connection="series", name="Top")
    # top_group.build_IV()
    # plt.plot(top_group.IV_table[0,:],top_group.IV_table[1,:])
    # plt.scatter(top_group.IV_table[0,:],top_group.IV_table[1,:])
    # plt.show()
    # assert(1==0)

















    np.random.seed(1)
    module = quick_butterfly_module(Si_intrinsic_limit=False)

    # module2 = quick_butterfly_module(num_strings=3, num_cells_per_halfstring=20) 
    # module3 = quick_module(num_strings=3, num_cells_per_halfstring=20)
    # tile_elements([module2,module3],rows=1,x_gap=20,turn=False)
    # draw_modules([module2,module3],show_names=True)
    















    for cell in tqdm(module.cells):
        diode_branch = cell.diode_branch
        diode_branch.subgroups = [element for element in diode_branch.subgroups if not isinstance(element,ReverseDiode)]
        # diode_branch.null_all_IV()
        # cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
        # cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        # cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        # cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
        # cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))

    # t2 = 0
    # for rep in range(10):
    #     module.null_all_IV()
    #     t1 = time.time()
    #     module.build_IV()
    #     t2 += time.time()-t1
    #     print(f"time = {time.time()-t1}")
    # print(f"avg time = {t2/10}")
    # find_ = np.where((module.IV_table[0,:]>=0) & (module.IV_table[1,:]<=0))[0]
    # print(len(find_))
    # module.plot()
    # module.show()

    # # export to LTspice
    # export_ltspice_netlist(
    #     module,
    #     filename="pv_string.net",
    #     top_node="NOUT",
    #     gnd_node="0",
    #     add_vsource=True,  # or True if you want a source in the netlist
    # )

    modules = []
    for i in tqdm(range(26)):
        module_ = circuit_deepcopy(module)
        # for cell in module_.cells:
        #     cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
        #     cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        #     cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        #     cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
        #     cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
        modules.append(module_)

    string = CircuitGroup(modules)

    # t2 = 0
    # for rep in range(10):
    #     string.null_all_IV()
    #     t1 = time.time()
    #     string.build_IV()
    #     t2 += time.time()-t1
    #     print(f"time = {time.time()-t1}")
    # print(f"avg time = {t2/10}")
    # find_ = np.where((string.IV_table[0,:]>=0) & (string.IV_table[1,:]<=0))[0]
    # print(len(find_))
    # string.plot()
    # string.show()


    # # export to LTspice
    # export_ltspice_netlist(
    #     string,
    #     filename="pv_string.net",
    #     top_node="NOUT",
    #     gnd_node="0",
    #     add_vsource=True,  # or True if you want a source in the netlist
    # )

    strings = []
    for i in tqdm(range(40)):
        string_ = circuit_deepcopy(string)
        for module_ in string_.subgroups:
            for cell in module_.cells:
                cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
                cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
                cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
                cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
                cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
        strings.append(string_)

    block = CircuitGroup(strings,connection="parallel")

    t2 = 0
    for rep in range(1):
        block.null_all_IV()
        t1 = time.time()
        block.build_IV()
        t2 += time.time()-t1
        print(f"time = {time.time()-t1}")
    # for string_ in block.subgroups:
    #     plt.plot(string_.IV_table[0,:],string_.IV_table[1,:])
    #     plt.scatter(string_.IV_table[0,:],string_.IV_table[1,:],s=2)
    #     plt.show()
    plt.plot(block.IV_table[0,:],block.IV_table[1,:])
    plt.scatter(block.IV_table[0,:],block.IV_table[1,:],s=2)
    plt.show()
    print(f"avg time = {t2/1}")
    find_ = np.where((block.IV_table[0,:]>=0) & (block.IV_table[1,:]<=0))[0]
    print(len(find_))
    block.plot()
    block.show()


    # export to LTspice
    export_ltspice_netlist(
        block,
        filename="pv_string.net",
        top_node="NOUT",
        gnd_node="0",
        add_vsource=True,  # or True if you want a source in the netlist
    )