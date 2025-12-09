from typing import Dict, Tuple, Set

from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
import re

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
        model_line = f".model {model_name} D(IS={I0:.16e} N={n:.16e})"

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
    
    else:

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
            out_lines.append(f"{name} {node_plus} {node_minus} {R:.16e}")
            return

        # Current source
        elif isinstance(obj, CurrentSource):
            name = state.next_isrc_name(obj)
            IL = float(obj.IL*area)
            # In your original netlists you used negative DC to represent PV current
            # source delivering current into the node. Keep that convention:
            value = -IL
            out_lines.append(f"{name} {node_plus} {node_minus} DC {value:.16e}")
            return

        # Diodes (forward, reverse, photon-coupling, or generic Diode)
        elif isinstance(obj, (ForwardDiode, ReverseDiode, Diode)):
            if obj.V_shift != 0:
                raise NotImplementedError(
                f"LTspice export does not yet support diodes that have non-zero Vshifts"
                )
            base_name = state.next_diode_name(obj)
            model_name = state.register_diode_model(obj,area)
            if isinstance(obj, ForwardDiode):
                # DF...  (forward)
                name = "DF" + base_name[1:]
            elif isinstance(obj, ReverseDiode):
                # DR...  (reverse)
                name = "DR" + base_name[1:]
    
            if isinstance(obj, ReverseDiode):
                out_lines.append(f"{name} {node_minus} {node_plus} {model_name}")
            else:
                out_lines.append(f"{name} {node_plus} {node_minus} {model_name}")
            return
            
        else: # don't do PhotonCouplingDiode, LT spice can't calculate it
            raise NotImplementedError(
                f"LTspice export does not yet support element type {type(obj).__name__}"
            )


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
            f"{vsource_name} {top_node} {gnd_node} DC {float(vsource_value):.16e}"
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
    add_vsource: bool = True,
    vsource_name: str = "V1",
    vsource_value: float = 0.0,
) -> None:
    
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


# ---------------------------------------------------------------------------
# LTspice → CircuitComponent importer
# ---------------------------------------------------------------------------

class _NetEdge:
    """
    Internal representation of a 2-terminal element in the netlist graph.
    """
    __slots__ = ("eid", "n1", "n2", "comp")

    def __init__(self, eid: int, n1: str, n2: str, comp: CircuitComponent):
        self.eid = eid
        self.n1 = n1
        self.n2 = n2
        self.comp = comp


def _parse_spice_value(token: str) -> float:
    """
    Parse a SPICE numeric value with optional suffix, e.g. 1k, 10meg, 5u.
    For our own exported netlists, plain float() is enough, but this makes
    the importer more robust if you edit values by hand.
    """
    token = token.strip()
    # Fast path
    try:
        return float(token)
    except ValueError:
        pass

    m = re.match(
        r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)([a-zA-Z]+)$",
        token
    )
    if not m:
        raise ValueError(f"Cannot parse SPICE value: {token!r}")
    base = float(m.group(1))
    suffix = m.group(2).lower()

    multipliers = {
        "t": 1e12,
        "g": 1e9,
        "meg": 1e6,
        "k": 1e3,
        "m": 1e-3,   # milli
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12,
        "f": 1e-15,
    }
    if suffix not in multipliers:
        raise ValueError(f"Unknown SPICE suffix {suffix!r} in {token!r}")
    return base * multipliers[suffix]


def _parse_diode_models(lines):
    """
    First pass: parse .model lines of the form:
      .model DMOD1 D(IS=1e-12 N=1.2)
    Returns dict: model_name -> {"IS": float, "N": float}
    """
    models: Dict[str, Dict[str, float]] = {}
    model_re = re.compile(r"\.model\s+(\S+)\s+D\((.*?)\)", re.IGNORECASE)

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        m = model_re.match(line)
        if not m:
            continue
        name = m.group(1)
        param_str = m.group(2)
        params: Dict[str, float] = {}
        for chunk in param_str.replace(",", " ").split():
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            k = k.strip().upper()
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = _parse_spice_value(v)
        models[name] = params
    return models


def _parse_netlist_to_edges(
    netlist_text: str,
    top_node: str,
    gnd_node: str,
):
    """
    Parse a LTspice netlist (compatible with export_ltspice_netlist)
    into a graph of _NetEdge objects and a node→edges adjacency map.
    """
    lines = netlist_text.splitlines()
    models = _parse_diode_models(lines)

    edges: Dict[int, _NetEdge] = {}
    nodes: Dict[str, Set[int]] = {}
    next_eid = 0

    def add_edge(n1: str, n2: str, comp: CircuitComponent):
        nonlocal next_eid
        eid = next_eid
        next_eid += 1
        e = _NetEdge(eid, n1, n2, comp)
        edges[eid] = e
        nodes.setdefault(n1, set()).add(eid)
        nodes.setdefault(n2, set()).add(eid)

    for raw in lines:
        # Strip comments
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("*"):
            continue

        # Skip control lines except .model (already parsed)
        if line[0] == ".":
            # .model handled in first pass; .dc/.end/etc are ignored here
            continue

        tokens = line.split()
        if not tokens:
            continue
        prefix = tokens[0][0].upper()

        # Voltage source from export_ltspice_netlist: we ignore it for now.
        if prefix == "V":
            continue

        # Resistor: Rname node1 node2 value
        if prefix == "R":
            if len(tokens) < 4:
                raise ValueError(f"Malformed resistor line: {line!r}")
            _, n1, n2, rval = tokens[:4]
            R = _parse_spice_value(rval)
            if R <= 0:
                cond = 0.0
            else:
                cond = 1.0 / R
            comp = Resistor(cond=cond)
            add_edge(n1, n2, comp)
            continue

        # Current source: Iname node+ node- DC value
        if prefix == "I":
            if len(tokens) < 5 or tokens[3].upper() != "DC":
                raise NotImplementedError(
                    f"Only DC current sources of form 'I ... DC value' "
                    f"are supported. Got: {line!r}"
                )
            _, n1, n2, _, ival = tokens[:5]
            val = _parse_spice_value(ival)
            # export_ltspice_netlist writes DC -IL, so IL = -val here
            IL = -val
            comp = CurrentSource(IL=IL)
            add_edge(n1, n2, comp)
            continue

        # Diodes: Dname anode cathode model
        if prefix == "D":
            if len(tokens) < 4:
                raise ValueError(f"Malformed diode line: {line!r}")
            raw_name, n1, n2, model_name = tokens[:4]
            model_params = models.get(model_name, {})
            I0 = model_params.get("IS", 1e-15)
            n = model_params.get("N", 1.0)
            kind = raw_name[1].upper() if len(raw_name) > 1 else "F"
            if kind == "R":
                cls = ReverseDiode
            else:
                cls = ForwardDiode

            comp = cls(I0=I0, n=n)
            add_edge(n1, n2, comp)
            continue

        # Anything else is currently unsupported (L, C, etc.)
        raise NotImplementedError(
            f"LTspice importer does not yet support element line: {line!r}"
        )

    # Ensure terminal nodes exist in the node table (even if unused).
    nodes.setdefault(top_node, set())
    nodes.setdefault(gnd_node, set())
    return edges, nodes

def _flatten_series_parallel(comp: CircuitComponent) -> CircuitComponent:
    """
    Recursively flatten nested *series* CircuitGroups.
    Parallel structure is preserved, but we still recurse into their children.
    """
    if not isinstance(comp, CircuitGroup):
        return comp
    flat_children = []
    for child in comp.subgroups:
        # Always recurse, regardless of connection type
        child_flat = _flatten_series_parallel(child)
        # Only splice if *parent* is series AND child is series
        if (isinstance(child_flat, CircuitGroup) and child_flat.connection == comp.connection):
            flat_children.extend(child_flat.subgroups)
        else:
            flat_children.append(child_flat)

    comp.subgroups = flat_children
    return comp

def _build_series_parallel_tree(
    edges: Dict[int, _NetEdge],
    nodes: Dict[str, Set[int]],
    top_node: str,
    gnd_node: str,
) -> CircuitComponent:
    """
    Given a graph of edges and adjacency, repeatedly apply parallel and
    series reductions until we end up with a single equivalent 2-terminal
    CircuitComponent between (top_node, gnd_node).
    This assumes the netlist came from a pure series/parallel CircuitGroup
    (which export_ltspice_netlist guarantees).
    """
    def combine_parallel() -> bool:
        """
        Find any pair (or more) of parallel edges (same endpoints)
        and merge them into CircuitGroup(..., connection='parallel').
        Returns True if any change was made.
        """
        pair_map: Dict[tuple[str, str], list[int]] = {}
        for eid, e in edges.items():
            a, b = sorted((e.n1, e.n2))
            pair_map.setdefault((a, b), []).append(eid)

        for (a, b), eids in pair_map.items():
            if len(eids) <= 1:
                continue

            # Build a parallel group of all these components
            subgroups = [edges[eid].comp for eid in eids]
            new_group = CircuitGroup(subgroups=subgroups, connection="parallel")

            # Remove old edges from adjacency
            for eid in eids:
                e = edges.pop(eid)
                nodes[e.n1].discard(eid)
                nodes[e.n2].discard(eid)

            # Add new edge
            new_eid = max(edges.keys(), default=-1) + 1
            new_edge = _NetEdge(new_eid, a, b, new_group)
            edges[new_eid] = new_edge
            nodes.setdefault(a, set()).add(new_eid)
            nodes.setdefault(b, set()).add(new_eid)
            return True  # one change is enough for this pass
        return False

    def combine_series() -> bool:
        """
        Find any internal node (≠ top_node,gnd_node) of degree 2 and
        merge the two incident edges into a series CircuitGroup.
        Returns True if a change was made.
        """
        for node_name, incident in list(nodes.items()):
            if node_name in (top_node, gnd_node):
                continue
            if len(incident) != 2:
                continue

            e1_id, e2_id = tuple(incident)
            e1 = edges[e1_id]
            e2 = edges[e2_id]

            # Find the "other" endpoints
            def other_node(e: _NetEdge, node: str) -> str:
                return e.n2 if e.n1 == node else e.n1

            a = other_node(e1, node_name)
            b = other_node(e2, node_name)

            # If a == b, we'd form a loop; skip this node
            if a == b:
                continue

            new_group = CircuitGroup(
                subgroups=[e1.comp, e2.comp],
                connection="series",
            )

            # Remove old edges and the internal node
            for eid in (e1_id, e2_id):
                e = edges.pop(eid)
                nodes[e.n1].discard(eid)
                nodes[e.n2].discard(eid)

            nodes.pop(node_name, None)

            # Add new edge between a and b
            new_eid = max(edges.keys(), default=-1) + 1
            new_edge = _NetEdge(new_eid, a, b, new_group)
            edges[new_eid] = new_edge
            nodes.setdefault(a, set()).add(new_eid)
            nodes.setdefault(b, set()).add(new_eid)
            return True
        return False

    # Repeatedly apply parallel then series reduction until no more changes
    while True:
        changed = combine_parallel()
        if changed:
            continue
        changed = combine_series()
        if not changed:
            break

    # At the end, we expect one or more edges between top_node and gnd_node.
    remaining = [
        e for e in edges.values()
        if {e.n1, e.n2} == {top_node, gnd_node}
    ]

    if not remaining:
        raise ValueError(
            "Could not reduce LTspice netlist to a single 2-terminal component "
            f"between {top_node!r} and {gnd_node!r}. "
            "Is this a pure series/parallel network produced by export_ltspice_netlist?"
    )

    if len(remaining) == 1:
        root = remaining[0].comp
    else:
        # Multiple remaining edges => they are all in parallel between the terminals
        root = CircuitGroup(
            subgroups=[e.comp for e in remaining],
            connection="parallel",
    )
        
    # Normalize structure: flatten nested groups with same connection
    return _flatten_series_parallel(root)


def circuit_from_ltspice_netlist_text(
    netlist_text: str,
    top_node: str = "N001",
    gnd_node: str = "0",
) -> CircuitComponent:
    """
    Construct a CircuitComponent (usually a CircuitGroup) from a LTspice
    netlist text that was produced by export_ltspice_netlist.

    Notes / limitations:
    - Only R, I (DC), and D devices are supported (no C, L, etc.).
    - The original high-level structure (Module, Cell, nested groups)
      is not recoverable; we rebuild an equivalent series/parallel tree.
    - We cannot distinguish ForwardDiode vs ReverseDiode vs generic Diode
      from LTspice alone, so all diodes become generic Diode objects
      with the same (I0, n) extracted from the .model lines.
    """
    edges, nodes = _parse_netlist_to_edges(
        netlist_text=netlist_text,
        top_node=top_node,
        gnd_node=gnd_node,
    )
    return _build_series_parallel_tree(edges, nodes, top_node, gnd_node)


def import_ltspice_netlist(
    filename: str,
    top_node: str = "N001",
    gnd_node: str = "0",
) -> CircuitComponent:
    """
    Convenience wrapper: read a LTspice netlist file from disk and
    rebuild an equivalent CircuitComponent between (top_node, gnd_node).

    Example
    -------
    >>> circuit = import_ltspice_netlist("my_circuit.net")
    >>> circuit.build_IV()
    """
    with open(filename, "r") as f:
        netlist_text = f.read()
    return circuit_from_ltspice_netlist_text(
        netlist_text=netlist_text,
        top_node=top_node,
        gnd_node=gnd_node,
    )


if __name__ == "__main__": 
    # check that I can export, import back and get the same results

    np.random.seed(1)
    module = quick_butterfly_module(Si_intrinsic_limit=False) # no instrinic Si diodes
    for cell in tqdm(module.cells): # pluck out all the cell ReverseDiodes
        diode_branch = cell.diode_branch
        diode_branch.subgroups = [element for element in diode_branch.subgroups if not isinstance(element,ReverseDiode)]
        cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
        cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
        cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
    module.cells[0].set_JL(module.cells[0].JL()/2)
    module = module.clone() # cloning will reconstruct the module, leading to corrected num_circuit_elements
    export_ltspice_netlist(module, "test_circuit.net")
    recovered = import_ltspice_netlist("test_circuit.net")

    module.plot()
    recovered.plot()
    recovered.show()

    
