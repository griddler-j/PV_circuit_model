import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import PV_Circuit_Model.utilities as utilities
import PV_Circuit_Model.utilities_silicon as silicon
from tqdm import tqdm
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, TypeVar, Union
try:
    import PV_Circuit_Model.IV_jobs as IV_jobs
except Exception as e:
    raise ImportError(
        "Failed to import the compiled extension 'PV_Circuit_Model.IV_jobs'.\n"
        "This usually means the C++/Cython extension was not built correctly.\n\n"
        "Try:\n"
        "  pip install -e .\n"
        "or (if installing from source):\n"
        "  pip install PV_Circuit_Model\n\n"
        "If you are on Windows, ensure Visual Studio Build Tools are installed.\n"
    ) from e

import gc

def in_notebook() -> Tuple[bool, Optional[Any], Optional[Any]]:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None or not hasattr(ip, "kernel"):
            return False, None, None
        from IPython.display import HTML, display
        return True, HTML, display
    except Exception:
        return False, None, None
IN_NOTEBOOK, HTML, display = in_notebook()

# Solver tuning pulled from ParameterSet to control IV remeshing thresholds.
solver_env_variables = utilities.ParameterSet.get_set("solver_env_variables")
if solver_env_variables is None:
    raise TypeError("Cannot load solver_env_variables")
REMESH_POINTS_DENSITY = solver_env_variables["REMESH_POINTS_DENSITY"]
REMESH_NUM_ELEMENTS_THRESHOLD = solver_env_variables["REMESH_NUM_ELEMENTS_THRESHOLD"]
DEFAULT_MAX_DIODE_CURRENT_DENSITY = solver_env_variables["DEFAULT_MAX_DIODE_CURRENT_DENSITY"]
      
T_CircuitComponent = TypeVar("T_CircuitComponent", bound="CircuitComponent")

# Base class for any circuit component; manages IV artifacts and serialization hooks.
class CircuitComponent(utilities.Artifact):
    """Base class for all circuit components.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import Resistor
        component = Resistor(cond=1.0)
        isinstance(component, CircuitComponent) # True
        ```
    """
    _critical_fields = ()
    _parent_pointer_name = "parent"
    _parent_pointer_class = None
    _ephemeral_fields = ("IV_V", "IV_I", "IV_V_lower", "IV_I_lower", "IV_V_upper", "IV_I_upper","extrapolation_allowed", "extrapolation_dI_dV",
                  "has_I_domain_limit","job_heap", "refined_IV","operating_point","bottom_up_operating_point")
    _dont_serialize = ("circuit_depth", "num_circuit_elements", "IV_table", "dark_IV_table")
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
    registered_type_numbers = set()

    def __init__(self, tag: Optional[str] = None) -> None:
        self.parent = None
        self.tag = tag
        CircuitComponent.__compile__(self)

    def __post_init__(self):
        utilities.Artifact.__post_init__(self)
        CircuitComponent.__compile__(self)
        
    def __compile__(self):
        if "circuit_diagram_extent" not in self.__dict__:
            self.circuit_diagram_extent = [0, 0.8]
        if "aux" not in self.__dict__:
            self.aux = {}
        if "tag" not in self.__dict__:
            self.tag = None
        if "extrapolation_allowed" not in self.__dict__:
            self.extrapolation_allowed = [False,False]
            self.extrapolation_dI_dV = [0,0]
            self.has_I_domain_limit = [False,False]

    def __init_subclass__(cls, *, _type_number: int = -1, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if _type_number == -1:
            for base in cls.__mro__[1:]:
                tn = getattr(base, "_type_number", -1)
                if tn >= 0:
                    _type_number = tn
                    cls._type_number = _type_number
                    return
        if _type_number >= 0:
            if _type_number in CircuitComponent.registered_type_numbers:
                raise ValueError(
                        f"_type_number {_type_number} already used"
                    )
            CircuitComponent.registered_type_numbers.add(_type_number)
        cls._type_number = _type_number

    def __len__(self) -> int:
        """Return the number of circuit elements contained.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import *
            group = R(1) + R(2)
            len(group) #2
            ```
        """
        return self.num_circuit_elements

    @property
    def IV_table(self) -> Optional[np.ndarray]:
        if self.IV_V is None or self.IV_I is None:
            return None
        # This allocates a fresh 2xN array for user-land / plotting.
        return np.stack([self.IV_V, self.IV_I], axis=0)

    @IV_table.setter
    def IV_table(self, value: Optional[np.ndarray]) -> None:
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

    def null_IV(self) -> None:
        self.clear_ephemeral_fields()
        if self.parent is not None:
            self.parent.null_IV()

    def null_all_IV(self) -> None:
        self.clear_ephemeral_fields()
        if isinstance(self,CircuitGroup):
            for element in self.subgroups:
                element.null_all_IV()

    def build_IV(self) -> None:
        """Compute the IV curve for this component or group.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            r = R(1.0)
            r.build_IV()
            ```
        """
        gc.disable()
        # Run IV solver; if needed, expand diode current range and retry.
        self.job_heap = IV_jobs.IV_Job_Heap(self)
        success = self.job_heap.run_IV()
        if not success:
            diodes = self.findElementType(Diode)
            for _ in range(5): # try to autorange a few times before giving up
                for diode in diodes:
                    diode.max_I *= 10
                self.null_all_IV()
                self.job_heap = IV_jobs.IV_Job_Heap(self)
                if self.job_heap.run_IV():
                    break
            if not success:
                raise RuntimeError
        gc.enable()

    def refine_IV(self) -> None:
        """Refine the IV curve near the current operating point.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            r = R(1.0)
            r.build_IV()
            r.set_operating_point(V=0.1)
            r.refine_IV()
            ```
        """
        if hasattr(self,"job_heap") and getattr(self,"operating_point",None) is not None:
            gc.disable()
            # Refine around the current operating point if available.
            self.job_heap.refine_IV()
            self.refined_IV = True
            gc.enable()

    def calc_uncertainty(self) -> None:
        if hasattr(self,"job_heap"):
            if hasattr(self.job_heap,"calc_uncertainty"): # python version does not have this
                self.job_heap.calc_uncertainty()

    def __call__(self, *, atomic: bool = True) -> "CircuitComponent":
        # Clone while optionally marking as atomic to avoid flattening.
        clone = self.clone()
        clone._is_atomic = atomic
        return clone

    def __add__(self, other: "CircuitComponent") -> "CircuitGroup":
        """Connect components in series with the `+` operator.

        Args:
            other (CircuitComponent): The component to place in series.

        Returns:
            CircuitGroup: A series-connected circuit group.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            group = R(1.0) + R(2.0)
            group.connection # 'series'
            ```
        """
        if not isinstance(other, CircuitComponent):
            return NotImplemented
        return series(self, other, flatten_connection_=True)
    
    def __or__(self, other: "CircuitComponent") -> "CircuitGroup":
        """Connect components in parallel with the `|` operator.

        Args:
            other (CircuitComponent): The component to place in parallel.

        Returns:
            CircuitGroup: A parallel-connected circuit group.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            group = R(1.0) | R(2.0)
            group.connection #'parallel'
            ```
        """
        if not isinstance(other, CircuitComponent):
            return NotImplemented
        return parallel(self, other, flatten_connection_=True)
    
    def __mul__(self, other: Union[int, float]) -> "CircuitGroup":
        """Repeat a component in series using `*`.

        Args:
            other (Union[int, float]): Number of repeats.

        Returns:
            CircuitGroup: A series-connected circuit group.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            group = R(1.0) * 3
            len(group) #3
            ```
        """
        # component * scalar
        if isinstance(other, (int, float)):
            return series(*[circuit_deepcopy(self)() for _ in range(int(other))])
        return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> "CircuitGroup":
        # scalar * component
        return self.__mul__(other)
    
    def __imul__(self, other: Union[int, float]) -> "CircuitGroup":
        return self.__mul__(other)

    def __pow__(self, other: Union[int, float]) -> "CircuitGroup":
        """Repeat a component in parallel using `**`.

        Args:
            other (Union[int, float]): Number of repeats.

        Returns:
            CircuitGroup: A parallel-connected circuit group.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            group = R(1.0) ** 2
            group.connection # 'parallel'
            ```
        """
        # component ** scalar
        if isinstance(other, (int, float)):
            return parallel(*[circuit_deepcopy(self)() for _ in range(int(other))])
        return NotImplemented

    def __rpow__(self, other: Union[int, float]) -> "CircuitGroup":
        # scalar * component
        return self.__pow__(other)

    def __ipow__(self, other: Union[int, float]) -> "CircuitGroup":
        return self.__pow__(other)
    
    def structure(self) -> Tuple[Any, Any, Tuple[Any, ...]]:
        """Return a tuple describing circuit topology.

        Args:
            None

        Returns:
            Tuple[Any, Any, Tuple[Any, ...]]: A structural signature tuple.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1), R(2))
            group2 = series(R(3), R(4))
            group.structure == group2.structure # True
            ```
        """
        # Tuple structure used for comparing circuit topology.
        children = getattr(self, "subgroups", None)
        return (
            type(self),
            getattr(self, "connection", None),
            tuple(c.structure() for c in children) if children else (),
        )
    
    def copy_values(self, other: "CircuitComponent") -> None:
        """Copy only critical fields from another component.

        Args:
            other (CircuitComponent): Component to copy from.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Resistor
            a = Resistor(cond=1.0)
            b = Resistor(cond=2.0)
            a.copy_values(b)
            a.cond == b.cond # True
            ```
        """
        for field_ in self._critical_fields:
            if hasattr(self,field_) and hasattr(other,field_):
                setattr(self,field_,getattr(other,field_))
    
    def plot(self,
        fourth_quadrant: bool = True,
        show_IV_parameters: bool = True,
        title: str = "I-V Curve",
        show_solver_summary: bool = False,
    ) -> None:
        pass

    def show(self) -> None:
        pass

CircuitComponent._parent_pointer_class=CircuitComponent

# Helper to collapse nested CircuitGroups with the same connection type.
def flatten_connection(parts_list: Sequence["CircuitComponent"], connection: str) -> List["CircuitComponent"]:
    flat_list = []
    for part in parts_list:
        # do not flatten if it's Device
        if isinstance(part,CircuitGroup) and part.connection==connection and not getattr(part,"_is_atomic",False):
            flat_list.extend(part.subgroups)
        else:
            flat_list.append(part)
    return flat_list

# Normalize args and build a CircuitGroup with optional tiling and flattening.
def connect(
    *args: Union["CircuitComponent", Sequence["CircuitComponent"]],
    connection: str = "series",
    flatten_connection_: bool = False,
    **kwargs: Any,
) -> "CircuitGroup":
    """Build a CircuitGroup from components and sequences.

    Args:
        *args (Union[CircuitComponent, Sequence[CircuitComponent]]): Components
            or iterables of components.
        connection (str): "series" or "parallel" connection type.
        flatten_connection_ (bool): If True, collapse nested groups.
        **kwargs (Any): Forwarded to tiling and CircuitGroup init.

    Returns:
        CircuitGroup: The connected circuit group.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import connect, R
        group = connect(R(1), R(2), connection="series")
        group.connection # 'series'
        ```
    """
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
            try:
                del item._is_atomic
            except AttributeError:
                pass
        if not hasattr(item,"extent"):
            all_items_have_extent = False
    if all_items_have_extent:
        safe_kwargs = utilities.filter_kwargs(tile_elements, kwargs)
        tile_elements(flat_list, **safe_kwargs)
    safe_kwargs = utilities.filter_kwargs(CircuitGroup.__init__, kwargs)
    return CircuitGroup(subgroups=flat_list,connection=connection,**safe_kwargs)

# Convenience constructors for series/parallel connections.
def series(
    *args: Union["CircuitComponent", Sequence["CircuitComponent"]],
    flatten_connection_: bool = False,
    **kwargs: Any,
) -> "CircuitGroup":
    """Connect components in series.

    Args:
        *args (Union[CircuitComponent, Sequence[CircuitComponent]]): Components
            or iterables of components.
        flatten_connection_ (bool): If True, collapse nested groups.
        **kwargs (Any): Forwarded to `connect` and layout helpers.

    Returns:
        CircuitGroup: A series-connected circuit group.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import series, R
        group = series(R(1), R(2))
        group.connection # 'series'
        ```
    """
    kwargs.pop("connection", None)
    if "rows" not in kwargs:
        kwargs["rows"] = 1
    return connect(*args,connection="series",flatten_connection_=flatten_connection_,**kwargs)

def parallel(
    *args: Union["CircuitComponent", Sequence["CircuitComponent"]],
    flatten_connection_: bool = False,
    **kwargs: Any,
) -> "CircuitGroup":
    """Connect components in parallel.

    Args:
        *args (Union[CircuitComponent, Sequence[CircuitComponent]]): Components
            or iterables of components.
        flatten_connection_ (bool): If True, collapse nested groups.
        **kwargs (Any): Forwarded to `connect` and layout helpers.

    Returns:
        CircuitGroup: A parallel-connected circuit group.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import parallel, R
        group = parallel(R(1), R(2))
        group.connection # 'parallel'
        ```
    """
    kwargs.pop("connection", None)
    if "cols" not in kwargs:
        kwargs["cols"] = 1
    return connect(*args,connection="parallel",flatten_connection_=flatten_connection_,**kwargs)

# Leaf element interface: must implement IV behavior and drawing hooks.
class CircuitElement(CircuitComponent):
    """A circuit element class
    """
    def set_operating_point(self, V: Optional[float] = None, I: Optional[float] = None) -> None: # noqa: E741
        """Set the operating point for this element.

        Args:
            V (Optional[float]): Voltage to evaluate.
            I (Optional[float]): Current to evaluate.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Resistor
            r = Resistor(cond=2.0)
            r.set_operating_point(V=0.5)
            r.operating_point[1] == 1.0 # True
            ```
        """
        if V is not None:
            self.operating_point = [V, self.calc_I(V)]
        else:
            self.operating_point = [utilities.interp_(I,self.IV_I,self.IV_V),I]
    def get_value_text(self) -> str:
        pass
    def get_draw_func(self) -> Callable[..., Any]:
        pass
    def draw(
        self,
        ax: Optional[Any] = None,
        x: float = 0,
        y: float = 0,
        color: str = "black",
        fontsize: float = 6,
        linewidth: float = 1.5,
        display_value: bool = False,
    ) -> None:
        """Draw this element on a matplotlib axis.

        Args:
            ax (Optional[Any]): Matplotlib Axes to draw into, or None.
            x (float): X position of the element center.
            y (float): Y position of the element center.
            color (str): Line color for the symbol.
            display_value (bool): If True, show the value label.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import R
            R(10).draw(ax=ax)
            ```
        """
        if ax is None:
            return
        text = None
        if display_value:
            text = self.get_value_text()
        utilities.draw_symbol(self.get_draw_func(),ax=ax,x=x,y=y,color=color,text=text,fontsize=fontsize,linewidth=linewidth)
        if "pos_node" in self.aux:
            ax.text(x,y-0.5,str(self.aux["neg_node"]), va='center', fontsize=fontsize)
            ax.text(x,y+0.5,str(self.aux["pos_node"]), va='center', fontsize=fontsize)
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute current for a given voltage.

        Args:
            V (Union[float, np.ndarray]): Voltage value(s).

        Returns:
            Union[float, np.ndarray]: Current value(s).

        Raises:
            NotImplementedError: If not implemented in a subclass.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Resistor
            Resistor(cond=2.0).calc_I(0.5) # 1.0
            ```
        """
        raise NotImplementedError("Every child class of CircuitElement must have its own calc_I function")
    def get_V_range(self) -> np.ndarray:
        raise NotImplementedError("Every child class of CircuitElement must have its own get_V_range function")

# Ideal current source with optional temperature and irradiance scaling.
class CurrentSource(CircuitElement,_type_number=0):
    """Ideal current source with temperature and irradiance scaling.
    """
    _critical_fields = CircuitComponent._critical_fields + ("IL","refSuns","Suns","refIL","refT","T","temp_coeff")
    def __init__(
        self,
        IL: float,
        Suns: float = 1.0,
        temperature: float = 25,
        temp_coeff: float = 0.0,
        tag: Optional[str] = None,
    ) -> None:
        """Initialize a current source.

        Args:
            IL (float): Light-generated current at reference conditions.
            Suns (float): Relative irradiance multiplier.
            temperature (float): Temperature in Celsius.
            temp_coeff (float): Temperature coefficient for IL scaling.
            tag (Optional[str]): Optional identifier for this source.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import CurrentSource
            src = CurrentSource(IL=1.0, temperature=25)
            src.IL # 1.0
            ```
        """
        super().__init__(tag=tag)
        self.IL = IL
        self.refSuns = Suns
        self.Suns = Suns
        self.refIL = IL
        self.refT = temperature
        self.T = temperature
        self.temp_coeff = temp_coeff
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self,"Suns"):
            self.Suns = 1.0
        if not hasattr(self,"refSuns"):
            self.refSuns = 1.0
        if not hasattr(self,"refIL"):
            self.ref_IL = self.IL
        if not hasattr(self,"temp_coeff"):
            self.temp_coeff = 0
        if not hasattr(self,"temperature"):
            self.temperature = 25 
    def set_operating_point(self, V: Optional[float] = None, I: Optional[float] = None) -> None: # noqa: E741
        if I is not None:
            raise NotImplementedError
        super().set_operating_point(V,I)
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -self.IL if np.isscalar(V) else -self.IL*np.ones_like(V)
    def get_V_range(self) -> np.ndarray:
        return np.array([0.0])
    def set_IL(self, IL: float) -> None:
        """Update IL and invalidate IV caches.

        Args:
            IL (float): New light-generated current.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import CurrentSource
            src = CurrentSource(IL=1.0)
            src.set_IL(2.0)
            src.IL # 2.0
            ```
        """
        self.IL = IL
        self.null_IV()
    def changeTemperatureAndSuns(
        self,
        temperature: Optional[float] = None,
        Suns: Optional[float] = None,
    ) -> None:
        """Update temperature and irradiance, then recompute IL.

        Args:
            temperature (Optional[float]): New temperature in Celsius.
            Suns (Optional[float]): New irradiance multiplier.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import CurrentSource
            src = CurrentSource(IL=1.0, Suns=1.0)
            src.changeTemperatureAndSuns(Suns=2.0)
            src.Suns # 2.0
            ```
        """
        if Suns is not None:
            self.Suns = Suns
        if temperature is not None:
            self.T = temperature
        self.set_IL(self.Suns*(self.refIL / self.refSuns + self.temp_coeff * (self.T - self.refT)))

    def __str__(self) -> str:
        return "Current Source: IL = " + self.get_value_text()
    
    def get_value_text(self) -> str:
        return f"{self.IL:.4f} A"
    def get_draw_func(self) -> Callable[..., Any]:
        return utilities.draw_CC_symbol
    
def IL(*args, **kwargs):
    """
    This is a convenience alias for :class:`CurrentSource`, typically used
    to represent the light-generated current term in a PV circuit.

    Example:
        ```python
        IL = IL(5.0)
        cell = IL + diode
        ```
    """
    return CurrentSource(*args, **kwargs)

# Ohmic resistor represented by conductance (1/R).
class Resistor(CircuitElement,_type_number=1):
    """Ohmic resistor represented by conductance (1/R).
    """
    _critical_fields = CircuitComponent._critical_fields + ("cond",)
    def __init__(self, cond: float = 1.0, tag: Optional[str] = None) -> None:
        """Initialize a resistor.

        Args:
            cond (float): Conductance in 1/ohm.
            tag (Optional[str]): Optional identifier for this resistor.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Resistor
            r = Resistor(cond=1.0)
            r.cond # 1.0
            ```
        """
        super().__init__(tag=tag)
        self.cond = cond
    def set_cond(self, cond: float) -> None:
        """Update conductance and invalidate IV caches.

        Args:
            cond (float): New conductance in 1/ohm.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Resistor
            r = Resistor(cond=1.0)
            r.set_cond(2.0)
            r.cond # 2.0
            ```
        """
        self.cond = cond
        self.null_IV()
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return V*self.cond
    def get_V_range(self) -> np.ndarray:
        cond = self.cond
        step = 1e-3
        if (cond > 1):
            step /= cond
        return np.array([-step,step])
    def __str__(self) -> str:
        return "Resistor: R = " + self.get_value_text()
    def get_value_text(self) -> str:
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
    def get_draw_func(self) -> Callable[..., Any]:
        return utilities.draw_resistor_symbol
    
# simplified initializer
def R(R: float, tag: Optional[str] = None) -> Resistor:
    """Create a resistor by specifying resistance in ohms.

    Args:
        R (float): Resistance in ohms.
        tag (Optional[str]): Optional identifier for this resistor.

    Returns:
        Resistor: A resistor with conductance 1/R.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import R
        R(10).cond # 0.1
        ```
    """
    return Resistor(cond=1/R, tag=tag)

# Generic diode with temperature-dependent thermal voltage and shift.
class Diode(CircuitElement,_type_number=2):
    """Generic diode class
    """
    _critical_fields = CircuitComponent._critical_fields + ("I0","n","V_shift","VT","refI0","refT")
    max_I = DEFAULT_MAX_DIODE_CURRENT_DENSITY
    def __init__(
        self,
        I0: float = 1e-15,
        n: float = 1,
        V_shift: float = 0,
        max_I: Optional[float] = None,
        tag: Optional[str] = None,
        temperature: float = 25,
    ) -> None:
        """Initialize a diode.

        Args:
            I0 (float): Saturation current.
            n (float): Ideality factor.
            V_shift (float): Voltage shift (e.g., breakdown offset).
            max_I (Optional[float]): Maximum current density for IV ranges.
            tag (Optional[str]): Optional identifier for this diode.
            temperature (float): Temperature in Celsius.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Diode
            d = Diode(I0=1e-12, n=1.0)
            d.I0 # 1e-12
            ```
        """
        super().__init__(tag=tag)
        self.I0 = I0
        self.n = n
        self.V_shift = V_shift
        self.VT = utilities.get_VT(temperature)
        self.refI0 = I0
        self.refT = temperature
        if max_I is not None:
            self.max_I = max_I
    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self,"refI0"):
            self.refI0 = self.I0
        if not hasattr(self,"refT"):
            self.refT = 25
        self.VT = utilities.get_VT(self.refT)
    def set_I0(self, I0: float) -> None:
        """Update saturation current and invalidate IV caches.

        Args:
            I0 (float): New saturation current.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Diode
            d = Diode()
            d.set_I0(1e-12)
            d.I0 # 1e-12
            ```
        """
        self.I0 = I0
        self.null_IV()
    def get_V_range(self) -> np.ndarray:
        max_num_points = self.max_num_points
        if max_num_points is None:
            max_num_points = 100
        max_I = self.max_I
        if max_I is None:
            max_I = 0.2
        max_num_points_ = max_num_points*max_I/0.2
        Voc = self.estimate_Voc(max_I)
        V = [-1.1,-1.0,0]+list(Voc*np.log(np.arange(1,max_num_points_))/np.log(max_num_points_-1))
        V = np.array(V) + self.V_shift
        return V
    def estimate_Voc(self, max_I: float) -> float:
        Voc = 10
        if self.I0>0:
            Voc = self.n*self.VT*np.log(max_I/self.I0) 
        return Voc
    def changeTemperature(self, temperature: float) -> None:
        """Update temperature and scale saturation current.

        Args:
            temperature (float): Temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Diode
            d = Diode()
            d.changeTemperature(35)
            ```
        """
        self.VT = utilities.get_VT(temperature)
        old_ni  = silicon.get_ni(self.refT)
        new_ni  = silicon.get_ni(temperature)
        scale_factor = (new_ni/old_ni)**(2/self.n)
        self.set_I0(self.refI0*scale_factor)
    
# Standard Shockley diode (forward direction).
class ForwardDiode(Diode):
    """Diode that points from '+' to '-'
    """
    def __str__(self) -> str:
        return "Forward Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n)
    def get_value_text(self) -> str:
        word = f"I0 = {self.I0:.3e}"
        if "error" in self.aux and not np.isnan(self.aux["error"]):
            word += f"\n\u00B1{self.aux['error']:.3e}"
        word += f" A\nn = {self.n:.2f}"
        return word
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.I0*(np.exp((V-self.V_shift)/(self.n*self.VT))-1)
    def get_draw_func(self) -> Callable[..., Any]:
        return utilities.draw_forward_diode_symbol
    
# simplified initializer
def D(*args, **kwargs):
    """
    This is a shorthand constructor for :class:`ForwardDiode`.
    """
    return ForwardDiode(*args, **kwargs)


def D1(*args, **kwargs):
    """
    Create a forward diode with ideality factor n = 1.
    """
    n = kwargs.pop("n", 1)
    return ForwardDiode(*args, n=n, **kwargs)


def D2(*args, **kwargs):
    """
    Create a forward diode with ideality factor n = 2.
    """
    n = kwargs.pop("n", 2)
    return ForwardDiode(*args, n=n, **kwargs)
    
# LED-like diode used for photon coupling in tandem models.
class PhotonCouplingDiode(ForwardDiode):
    """LED-like diode for photon coupling in tandem models.
    """
    def get_draw_func(self) -> Callable[..., Any]:
        return utilities.draw_LED_diode_symbol
    def __str__(self) -> str:
        return "Photon Coupling Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n)

# simplified initializer
def Dpc(*args, **kwargs):
    """
    This is a shorthand constructor for :class:`PhotonCouplingDiode`.
    """
    return PhotonCouplingDiode(*args, **kwargs)

# Reverse (breakdown) diode; shifts voltage and flips IV direction.
class ReverseDiode(Diode,_type_number=3):
    """Reverse (breakdown) diode with inverted IV direction.
    """
    def __str__(self) -> str:
        return "Reverse Diode: I0 = " + str(self.I0) + "A, n = " + str(self.n) + ", breakdown V = " + str(self.V_shift)
    def get_value_text(self) -> str:
        return f"I0 = {self.I0:.3e}A\nn = {self.n:.2f}\nbreakdown V = {self.V_shift:.2f}"
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return -self.I0*np.exp((-V-self.V_shift)/(self.n*self.VT))
    def get_V_range(self) -> np.ndarray:
        V_range = super().get_V_range()
        return -V_range[::-1]
    def get_draw_func(self) -> Callable[..., Any]:
        return utilities.draw_reverse_diode_symbol

# simplified initializer
def Drev(*args, **kwargs):
    """
    This is a shorthand constructor for :class:`ReverseDiode`.
    """
    return ReverseDiode(*args, **kwargs)
    
# Intrinsic Si diode with recombination physics for base region.
class Intrinsic_Si_diode(ForwardDiode,_type_number=4):
    """Intrinsic silicon diode with base recombination physics.
    """
    _critical_fields = CircuitComponent._critical_fields + ("base_thickness","base_type","base_doping","temperature","VT","ni","area")
    bandgap_narrowing_RT = np.array(silicon.bandgap_narrowing_RT)
    # area is 1 is OK because the cell subgroup has normalized area of 1
    def __init__(
        self,
        base_thickness: float = 180e-4,
        base_type: str = "n",
        base_doping: float = 1e+15,
        area: float = 1.0,
        temperature: float = 25,
        max_I: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Initialize an intrinsic silicon diode.

        Args:
            base_thickness (float): Base thickness in cm.
            base_type (str): "p" or "n" base type.
            base_doping (float): Base doping concentration in cm^-3.
            area (float): Active area multiplier.
            temperature (float): Temperature in Celsius.
            max_I (Optional[float]): Maximum current density for IV ranges.
            tag (Optional[str]): Optional identifier.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import Intrinsic_Si_diode
            d = Intrinsic_Si_diode()
            d.base_type # 'n'
            ```
        """
        CircuitElement.__init__(self, tag)
        self.base_thickness = base_thickness
        self.base_type = base_type
        self.base_doping = base_doping
        self.temperature = temperature
        self.area = area
        self.V_shift = 0
        self.VT = utilities.get_VT(self.temperature)
        self.ni = silicon.get_ni(self.temperature)
        if max_I is not None:
            self.max_I = max_I
    def __post_init__(self):
        CircuitElement.__post_init__(self)
        if not hasattr(self,"base_thickness"):
            self.base_thickness = 180e-4
        if not hasattr(self,"base_type"):
            self.base_type = "n"
        if not hasattr(self,"base_doping"):
            self.base_doping = 1e15
        if not hasattr(self,"area"):
            self.area = 1
        if not hasattr(self,"temperature"):
            self.temperature = 25
        self.VT = utilities.get_VT(self.temperature)
        self.ni = silicon.get_ni(self.temperature)
    def __str__(self) -> str:
        return "Si Intrinsic Diode"
    def calc_I(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
        BGN = utilities.interp_(delta_n,self.bandgap_narrowing_RT[:,0],self.bandgap_narrowing_RT[:,1])
        ni_eff = ni*np.exp(BGN/2/VT)
        geeh = 1 + 13*(1-np.tanh((n0/3.3e17)**0.66))
        gehh = 1 + 7.5*(1-np.tanh((p0/7e17)**0.63))
        Brel = 1
        Blow = 4.73e-15
        intrinsic_recomb = (pn - ni_eff**2)*(2.5e-31*geeh*n0+8.5e-32*gehh*p0+3e-29*delta_n**0.92+Brel*Blow) # in units of 1/s/cm3
        return utilities.q*intrinsic_recomb*self.base_thickness*self.area
    def estimate_Voc(self, max_I: float) -> float:
        Voc = 10
        if self.base_thickness>0:
            Voc = 0.7
            for _ in range(10):
                I_ = self.calc_I(Voc)
                if I_ >= max_I and I_ <= max_I*1.1:
                    break
                Voc += self.VT*np.log(max_I/I_)
        return Voc
    def get_value_text(self) -> str:
        word = f"intrinsic:\nt={self.base_thickness:.2e}\n{self.base_type} type\n{self.base_doping:.2e} cm-3"
        return word
    def set_I0(self, I0: float) -> None:
        pass # does nothing
    def changeTemperature(self, temperature: float) -> None:
        self.temperature = temperature
        self.VT = utilities.get_VT(self.temperature)
        self.ni = silicon.get_ni(self.temperature)
        self.null_IV()

# simplified initializer
def Dintrinsic_Si(*args, **kwargs):
    """
    This is a shorthand constructor for :class:`Intrinsic_Si_diode`.
    """
    return Intrinsic_Si_diode(*args, **kwargs)

# Composite circuit node that holds subgroups in series or parallel.
class CircuitGroup(CircuitComponent,_type_number=5):
    """Circuit with series or parallel connected parts.
    """
    _critical_fields = CircuitComponent._critical_fields + ("connection","subgroups")
    def __init__(
        self,
        subgroups: Sequence[CircuitComponent],
        connection: str = "series",
        name: Optional[str] = None,
        location: Optional[Sequence[float]] = None,
        rotation: float = 0,
        x_mirror: int = 1,
        y_mirror: int = 1,
        extent: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize a CircuitGroup.

        Args:
            subgroups (Sequence[CircuitComponent]): Child components/groups.
            connection (str): "series" or "parallel" connection type.
            name (Optional[str]): Optional name for lookup.
            location (Optional[Sequence[float]]): XY location for layout.
            rotation (float): Rotation in degrees.
            x_mirror (int): X mirror flag (+1 or -1).
            y_mirror (int): Y mirror flag (+1 or -1).
            extent (Optional[Sequence[float]]): Precomputed extent (width, height).

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import CircuitGroup, R
            group = CircuitGroup([R(1), R(2)], connection="series")
            group.connection # 'series' 
            ```
        """
        super().__init__()
        self.connection = connection
        self.subgroups = subgroups
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
        CircuitGroup.__compile__(self)

    def __post_init__(self):
        super().__post_init__()
        CircuitGroup.__compile__(self)

    def __compile__(self):
        self.circuit_diagram_extent = get_circuit_diagram_extent(self.subgroups,self.connection)
        self.num_circuit_elements = 0
        max_I = 0
        max_max_I = 0
        for element in self.subgroups:
            element.parent = self
            self.num_circuit_elements += element.num_circuit_elements
            self.circuit_depth = max(self.circuit_depth,element.circuit_depth+1)
            if element.max_I is not None:
                max_max_I = max(max_max_I, element.max_I)
                if self.connection=="series":
                    max_I = max(max_I, element.max_I)
                else:
                    max_I += element.max_I
        if max_I > 0:
            self.max_I = max_I
        if max_max_I > 0:
            for element in self.subgroups:
                if isinstance(element,Diode) and element.max_I is None:
                    element.max_I = max_max_I
        if self.num_circuit_elements > REMESH_NUM_ELEMENTS_THRESHOLD:
            self.max_num_points = int(REMESH_POINTS_DENSITY*np.sqrt(self.num_circuit_elements))

    @property
    def parts(self) -> Sequence["CircuitComponent"]:
        """Alias for subgroups.

        Args:
            None

        Returns:
            Sequence[CircuitComponent]: Child components/groups.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1), R(2))
            len(group.parts) # 2
            ```
        """
        return self.subgroups

    @parts.setter
    def parts(self, value: Sequence["CircuitComponent"]) -> None:
        self.subgroups = value

    @property
    def children(self) -> Sequence["CircuitComponent"]:
        """Another alias for subgroups.

        Args:
            None

        Returns:
            Sequence[CircuitComponent]: Child components/groups.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1), R(2))
            len(group.children) # 2
            ```
        """
        return self.subgroups

    @children.setter
    def children(self, value: Sequence["CircuitComponent"]) -> None:
        self.subgroups = value

    def set_operating_point(
        self,
        V: Optional[float] = None,
        I: Optional[float] = None, # noqa: E741
        refine_IV: bool = False,
        shallow: bool = False,
        use_python: bool = False,
    ) -> None: 
        """Set the operating point for the circuit group.

        Args:
            V (Optional[float]): Voltage to set or evaluate.
            I (Optional[float]): Current to set or evaluate.
            refine_IV (bool): If True, refine IV around operating point.
            shallow (bool): If True, skip recursion into subgroups.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import *
            group = R(1) + R(2)
            group.build_IV()
            group.set_operating_point(V=0.5, shallow=True)
            group.operating_point[0] # 0.5
            ```
        """
        if not hasattr(self,"job_heap"):
            self.build_IV()
        elif self.IV_V is None:
            self.job_heap.run_IV()

        # Auto-range IV curve if requested point falls outside current bounds.
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

        use_python_ = use_python
        pc_diodes = []
        if not use_python:
            pc_diodes = self.findElementType(PhotonCouplingDiode)
            if len(pc_diodes)>0:
                use_python_ = True
        if use_python_ or shallow or self.num_circuit_elements < 10000: # no need to go c++ overkill
            # Use python interpolation for shallow/small circuits.
            V_ = V
            I_ = I
            if V is not None:
                I_ = np.interp(V,self.IV_V,self.IV_I)
            if I is not None:
                V_ = np.interp(I,self.IV_I,self.IV_V)
            self.operating_point = [V_,I_]
            if type(self).__name__=="Cell": # cell
                I_ /= self.area
            if shallow:   
                return
            I_offset_ = 0
            for item in reversed(self.subgroups):
                if isinstance(item,CircuitElement):
                    if self.connection=="series":
                        item.set_operating_point(I=I_)
                    else:
                        item.set_operating_point(V=V_)
                else:
                    if self.connection=="series":
                        item.set_operating_point(I=I_-I_offset_,use_python=True)
                        I_offset_ = 0
                        if len(pc_diodes)>0 and type(item).__name__=="Cell":
                            photon_coupling_diodes = item.photon_coupling_diodes
                            if photon_coupling_diodes and len(photon_coupling_diodes)>0:
                                I_offset_ = -photon_coupling_diodes[0].operating_point[1]*item.area
                    else:
                        item.set_operating_point(V=V_,use_python=True)
        else:
            self.job_heap.set_operating_point(V,I)

        gc.disable()
        if refine_IV:
            self.job_heap.refine_IV()
        self.refined_IV = True
        gc.enable()

    def removeElementOfTag(self, tag: Any) -> None:
        """Remove all elements with a matching tag.

        Args:
            tag (Any): Tag value to remove.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R
            group = series(R(1, tag="R1"), R(2, tag="R2"))
            group.removeElementOfTag("R1")
            len(group.subgroups) # 1
            ```
        """
        for j in range(len(self.subgroups) - 1, -1, -1):
            element = self.subgroups[j]
            if isinstance(element, CircuitElement):
                if element.tag == tag:
                    self.subgroups.pop(j)
            elif isinstance(element, CircuitGroup):
                element.removeElementOfTag(tag)
        self.null_IV()

    def set_temperature(self, temperature: float) -> None:
        """Update temperature for all diodes and current sources.

        Args:
            temperature (float): Temperature in Celsius.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R, CurrentSource
            group = series(CurrentSource(1.0), R(1))
            group.set_temperature(35)
            ```
        """
        diodes = self.findElementType(Diode)
        for diode in diodes:
            diode.changeTemperature(temperature)
        currentSources = self.findElementType(CurrentSource)
        for currentSource in currentSources:
            currentSource.changeTemperatureAndSuns(temperature=temperature)

    def findElementType(self, type_: Union[type, str], stop_at_type_: Union[type, str] | None = None) -> List["CircuitComponent"]:
        """Find all elements of a given type in the subtree.

        Args:
            type_ (Union[type, str]): Class or class name to match.
            stop_at_type_ (Union[type, str]): if encounter element of stop_at_type_, do not search within it

        Returns:
            List[CircuitComponent]: Matching elements.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R, Diode
            group = series(R(1), Diode())
            len(group.findElementType(Diode)) # 1
            ```
        """
        list_ = []
        for element in self.subgroups:
            if (not isinstance(type_,str) and isinstance(element,type_))  or (isinstance(type_,str) and type(element).__name__==type_):
                list_.append(element)
            elif stop_at_type_ is not None and ((not isinstance(stop_at_type_,str) and isinstance(element,stop_at_type_))  or (isinstance(stop_at_type_,str) and type(element).__name__==stop_at_type_)):
                pass
            elif isinstance(element,CircuitGroup):
                list_.extend(element.findElementType(type_,stop_at_type_))
        return list_
    
    def __getitem__(self, type_: Union[type, str]) -> List["CircuitComponent"]:
        """Shortcut for `findElementType`.

        Example:
            ```python
            from PV_Circuit_Model.circuit_model import series, R, Diode
            group = series(R(1), Diode())
            len(group[Diode])
            ```
        """
        return self.findElementType(type_)
    
    def __str__(self) -> Optional[str]:
        if self.num_circuit_elements > 2000:
            print(f"There are too many elements to draw ({self.num_circuit_elements}).  I give up!")
            return
        word = self.connection + " connection:\n"
        for i, element in enumerate(self.subgroups):
            if isinstance(element,CircuitGroup):
                word += "Subgroup " + str(i) + ":\n"
            word += str(element) + "\n"
        return word  
    
    def draw(
        self,
        x: float = 0,
        y: float = 0,
        display_value: bool = False,
        title: str = "Model",
        linewidth: float = 1.5,
        animate: bool = False,
        V_sweep_frames: Optional[Any] = None,
        split_screen_with_IV: bool = False,
        current_flow_log10_range: float = 4,
        area_multiplier: float = 1,
        fontsize: float = 6,
        ax2_width: float = 0.5,
        ax2_xlim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        operating_point_size: float = 20,
        max_electron_size: float = 0.05,
    ) -> None:
        """Draw the CircuitGroup as a schematic, optionally with animation and an IV subplot.

        If ``animate=True``, the method animates "current flow" using moving circles along the wires.
        If ``V_sweep_frames`` is provided, the operating point is swept over the specified voltages and
        the animation updates the operating point (and optionally the IV subplot markers) per frame.

        Args:
            x: X position of the schematic center in diagram coordinates.
            y: Y position of the schematic center in diagram coordinates.
            display_value: If True, display component values (where supported by elements).
            title: Figure window title when creating a new figure.
            linewidth: Line width for wiring and plots
            animate: If True, animate current flow using moving circles on the wires. Requires that
                operating points have been computed (e.g., via ``get_Pmax()``, ``set_operating_point()``,
                or similar).
            V_sweep_frames: Optional sequence of voltages to sweep. If provided, each animation frame
                sets the circuit operating point to ``V_sweep_frames[frame]``. When omitted, the
                animation runs as a steady "flow" at the current operating point.
            split_screen_with_IV: If True, creates a second subplot showing the IV curve 
            current_flow_log10_range: Controls visual scaling of circle size with current magnitude.
                Circle radii scale approximately with ``log10(|I|)`` over this range; larger values
                compress the visual range (less variation), smaller values exaggerate it.
            area_multiplier: Not passed by users.
            fontsize: font size to display
            ax2_width: width of ax2 (max 1)
            ax2_xlim: xlim of ax2 (dafault None)
            ax2_ylim: ylim of ax2 (dafault None)
            figsize: figure width and height (default None)
            operating_point_size: size of operating point to plot (default 20)
            max_electron_size: size of electrons in animation (default 0.05)

        Example:
            Basic schematic:

            ```python
            from PV_Circuit_Model.circuit_model import series, R
            (R(1) + R(2)).draw()
            ```

            Animate current flow (after computing an operating point):

            ```python
            circuit.get_Pmax()
            circuit.draw(animate=True)
            ```

            Sweep operating point across voltages and show IV subplot:

            ```python
            Vs = np.linspace(0, circuit.get_Voc(), 60)
            circuit.get_Pmax()
            circuit.draw(animate=True, V_sweep_frames=Vs, split_screen_with_IV=True)
            ```
        """
        self._draw_internal(
        x=x,
        y=y,
        display_value=display_value,
        title=title,
        linewidth=linewidth,
        animate=animate,
        V_sweep_frames=V_sweep_frames,
        split_screen_with_IV=split_screen_with_IV,
        current_flow_log10_range=current_flow_log10_range,
        area_multiplier=area_multiplier,
        fontsize=fontsize,
        ax2_width=ax2_width,
        ax2_xlim=ax2_xlim,
        ax2_ylim=ax2_ylim,
        figsize=figsize,
        operating_point_size=operating_point_size,
        max_electron_size=max_electron_size,
        is_root= True)

    def _draw_internal(
        self,
        ax: Optional[Any] = None,
        ax2: Optional[Any] = None,
        x: float = 0,
        y: float = 0,
        display_value: bool = False,
        title: str = "Model",
        linewidth: float = 1.5,
        animate: bool = False,
        V_sweep_frames: Optional[Any] = None,
        split_screen_with_IV: bool = False,
        current_flow_log10_range: float = 4,
        area_multiplier: float = 1,
        fontsize: float = 6,
        ax2_width: float = 0.5,
        ax2_xlim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        operating_point_size: float = 20,
        max_electron_size: float = 0.05,*,
        is_root: bool = True
    ) -> list[Any] | None:
        
        if self.num_circuit_elements > 2000:
            print(f"There are too many elements to draw ({self.num_circuit_elements}).  I give up!")
            return
        
        global pbar
        ax2_pts = []
        if is_root:
            # Create a fresh figure and progress bar for interactive drawing.
            num_of_elements = self.num_circuit_elements
            pbar = tqdm(total=num_of_elements)
            if V_sweep_frames is not None:
                self.set_operating_point(V=V_sweep_frames[0])
            if split_screen_with_IV:
                if self.IV_V is None:
                    self.get_Pmax()
                if figsize is not None:
                    fig, (ax,ax2) = plt.subplots(2,1,figsize=figsize) 
                else:
                    fig, (ax,ax2) = plt.subplots(2,1) 
                color_scheme = ["red","green","blue","purple"]
                Voc = self.get_Voc()
                Isc = self.get_Isc()
                if self.operating_point is not None:
                    ax2_pts.append(ax2.scatter([self.operating_point[0]], [-self.operating_point[1]], s=operating_point_size,color="black"))
                cells = self.findElementType("Cell")
                ax2.plot(self.IV_V,-self.IV_I,color="black",linewidth=linewidth)
                for j, cell in enumerate(cells):
                    number = j % len(color_scheme)
                    if cell.operating_point is not None:
                        ax2_pts.append(ax2.scatter([cell.operating_point[0]], [-cell.operating_point[1]], s=operating_point_size,color=color_scheme[number]))
                    ax2.plot(cell.IV_V,-cell.IV_I,color=color_scheme[number],linewidth=linewidth)
                    Voc = max(Voc,cell.get_Voc())
                    Isc = max(Isc,cell.get_Isc())
                if ax2_xlim is None:
                    ax2.set_xlim((0,Voc*1.1))
                else:
                    ax2.set_xlim(ax2_xlim)
                if ax2_ylim is None:
                    ax2.set_ylim((0,Isc*1.1))
                else:
                    ax2.set_ylim(ax2_ylim)
                ax2.set_xlabel("V (V)")
                ax2.set_ylabel("I (A)")
                ax2.tick_params(axis="both", labelsize=fontsize)
                ax2.xaxis.label.set_size(fontsize)
                ax2.yaxis.label.set_size(fontsize)
            else:
                fig, ax = plt.subplots() 
        
        if "current_offset" in self.aux:
            current_offset = self.aux["current_offset"]
            if type(self).__name__=="Cell":
                current_offset /= self.area
            if self.connection=="series":
                for element in self.subgroups:
                    element.aux["current_offset"] = current_offset
            else:
                current_source = self.findElementType(CurrentSource)
                current_source[0].aux["current_offset"] = current_offset

        def update(frame):
            global circles
            global max_I_
            if frame >= 0 and V_sweep_frames is None:
                for circle_info in circles:
                    circle = circle_info[0]
                    pos = circle_info[1]
                    distance = circle_info[2]
                    speed = circle_info[3]
                    start_pt = circle_info[4]
                    unit_vector = circle_info[5]
                    balls_end_to_end_distance = circle_info[6]
                    new_pos = speed + pos
                    if new_pos > distance:
                        new_pos -= balls_end_to_end_distance
                    x_ = start_pt[0] + unit_vector[0]*new_pos + 0.02
                    y_ = start_pt[1] + unit_vector[1]*new_pos + 0.02
                    circle_info[1] = new_pos
                    circle.center = (x_,y_)
                    if new_pos < 0:
                        circle.set_visible(False)
                    else:
                        circle.set_visible(True)
                return [ci[0] for ci in circles] 

            branch_currents = []
            area_ = area_multiplier
            current_x = x - self.circuit_diagram_extent[0]/2
            current_y = y - self.circuit_diagram_extent[1]/2
            if self.connection != "series":
                current_y += 0.1
            if type(self).__name__=="Cell":
                area_ *= self.area
            cumulative_I = 0

            ax_ = None
            if frame==-1:
                ax_ = ax
            if V_sweep_frames is not None:
                frame_ = max(frame,0)
                self.set_operating_point(V=V_sweep_frames[frame_])
                if split_screen_with_IV:
                    ax2_pts[0].set_offsets([[self.operating_point[0], -self.operating_point[1]]])
                    for j, cell in enumerate(cells):
                        ax2_pts[j+1].set_offsets([[cell.operating_point[0], -cell.operating_point[1]]])

            if animate and self.operating_point is not None:
                for i_, element in enumerate(reversed(self.subgroups)):
                    i = len(self.subgroups)-i_-1
                    if type(element).__name__=="Cell":
                        pc_diodes = element.photon_coupling_diodes
                        if len(pc_diodes) > 0:
                            if i > 0:
                                current_offset = -pc_diodes[0].operating_point[1]*element.area
                                cell_below = self.subgroups[i-1]
                                if type(cell_below).__name__=="Cell":
                                    cell_below.aux["current_offset"] = current_offset
                                    current_source = cell_below.findElementType(CurrentSource)
                                    if len(current_source)>0:
                                        current_source[0].aux["pc_partner"] = pc_diodes[0]
                                        pc_diodes[0].aux["pc_partner"] = current_source[0]

            for i, element in enumerate(self.subgroups):
                if isinstance(element,CircuitElement) and frame==-1:
                    pbar.update(1)
                center_x = current_x+element.circuit_diagram_extent[0]/2
                center_y = current_y+element.circuit_diagram_extent[1]/2
                if self.connection == "series":
                    center_x = x
                else:
                    center_y = y
                if isinstance(element,CircuitElement):
                    element.draw(ax=ax_, x=center_x, y=center_y, display_value=display_value, fontsize=fontsize, linewidth=linewidth)
                else:
                    branch_currents.extend(element._draw_internal(ax=ax_, x=center_x, y=center_y, display_value=display_value,animate=animate,area_multiplier=area_,is_root=False,fontsize=fontsize,linewidth=linewidth))
                I_ = None
                if animate and element.operating_point is not None:
                    I_ = element.operating_point[1]*area_
                    if "current_offset" in element.aux:
                        I_ += element.aux["current_offset"]*area_
                    if "pc_partner" in element.aux:
                        branch_currents.append([element,element.aux["pc_partner"],center_x,center_y,I_])
                if self.connection=="series":
                    if i > 0:
                        if I_ is not None:
                            if isinstance(element,CircuitElement):
                                branch_currents.append([x,x,current_y-utilities.y_spacing, current_y+element.circuit_diagram_extent[1],I_])
                            else:
                                branch_currents.append([x,x,current_y-utilities.y_spacing, current_y,I_])
                        if ax_ is not None:
                            line = plt.Line2D([x,x],[current_y-utilities.y_spacing, current_y], color="black", linewidth=linewidth)
                            ax.add_line(line)
                    current_y += element.circuit_diagram_extent[1]+utilities.y_spacing
                else:
                    if I_ is not None:
                        if isinstance(element,CircuitElement):
                            branch_currents.append([center_x,center_x,y-self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2,I_])
                        else:
                            branch_currents.append([center_x,center_x,center_y+element.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2,I_])
                            branch_currents.append([center_x,center_x,center_y-element.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2,I_])
                        if i < len(self.subgroups)-1:
                            next_center_x = current_x+element.circuit_diagram_extent[0]+utilities.x_spacing
                            cumulative_I += I_ 
                            if center_x <= x:
                                branch_currents.append([center_x,min(next_center_x,x),
                                                    y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2,cumulative_I])
                                branch_currents.append([center_x,min(next_center_x,x),
                                                        y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2,-cumulative_I])
                            else:    
                                branch_currents.append([center_x,next_center_x,
                                                    y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2,cumulative_I])
                                branch_currents.append([center_x,next_center_x,
                                                        y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2,-cumulative_I])
                            if center_x <= x and next_center_x > x:
                                cumulative_I = 0
                                for j in range(i+1,len(self.subgroups)):
                                    cumulative_I -= self.subgroups[j].operating_point[1]*area_
                                branch_currents.append([x,next_center_x,
                                                    y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2,cumulative_I])
                                branch_currents.append([x,next_center_x,
                                                        y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2,-cumulative_I])

                    if ax_ is not None:
                        line = plt.Line2D([center_x,center_x], [center_y+element.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                        ax.add_line(line)
                        line = plt.Line2D([center_x,center_x], [center_y-element.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                        ax.add_line(line)
                        if i > 0:
                            line = plt.Line2D([center_x,current_x-utilities.x_spacing-self.subgroups[i-1].circuit_diagram_extent[0]/2], [y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                            ax.add_line(line)
                            line = plt.Line2D([center_x,current_x-utilities.x_spacing-self.subgroups[i-1].circuit_diagram_extent[0]/2], [y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2], color="black", linewidth=linewidth)
                            ax.add_line(line)
                    current_x += element.circuit_diagram_extent[0]+utilities.x_spacing

            if not is_root:
                return branch_currents
            
            if animate and self.operating_point is not None:
                I_ = self.operating_point[1]*area_
                if "current_offset" in self.aux:
                    I_ += self.aux["current_offset"]*area_
                branch_currents.append([x,x,y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2-0.2,-I_])
                branch_currents.append([x,x,y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2+0.2,I_])

            if frame==-1:
                pbar.close()
                # Add terminals and clean up axes for diagram output.
                line = plt.Line2D([x,x], [y-self.circuit_diagram_extent[1]/2,y-self.circuit_diagram_extent[1]/2-0.2], color="black", linewidth=linewidth)
                ax.add_line(line)
                line = plt.Line2D([x,x], [y+self.circuit_diagram_extent[1]/2,y+self.circuit_diagram_extent[1]/2+0.2], color="black", linewidth=linewidth)
                ax.add_line(line)
                utilities.draw_symbol(utilities.draw_earth_symbol, ax=ax,  x=x, y=y-self.circuit_diagram_extent[1]/2-0.3)
                utilities.draw_symbol(utilities.draw_pos_terminal_symbol, ax=ax,  x=x, y=y+self.circuit_diagram_extent[1]/2+0.25)
                ax.set_aspect('equal')
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.tight_layout()
                if split_screen_with_IV:
                    bbox = ax2.get_position()
                    new_width = bbox.width * ax2_width
                    new_left = bbox.x0 + (bbox.width - new_width) / 2
                    ax2.set_position([new_left, bbox.y0, new_width, bbox.height])
                fig.canvas.manager.set_window_title(title)

            
            ball_spacing = 0.12/0.05*max_electron_size
            counter = 0
            if frame==-1:
                circles = []
                max_I_ = 0
                for branch_current in branch_currents:
                    max_I_ = max(max_I_,abs(branch_current[4]))
            for branch_current in branch_currents:
                color = "blue"
                if isinstance(branch_current[0],CircuitComponent): # a pc connection
                    if isinstance(branch_current[0],PhotonCouplingDiode):
                        I_ = branch_current[4]
                        start_pt = np.array([branch_current[2],branch_current[3]])
                        for branch_current_ in branch_currents:
                            if branch_current_[1] is branch_current[0]: 
                                end_pt = np.array([branch_current_[2],branch_current_[3]])
                        color = "orange"
                    else:
                        continue
                else:
                    I_ = branch_current[4]
                    if I_ < 0:
                        start_pt = np.array([branch_current[0],branch_current[2]])
                        end_pt = np.array([branch_current[1],branch_current[3]])
                    else:
                        end_pt = np.array([branch_current[0],branch_current[2]])
                        start_pt = np.array([branch_current[1],branch_current[3]])
                ratio = abs(I_)/max_I_

                log_ratio = np.log10(ratio)
                ball_size = max_electron_size*(current_flow_log10_range+log_ratio)/current_flow_log10_range
                ball_size = max(0,ball_size)
                displacement = end_pt-start_pt
                distance = np.sqrt(displacement[0]**2+displacement[1]**2)
                unit_vector = displacement / distance
                range_ = np.arange(0,distance,ball_spacing)
                num_balls = len(range_)
                balls_end_to_end_distance = num_balls*ball_spacing
                for pos in range_:
                    x_ = start_pt[0] + unit_vector[0]*pos + 0.02
                    y_ = start_pt[1] + unit_vector[1]*pos + 0.02
                    if frame==-1:
                        circles.append([Circle((x_,y_), radius=ball_size, fc=color), pos, distance, ball_spacing/4, start_pt, unit_vector, balls_end_to_end_distance, ball_size])
                        ax.add_patch(circles[-1][0])
                    else:
                        circles[counter][7] = ball_size
                        circles[counter][4] = start_pt
                        circles[counter][5] = unit_vector
                        counter += 1
            for circle_info in circles:
                circle = circle_info[0]
                pos = circle_info[1]
                distance = circle_info[2]
                speed = circle_info[3]
                start_pt = circle_info[4]
                unit_vector = circle_info[5]
                balls_end_to_end_distance = circle_info[6]
                new_radius = circle_info[7]
                new_pos = speed + pos
                if new_pos > distance:
                    new_pos -= balls_end_to_end_distance
                x_ = start_pt[0] + unit_vector[0]*new_pos + 0.02
                y_ = start_pt[1] + unit_vector[1]*new_pos + 0.02
                circle_info[1] = new_pos
                circle.center = (x_,y_)
                circle.set_radius(new_radius)
                if new_pos < 0:
                    circle.set_visible(False)
                else:
                    circle.set_visible(True)
            return [ci[0] for ci in circles] + [ax2_pt for ax2_pt in ax2_pts] # needed for blit=True

        branch_currents = update(-1)
        if "current_offset" in self.aux:
            del self.aux["current_offset"]
        if "pc_partner" in self.aux:
            del self.aux["pc_partner"]
        if not is_root:
            return branch_currents

        frames_ = 20
        if V_sweep_frames is not None:
            frames_ = len(V_sweep_frames)
        if animate and self.operating_point is not None:
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=frames_,
                interval=50,
                blit=True,
                cache_frame_data=False
            )
            if IN_NOTEBOOK:
                display(HTML(ani.to_jshtml()))
                               
        plt.show()

    
    def as_type(self, cls: Type[T_CircuitComponent], **kwargs: Any) -> T_CircuitComponent:
        """Convert this group to another circuit component type.

        Args:
            cls (Type[CircuitComponent]): Target class with `from_circuitgroup`.
            **kwargs (Any): Forwarded to `from_circuitgroup`.

        Returns:
            CircuitComponent: Converted instance of `cls`.

        Raises:
            TypeError: If `cls` is not a CircuitComponent or lacks conversion.

        Example:
            ```python
            from PV_Circuit_Model.device import *
            circuit_group = ( 
                (IL(41e-3) | D1(10e-15) | D2(5e-9) | Dintrinsic_Si(180e-4) | Drev(V_shift=10) | R(1e5)) 
                + R(1/3)
            )
            cell_ = circuit_group.as_type(Cell, **wafer_shape(format="M10",half_cut=True))
            ```
        """
        if not issubclass(cls, CircuitComponent):
            raise TypeError(...)
        if hasattr(cls, "from_circuitgroup"):
            return cls.from_circuitgroup(self, **kwargs)
        raise TypeError(f"{cls.__name__} does not support conversion")
    
    def tile_subgroups(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        x_gap: float = 0.0,
        y_gap: float = 0.0,
        turn: bool = False,
        xflip: bool = False,
        yflip: bool = False,
        col_wise_ordering: bool = True,
    ) -> "CircuitGroup":
        """Tile subgroups into a grid layout.

        Args:
            rows (Optional[int]): Number of rows.
            cols (Optional[int]): Number of columns.
            x_gap (float): Horizontal gap between tiles.
            y_gap (float): Vertical gap between tiles.
            turn (bool): Alternate rotation per column/row.
            xflip (bool): Flip X orientation for 2-column layout.
            yflip (bool): Flip Y orientation for 2-row layout.
            col_wise_ordering (bool): Fill columns before rows.

        Returns:
            CircuitGroup: The updated group (self).

        Example:
            ```python
            from PV_Circuit_Model.device import Cell_
            cell = Cell_()
            cells = cell*5
            cells.tile_subgroups(rows=1)
            ```
        """
        tile_elements(self.subgroups, rows=rows, cols=cols, x_gap = x_gap, y_gap = y_gap, turn=turn, xflip=xflip, yflip=yflip, col_wise_ordering=col_wise_ordering)
        self.extent = get_extent(self.subgroups)
        return self

def get_extent(elements: Sequence[Any], center: bool = True) -> Optional[List[float]]:
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

# Compute drawing extent based on series/parallel layout.
def get_circuit_diagram_extent(elements: Sequence[Any], connection: str) -> List[float]:
    total_extent = [0.0,0.0]
    for i, element in enumerate(elements):
        extent_ = element.circuit_diagram_extent
        if connection=="series":
            total_extent[0] = max(total_extent[0], extent_[0])
            total_extent[1] += extent_[1]
            if i > 0:
                total_extent[1] += utilities.y_spacing
        else:
            total_extent[1] = max(total_extent[1], extent_[1])
            total_extent[0] += extent_[0]
            if i > 0:
                total_extent[0] += utilities.x_spacing
    if connection!="series":
        total_extent[1] += 0.2 # the connectors
    return total_extent

# Tile elements into a grid for layout and diagram placement.
def tile_elements(
    elements: Sequence[Any],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    x_gap: float = 0.0,
    y_gap: float = 0.0,
    turn: bool = False,
    xflip: bool = False,
    yflip: bool = False,
    col_wise_ordering: bool = True,
) -> None:
    """Tile elements into a grid for layout and diagram placement.

    Args:
        elements (Sequence[Any]): Elements with `extent` and `location`.
        rows (Optional[int]): Number of rows.
        cols (Optional[int]): Number of columns.
        x_gap (float): Horizontal gap between tiles.
        y_gap (float): Vertical gap between tiles.
        turn (bool): Alternate rotation per column/row.
        xflip (bool): Flip X orientation for 2-column layout.
        yflip (bool): Flip Y orientation for 2-row layout.
        col_wise_ordering (bool): Fill columns before rows.

    Returns:
        None

    Example:
        ```python
            from PV_Circuit_Model.device import Cell_
            cell = Cell_()
            cells = cell*5
            tile_elements(cells.parts,rows=1)
        ```
    """
    tile_objects = []
    for element in elements:
        if hasattr(element,"extent"):
            tile_objects.append(element)
    if rows is None and cols is None:
        rows = 1
    if rows is None:
        rows = int(np.ceil(float(len(tile_objects)) /float(cols)))
    if cols is None:
        cols = int(np.ceil(float(len(tile_objects)) /float(rows)))
    row = 0
    col = 0
    rotation = 0
    pos = np.array([0,0]).astype(float)
    max_x_extent = 0.0
    max_y_extent = 0.0
    for element in tile_objects:
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
    if xflip and cols==2 and len(tile_objects)==2:
        tile_objects[1].x_mirror = -1
    if yflip and rows==2 and len(tile_objects)==2:
        tile_objects[1].y_mirror = -1

def circuit_deepcopy(circuit_component: CircuitComponent) -> CircuitComponent:
    """Return a deep clone of a circuit component.

    Warning:
        This uses the component's `clone` method and does not deep-copy
        external resources.

    Args:
        circuit_component (CircuitComponent): Component to clone.

    Returns:
        CircuitComponent: Cloned component.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import R, circuit_deepcopy
        r = R(1)
        r2 = circuit_deepcopy(r)
        ```
    """
    return circuit_component.clone()

# Search helpers for nested CircuitGroup structures.
def find_subgroups_by_name(circuit_group: "CircuitGroup", target_name: str) -> List[CircuitComponent]:
    """Find subgroups with a matching name.

    Args:
        circuit_group (CircuitGroup): Root group to search.
        target_name (str): Name to match.

    Returns:
        List[CircuitComponent]: Matching subgroups.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import CircuitGroup, R
        group = CircuitGroup([R(1)], connection="series", name="root")
        len(find_subgroups_by_name(group, "root")) # 1
        ```
    """
    result = []
    for element in circuit_group.subgroups:
        if hasattr(element, 'name') and element.name == target_name:
            result.append(element)
        if isinstance(element, CircuitGroup):
            result.extend(find_subgroups_by_name(element, target_name))
    return result

# Search helpers for tags in nested CircuitGroup structures.
def find_subgroups_by_tag(circuit_group: "CircuitGroup", tag: Any) -> List[CircuitComponent]:
    """Find subgroups with a matching tag.

    Warning:
        This uses name-based recursion for nested groups and may not find
        tagged children in all cases.

    Args:
        circuit_group (CircuitGroup): Root group to search.
        tag (Any): Tag value to match.

    Returns:
        List[CircuitComponent]: Matching subgroups.

    Example:
        ```python
        from PV_Circuit_Model.circuit_model import series, R
        group = series(R(1, tag="R1"), R(2))
        len(find_subgroups_by_tag(group, "R1")) # 1
        ```
    """
    result = []
    for element in circuit_group.subgroups:
        if hasattr(element, 'tag') and element.tag == tag:
            result.append(element)
        if isinstance(element, CircuitGroup):
            result.extend(find_subgroups_by_name(element, tag))
    return result



