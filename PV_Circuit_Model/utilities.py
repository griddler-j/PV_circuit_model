import numpy as np
import numbers
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numbers import Number
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Callable, Union, Literal
import json
import importlib
import math
import inspect
from pathlib import Path
from datetime import date, datetime
import base64
try:
    import bson
except ModuleNotFoundError:
    bson = None

@dataclass
class ParameterSet:
    # ----- identity / metadata -----
    name: str                              # e.g. "wafer_formats"
    filename: str | None = None            # e.g. "wafer_formats.json"
    loader: Callable[[str], Any] | None = None  # optional custom loader
    data: Any = None
    success_init: bool = field(init=False, default=True)

    # ----- global registry of all instances -----
    _registry: ClassVar[Dict[str, "ParameterSet"]] = {}

    def __post_init__(self):    
        if self.filename is not None and self.data is None:
            try:
                if self.loader is not None:
                    self.data = self.loader(self.filename)
                else: # try json
                    path = Path(self.filename)
                    with path.open("r", encoding="utf-8") as f:
                        self.data = json.load(f)
            except Exception as e:
                self.success_init = False
                print(f"[ParameterSet Warning] Failed to load {self.filename} "
                        f"Error: {e}")

        # register in global registry
        ParameterSet._registry[self.name] = self

    def __call__(self):
        return self.data

    def __getitem__(self, key):
        if isinstance(self.data, dict):
            return self.data.get(key)
        return None
    
    def set(self,key,value):
        if isinstance(self.data, dict):
            self.data[key] = value

    @classmethod
    def get_registry(cls) -> Dict[str, "ParameterSet"]:
        return dict(cls._registry)
    
    @classmethod
    def get_set(cls,name):
        dict_ = cls.get_registry()
        if name in dict_:
            return dict_[name]
        return None

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

ParameterSet(name="constants",filename=PARAM_DIR / "constants.json")
constants = ParameterSet.get_set("constants")()
VT_at_25C = constants["VT_at_25C"]
zero_C = constants["zero_C"]
q = constants["q"]

def get_VT(temperature):
    return VT_at_25C*(temperature + zero_C)/(25 + zero_C)

pbar = None
x_spacing = 1.5
y_spacing = 0.2

# requires x values to be in ascending order now
def interp_(x, xp, fp, optional_left_slope=None, optional_right_slope=None, extrap = True):
    if xp[-1] < xp[0]:
        raise AttributeError("interp_ requires xp to be in increasing order1")
    if xp.size==1 or xp[-1]==xp[0]:
        return fp[0]*np.ones_like(x)
    y = np.interp(x, xp, fp)
    if not extrap:
        return y
    if isinstance(x,Number):
        if x < xp[0]:
            if optional_left_slope is not None:
                left_slope = optional_left_slope
            else:
                for j in range(1,xp.size):
                    if xp[j] > xp[0]:
                        left_slope = (fp[j]-fp[0])/(xp[j]-xp[0])
                        break
                y = fp[0] + (x-xp[0])*left_slope
        elif x > xp[-1]:
            if optional_right_slope is not None:
                right_slope = optional_right_slope
            else:
                for j in range(xp.size-2,-1,-1):
                    if xp[j] < xp[-1]:
                        right_slope = (fp[-1]-fp[j])/(xp[-1]-xp[j])
                        break
            y = fp[-1] + (x-xp[-1])*right_slope
        return y

    if x[0] < xp[0]:
        if optional_left_slope is not None:
            left_slope = optional_left_slope
        else:
            for j in range(1,xp.size):
                if xp[j] > xp[0]:
                    left_slope = (fp[j]-fp[0])/(xp[j]-xp[0])
                    break
        find_ = np.where(x < xp[0])[0]
        y[find_] = fp[0] + (x[find_]-xp[0])*left_slope
    if x[-1] > xp[-1]:
        if optional_right_slope is not None:
            right_slope = optional_right_slope
        else:
            for j in range(xp.size-2,-1,-1):
                if xp[j] < xp[-1]:
                    right_slope = (fp[-1]-fp[j])/(xp[-1]-xp[j])
                    break
        find_ = np.where(x > xp[-1])[0]
        y[find_] = fp[-1] + (x[find_]-xp[-1])*right_slope
    return y

def rotate_points(xy_pairs, origin, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated = []
    ox, oy = origin
    for x, y in xy_pairs:
        vec = np.array([x - ox, y - oy])
        rx, ry = rot_matrix @ vec + np.array([ox, oy])
        rotated.append((rx, ry))
    return rotated

def draw_symbol(draw_func, ax=None,  x=0, y=0, color="black", text=None, fontsize=6, **kwargs):
    draw_immediately = False
    if ax is None:
        draw_immediately = True
        _, ax = plt.subplots()
    draw_func(ax=ax, x=x, y=y, color=color, **kwargs)
    if text is not None:
        text_x = 0.14
        text_y = 0.0
        if draw_func==draw_CC_symbol:
            text_x = 0.21
        elif draw_func==draw_resistor_symbol:
            text_y = -0.15
        ax.text(x+text_x,y+text_y,text, va='center', fontsize=fontsize, color=color)
    if draw_immediately:
        plt.show()

def draw_diode_symbol(ax, x=0, y=0, color="black", up_or_down="down", is_LED=False, rotation=0, linewidth=1.5):
    origin = (x, y)
    dir = 1 if up_or_down == "down" else -1

    # Diode circle for LED
    if is_LED:
        circle = patches.Circle(origin, 0.17, edgecolor=color, facecolor='white', linewidth=linewidth, fill=False)
        ax.add_patch(circle)

    # Diode triangle arrowhead
    arrow_start = (x, y + 0.075 * dir)
    arrow_end = (x, y + 0.074 * dir - 0.001 * dir)  # Very short shaft
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.15, head_length=0.15, fc=color, ec=color)

    # Diode bar
    bar_pts = rotate_points([(x - 0.075, y - 0.08 * dir), (x + 0.075, y - 0.08 * dir)], origin, rotation)
    ax.add_line(plt.Line2D([bar_pts[0][0], bar_pts[1][0]], [bar_pts[0][1], bar_pts[1][1]], color=color, linewidth=linewidth*2/1.5))

    # LED rays
    if is_LED:
        ray1 = rotate_points([(x - 0.05, y - 0.05 * dir), (x - 0.2, y - 0.2 * dir)], origin, rotation)
        ray2 = rotate_points([(x - 0.075, y + 0.025 * dir), (x - 0.225, y - 0.125 * dir)], origin, rotation)
        ax.arrow(*ray1[0], ray1[1][0] - ray1[0][0], ray1[1][1] - ray1[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')
        ax.arrow(*ray2[0], ray2[1][0] - ray2[0][0], ray2[1][1] - ray2[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')

    # Terminals
    term_top = rotate_points([(x, y + 0.08), (x, y + 0.4)], origin, rotation)
    term_bot = rotate_points([(x, y - 0.08), (x, y - 0.4)], origin, rotation)
    ax.add_line(plt.Line2D([term_top[0][0], term_top[1][0]], [term_top[0][1], term_top[1][1]], color=color, linewidth=linewidth))
    ax.add_line(plt.Line2D([term_bot[0][0], term_bot[1][0]], [term_bot[0][1], term_bot[1][1]], color=color, linewidth=linewidth))

def draw_forward_diode_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=False, rotation=rotation, linewidth=linewidth)

def draw_reverse_diode_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="up", is_LED=False, rotation=rotation, linewidth=linewidth)

def draw_LED_diode_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=True, rotation=rotation, linewidth=linewidth)

def draw_CC_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    origin = (x, y)

    # Draw rotated circle
    circle = patches.Circle((x, y), 0.17, edgecolor=color, facecolor="white", linewidth=2/1.5*linewidth)
    ax.add_patch(circle)

    # Arrow inside circle (from lower to upper)
    arrow_start = (x, y - 0.12)
    arrow_end = (x, y + 0.02)
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.1, head_length=0.1, width=0.01, fc=color, ec=color)

    # Vertical terminals (above and below the circle)
    line1 = rotate_points([(x, y + 0.18), (x, y + 0.4)], origin, rotation)
    line2 = rotate_points([(x, y - 0.18), (x, y - 0.4)], origin, rotation)
    ax.add_line(plt.Line2D([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color=color, linewidth=linewidth))
    ax.add_line(plt.Line2D([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color=color, linewidth=linewidth))

def draw_resistor_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    dx = 0.075
    dy = 0.02
    ystart = y + 0.15
    origin = (x, y)

    segments = [
        [(x, y+0.15), (x, y+0.4)],
        [(x, y-0.09), (x, y-0.4)],
    ]

    for _ in range(3):
        segments += [
            [(x, ystart), (x+dx, ystart-dy)],
            [(x+dx, ystart-dy), (x-dx, ystart-3*dy)],
            [(x-dx, ystart-3*dy), (x, ystart-4*dy)]
        ]
        ystart -= 4 * dy

    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=linewidth))

def draw_earth_symbol(ax, x=0, y=0, color="black", rotation=0, linewidth=1.5):
    origin = (x, y)
    segments = []
    # Vertical line
    segments.append([(x, y + 0.05), (x, y + 0.1)])
    # Horizontal lines
    for i in range(3):
        x1 = x - 0.03 * (i + 1)
        x2 = x + 0.03 * (i + 1)
        y_level = y - 1*0.05 + 0.05 * i
        segments.append([(x1, y_level), (x2, y_level)])
    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=linewidth*2/1.5))

def draw_pos_terminal_symbol(ax, x=0, y=0, color="black", linewidth=1.5):
    circle = patches.Circle((x, y), 0.04, edgecolor=color,facecolor="white",linewidth=linewidth*2/1.5, fill=True)
    ax.add_patch(circle)

def convert_ndarrays_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class SerializedPackage:
    # blah blah
    pass

class Artifact:
    """Serializable artifact base class.
    """
    _critical_fields = () # equality is based on _critical_fields
    _parent_pointer_name = None # name of the field that points to a parent
    _parent_pointer_class = None # class of parent
    _ephemeral_fields = () # these are erased on clear_ephemeral_fields().  They are also ignored and not saved.
    _dont_serialize = () # these fields are ignored and not saved. Usually they are pointers.
    _float_rtol = 1e-6 # used in comparison for equality
    _float_atol = 1e-23 # used in comparison for equality
    def __init__(self, object_) -> None:
        """Wrap an object as Artifact so that it can be saved via dump and loaded via load like pickle

        Args:
            object_ (Optional[Any]): artifact payload.

        Returns:
            None

        Example:
            ```python
            from PV_Circuit_Model.utilities import Artifact
            art = Artifact({"a": 1})
            ```
        """
        self.artifacts_to_save = object_

    def __post_init__(self) -> None: # One can add on for child classes, for instance to set other pointers
        if self.__class__._parent_pointer_name is not None and self.__class__._parent_pointer_class is not None:
            setattr(self,self.__class__._parent_pointer_name,None)
        for k, v in vars(self).items():
            if k == self.__class__._parent_pointer_name:
                continue
            if k in self.__class__._dont_serialize:
                continue
            if k in self.__class__._ephemeral_fields:
                continue
            if isinstance(v,Artifact) and v.__class__._parent_pointer_name is not None and v.__class__._parent_pointer_class is not None and isinstance(self,v.__class__._parent_pointer_class):
                if v is self: # if by some accident there's circular ref
                    continue
                setattr(v,v.__class__._parent_pointer_name,self)
            elif isinstance(v,list): 
                for item in v:
                    if item is self: # if by some accident there's circular ref
                        continue
                    if isinstance(item,Artifact) and item.__class__._parent_pointer_name is not None:
                        setattr(item,item.__class__._parent_pointer_name,self)

    @staticmethod
    def clone_or_package(obj,mode: Literal["clone","json","bson","unpack"]="clone", critical_fields_only: bool = False) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj,Artifact) or (mode == "unpack" and isinstance(obj, dict) and ("__class__" in obj or ("__module__" in obj and "__qualname__" in obj))):
            if isinstance(obj,Artifact):
                cls = type(obj)
                fields = vars(obj)
            else:
                if ("__module__" in obj and "__qualname__" in obj):
                    module_name = obj["__module__"]
                    qualname = obj["__qualname__"]
                else:
                    classpath = obj["__class__"]
                    parts = classpath.split(".")
                    module_name = None
                    qualname = None
                    for i in range(len(parts) - 1, 0, -1):
                        candidate_module = ".".join(parts[:i])
                        candidate_qualname = ".".join(parts[i:])
                        try:
                            importlib.import_module(candidate_module)
                            module_name = candidate_module
                            qualname = candidate_qualname
                            break
                        except Exception:
                            continue
                    if module_name is None:
                        raise ImportError(f"Could not resolve legacy __class__: {classpath!r}")

                mod = importlib.import_module(module_name)
                cls = mod
                for part in qualname.split("."):
                    cls = getattr(cls, part)

                fields = {k: v for k, v in obj.items() if k not in ("__module__", "__qualname__","__class__")}

            if mode in ("clone","unpack"):
                new = cls.__new__(cls)
            else:
                new = {"__module__": cls.__module__, "__qualname__": cls.__qualname__}
            skip = set(cls._ephemeral_fields) | set(cls._dont_serialize)
            if cls._parent_pointer_name is not None:
                skip.add(cls._parent_pointer_name)
            for k, v in fields.items():
                if k in skip:
                    continue
                if critical_fields_only and k not in cls._critical_fields:
                    continue
                if mode in ("clone","unpack"):
                    setattr(new, k, Artifact.clone_or_package(v,mode=mode,critical_fields_only=critical_fields_only))
                else:
                    new[k] = Artifact.clone_or_package(v,mode=mode,critical_fields_only=critical_fields_only)
            if mode in ("clone", "unpack"):
                new.__post_init__()
            return new
        
        if isinstance(obj, dict):
            if mode == "unpack":
                if "_Packaging_Flag" in obj:
                    flag = obj["_Packaging_Flag"]
                    if "value" not in obj:
                        raise TypeError(f"Malformed packaged object (missing 'value'): {obj!r}")
                    value = obj["value"]
                    if flag in ("tuple","set"):
                        items = [Artifact.clone_or_package(i, mode=mode,critical_fields_only=critical_fields_only) for i in value]
                        if flag=="tuple":
                            return tuple(items)
                        if flag=="set":
                            return set(items)
                    if flag == "np.generic":
                        return np.array(value).item()
                    if flag == "np.ndarray":
                        return np.array(value)
                    if flag == "datetime":
                        return value if isinstance(value, datetime) else datetime.fromisoformat(value)
                    if flag == "date":
                        if isinstance(value, datetime):
                            return value.date()          # you stored a datetime for BSON
                        return date.fromisoformat(value) # you stored ISO string for JSON
                    if flag == "Path":
                        return Path(value)
                    if flag == "__bytes_b64__":
                        return base64.b64decode(value.encode("ascii"))
                    raise TypeError(f"Unknown _Packaging_Flag: {flag!r}")
            if mode in ("clone","unpack"):
                return {k: Artifact.clone_or_package(v, mode=mode,critical_fields_only=critical_fields_only) for k, v in obj.items()}
            else:
                return {str(k): Artifact.clone_or_package(v, mode=mode,critical_fields_only=critical_fields_only) for k, v in obj.items()}

        if isinstance(obj, list):
            return [Artifact.clone_or_package(item,mode=mode,critical_fields_only=critical_fields_only) for item in obj]
        
        if isinstance(obj, tuple):
            items = [Artifact.clone_or_package(i, mode=mode,critical_fields_only=critical_fields_only) for i in obj]
            return tuple(items) if mode == "clone" else {"_Packaging_Flag": "tuple", "value": items}
        
        if isinstance(obj, set):
            if mode=="clone":
                return set([Artifact.clone_or_package(item,mode=mode,critical_fields_only=critical_fields_only) for item in obj])
            else:
                return {"_Packaging_Flag": "set",
                    "value": [Artifact.clone_or_package(item, mode=mode, critical_fields_only=critical_fields_only)
                    for item in sorted(obj, key=repr)]}
        
        if isinstance(obj, np.generic):
            if mode=="clone":
                return obj
            else:
                return {"_Packaging_Flag": "np.generic", "value":  obj.item()}
        
        if isinstance(obj, np.ndarray):      
            if mode=="clone":
                return obj.copy()
            else:      
                return {"_Packaging_Flag": "np.ndarray", "value":  obj.tolist()}
            
        if isinstance(obj, datetime):
            if mode=="clone":
                return obj
            if mode=="bson":
                return obj
            if mode=="json":
                return {"_Packaging_Flag": "datetime", "value": obj.isoformat()}
        
        if isinstance(obj, date):
            if mode=="clone":
                return obj
            if mode=="bson":
                return {"_Packaging_Flag": "date", "value": datetime(obj.year, obj.month, obj.day)}
            if mode=="json":
                return {"_Packaging_Flag": "date", "value": obj.isoformat()}
            
        if isinstance(obj, Path):
            if mode=="clone":
                return obj
            else:
                return {"_Packaging_Flag": "Path", "value": str(obj)}
        
        if isinstance(obj, bytes):
            if mode=="clone":
                return obj
            if mode=="bson":
                return obj
            if mode=="json":
                return {"_Packaging_Flag": "__bytes_b64__", "value":  base64.b64encode(obj).decode("ascii")}

        return None # whatever falls through isn't serializable, like Fit_Dashboard or something like that.  Just return None    
    
    def clone(self,critical_fields_only: bool = False) -> "Artifact":
        """Clone this Artifact.

        Args:
            None

        Returns:
            Artifact: Cloned instance.

        Example:
            ```python
            from PV_Circuit_Model.utilities import Artifact
            art = Artifact({"a": 1})
            clone = art.clone()
            ```
        """
        return Artifact.clone_or_package(self,mode="clone",critical_fields_only=critical_fields_only)
    
    def dump(self, path: Union[str, Path], *, indent: int = 2, critical_fields_only: bool = False) -> str:
        """Write serialized parameters to JSON or BSON.

        Args:
            path (Union[str, Path]): Output file path.
            indent (int): JSON indentation.
            critical_fields_only (bool): If True, only serialize critical fields.

        Returns:
            str: Final path written.

        Raises:
            NotImplementedError: If an unsupported file extension is used.

        Example:
            ```python
            import tempfile
            from PV_Circuit_Model.utilities import Artifact
            art = Artifact({"a": 1})
            with tempfile.TemporaryDirectory() as folder:
                art.dump(Path(folder) / "artifact.json")
            ```
        """
        path = str(path)
        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(Artifact.clone_or_package(self,mode="json",critical_fields_only=critical_fields_only), f, indent=indent)
        else:
            pos = path.find(".")
            if pos == -1:
                path += ".bson"
            if not path.endswith(".bson"):
                raise NotImplementedError("Artifact.dump only supports .json or .bson output")
            with open(path, "wb") as f:
                payload = Artifact.clone_or_package(self, mode="bson", critical_fields_only=critical_fields_only)
                f.write(bson.dumps(payload))
        return path
    
    @staticmethod
    def load(path: Union[str, Path]) -> Any:
        """Load an Artifact or serialized object from JSON or BSON.

        Args:
            path (Union[str, Path]): Input file path.

        Returns:
            Any: Restored object.

        Raises:
            NotImplementedError: If an unsupported file extension is used.

        Example:
            ```python
            import tempfile
            from PV_Circuit_Model.utilities import Artifact
            art = Artifact({"a": 1})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "artifact.json"
                art.dump(path)
                Artifact.load(path)
            ```
        """
        path = str(path)
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
        elif path.endswith(".bson"):
            with open(path, "rb") as f:
                params = bson.loads(f.read())
        else:
            raise NotImplementedError("Artifact.load only supports .json or .bson input")
        return Artifact.clone_or_package(params,mode="unpack")

    # equality that checks only _critical_fields, handles nesting too
    def __eq__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        
        for f in self._critical_fields:
            a = getattr(self, f, None)
            b = getattr(other, f, None)
            if a is None and b is None:
                continue
            if a is None or b is None:
                return False

            if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
                if not math.isclose(
                    float(a), float(b),
                    rel_tol=self._float_rtol,
                    abs_tol=self._float_atol,
                ):
                    return False
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if not np.allclose(
                    a, b,
                    rtol=self._float_rtol,
                    atol=self._float_atol,
                ):
                    return False
            else:
                if a != b:
                    return False

        return True
    
    def clear_ephemeral_fields(self) -> None:
        for field_ in self._ephemeral_fields:
            if hasattr(self, field_):
                if hasattr(type(self), field_):
                    default = getattr(type(self), field_)
                    if isinstance(default, (list, dict, set)):
                        setattr(self, field_, default.copy())
                    else:
                        setattr(self, field_, default)
                elif field_ in self.__dict__:
                    delattr(self, field_)
    
def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
