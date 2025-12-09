import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numbers import Number
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Callable
import json
import numpy as np
import json
import pickle
import importlib
import inspect
try:
    import bson
except:
    pass
from pathlib import Path

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
    
ParameterSet(name="VT_at_25C",data=0.02568)
VT_at_25C = ParameterSet.get_set("VT_at_25C")()

get_VT = lambda temperature, VT_at_25C: VT_at_25C*(temperature + 273.15)/(25 + 273.15)

pbar = None
x_spacing = 1.5
y_spacing = 0.2

def interp_(x, xp, fp):
    if xp.size==1:
        return fp[0]*np.ones_like(x)
    if xp[-1] > xp[0]:
        y = np.interp(x, xp, fp)
    else:
        y = np.interp(-x, -xp, fp)
    if isinstance(x,Number):
        return y
    if x[0] < xp[0]:
        left_slope = (fp[1]-fp[0])/(xp[1]-xp[0])
        find_ = np.where(x < xp[0])[0]
        y[find_] = fp[0] + (x[find_]-xp[0])*left_slope
    if x[-1] > xp[-1]:
        right_slope = (fp[-1]-fp[-2])/(xp[-1]-xp[-2])
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

def draw_symbol(draw_func, ax=None,  x=0, y=0, color="black", text=None, **kwargs):
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
        ax.text(x+text_x,y+text_y,text, va='center', fontsize=6, color=color)
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

def draw_forward_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=False, rotation=rotation)

def draw_reverse_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="up", is_LED=False, rotation=rotation)

def draw_LED_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=True, rotation=rotation)

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

class ParamSerializable:
    _critical_fields = ()
    _artifacts = ()
    _dont_serialize = ()
    def clone(self,parent=None):    
        new = self.__class__.__new__(self.__class__)
        subgroups = getattr(self,"subgroups",[])
        if subgroups:
            subgroups_clone = [item.clone(new) for item in subgroups]
            new.__init__(subgroups=subgroups_clone)
        new.parent = parent
        d = {}
        for k, v in self.__dict__.items():
            if k=="subgroups": # already done, skip
                continue 
            if k=="parent": # already done, skip
                continue 
            if isinstance(v,ParamSerializable):
                if k in self._critical_fields:
                    d[k] = v.clone()
            elif isinstance(v, list):
                if len(v)>0 and isinstance(v[0],ParamSerializable):
                    pass
                else:
                    d[k] = v[:]  # shallow list copy
            elif k in self._dont_serialize:
                pass # don't copy
            elif k in self._artifacts:
                if hasattr(type(self), k):
                    d[k] = getattr(type(self), k) # revert to default
                else:
                    pass # don't copy
            elif hasattr(v, "copy"):  # NumPy array or similar
                d[k] = v.copy()
            elif isinstance(v, dict):
                d[k] = v.copy()
            else:
                d[k] = v  # assume immutable or shared
        new.__dict__.update(d)
        return new
    
    # equality that checks only _critical_fields, handles nesting too
    def __eq__(self, other): 
        if self.__class__ is not other.__class__:
            return NotImplemented
        return all(
            getattr(self, f) == getattr(other, f)
            for f in self._critical_fields
        )
    
    def clear_artifacts(self):
        for field in self._artifacts:
            if hasattr(self, field):
                if hasattr(type(self), field):
                    setattr(self, field, getattr(type(self), field))
                elif field in self.__dict__:
                    delattr(self, field)

    def save_toParams(self, critical_fields_only=False):
        data = {
            "__class__": f"{self.__class__.__module__}.{self.__class__.__name__}"
        }
        for name, value in self.__dict__.items():
            if name in self._artifacts or name in self._dont_serialize or (critical_fields_only and name not in self._critical_fields):
                continue
            output = self._save_value(name,value,critical_fields_only=critical_fields_only,critical_fields=self._critical_fields)
            if output is not None:
                data[name] = output
        return data

    def save_to_json(self, path, *, indent=2):
        params = self.save_toParams()
        with open(path, "w") as f:
            json.dump(params, f, indent=indent)
        return path

    @staticmethod
    def restore_from_json(path):
        with open(path, "r") as f:
            params = json.load(f)
        return ParamSerializable.Restore_fromParams(params)
    
    def save_to_bson(self, path):
        params = self.save_toParams()
        data = bson.dumps(params)
        with open(path, "wb") as f:
            f.write(data)
        return path

    @staticmethod
    def restore_from_bson(path):
        with open(path, "rb") as f:
            params = bson.loads(f.read())
        return ParamSerializable.Restore_fromParams(params)

    def save_to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def restore_from_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def Restore_fromParams(params):
        return ParamSerializable._restore_value(params)

    @staticmethod
    def _save_value(field_name,value,critical_fields_only=False,critical_fields=None):
        if isinstance(value, ParamSerializable): # we don't store any references to other ParamSerializables, except those found in subgroups
            if field_name != "subgroups" and (critical_fields is None or field_name not in critical_fields):
                return None
            return value.save_toParams(critical_fields_only=critical_fields_only)

        if isinstance(value, (list, tuple)): # we don't store any references to other ParamSerializables, except those found in subgroups
            if field_name == "subgroups":
                return [ParamSerializable._save_value(field_name, v,critical_fields_only=critical_fields_only,critical_fields=critical_fields) for v in value]
            elif len(value)>0 and isinstance(value[0],ParamSerializable):
                return None
            else:
                return value[:]

        if isinstance(value, dict):
            return {k: ParamSerializable._save_value("generic",v,critical_fields_only=critical_fields_only,critical_fields=critical_fields) for k, v in value.items()}

        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": value.tolist(),
                "dtype": str(value.dtype),
                "shape": value.shape,
            }

        return value

    @staticmethod
    def _restore_value(value):
        # numpy array
        if isinstance(value, dict) and "__ndarray__" in value:
            arr = np.array(value["__ndarray__"], dtype=value["dtype"])
            return arr.reshape(value["shape"])

        # nested ParamSerializable subclass (or any class we serialized)
        if isinstance(value, dict) and "__class__" in value:
            cls_path = value["__class__"]
            module_name, class_name = cls_path.rsplit(".", 1)

            # dynamic import trick
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)

            # recursively restore all fields except __class__
            raw_kwargs = {
                k: ParamSerializable._restore_value(v)
                for k, v in value.items()
                if k != "__class__"
            }

            # Look at __init__ signature
            sig = inspect.signature(cls.__init__)
            params = sig.parameters

            # If class accepts **kwargs, just pass everything
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if accepts_var_kw:
                return cls(**raw_kwargs)

            # Otherwise, split into "constructor args" and "extra attrs"
            allowed_names = {
                name
                for name, p in params.items()
                if name != "self"
                and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }

            init_kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed_names}
            extra_kwargs = {k: v for k, v in raw_kwargs.items() if k not in allowed_names}

            # Instantiate with the allowed kwargs
            obj = cls(**init_kwargs)

            # Attach any extra fields as attributes
            for k, v in extra_kwargs.items():
                setattr(obj, k, v)

            return obj

        # list
        if isinstance(value, list):
            return [ParamSerializable._restore_value(v) for v in value]

        # regular dict
        if isinstance(value, dict):
            return {k: ParamSerializable._restore_value(v) for k, v in value.items()}

        # primitives
        return value