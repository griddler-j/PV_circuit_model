import numpy as np
import numbers
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import json
from collections.abc import Mapping, Sequence

def interp_(x, xp, fp):
    """
    Like np.interp, but extrapolates linearly outside the bounds using the slopes
    of the first two and last two points.
    """
    xp_ = xp.copy()
    fp_ = fp.copy()
    is_number = False
    if isinstance(x, numbers.Number):
        x = np.array([x])
        is_number = True
    x_ = x.copy()
    if isinstance(x_,list):
        x_ = np.array(x_)
    if isinstance(xp_,list):
        xp_ = np.array(xp_)
    if isinstance(fp_,list):
        fp_ = np.array(fp_)
    while xp_[0]==xp_[1]:
        xp_ = xp_[1:]
        fp_ = fp_[1:]
    while xp_[-1]==xp_[-2]:
        xp_ = xp_[:-1]
        fp_ = fp_[:-1]
    if xp_[0] > xp_[-1]:
        xp_ = -1*xp_
        x_ = -1*x_
    y_multiplier = 1.0
    if fp_[0] > fp_[-1]:
        y_multiplier = -1.0
        fp_ = -1*fp_
    slope_left = (fp_[1] - fp_[0]) / (xp_[1] - xp_[0])
    slope_right = (fp_[-1] - fp_[-2]) / (xp_[-1] - xp_[-2])
    y = np.interp(x_, xp_, fp_)
    
    try:
        y[x_ < xp_[0]] = fp_[0] + slope_left * (x_[x_ < xp_[0]] - xp_[0])
    except Exception as e:
        print(x_)
        print(xp_[0])
        print("Caught an error:", e)
        assert(1==0)
    
    y[x_ > xp_[-1]] = fp_[-1] + slope_right * (x_[x_ > xp_[-1]] - xp_[-1])
    y *= y_multiplier
    if is_number:
        return y[0]
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

def draw_diode_symbol(ax, x=0, y=0, color="black", up_or_down="down", is_LED=False, rotation=0):
    origin = (x, y)
    dir = 1 if up_or_down == "down" else -1

    # Diode circle for LED
    if is_LED:
        circle = patches.Circle(origin, 0.17, edgecolor=color, facecolor='white', linewidth=1.5, fill=False)
        ax.add_patch(circle)

    # Diode triangle arrowhead
    arrow_start = (x, y + 0.075 * dir)
    arrow_end = (x, y + 0.074 * dir - 0.001 * dir)  # Very short shaft
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.15, head_length=0.15, fc=color, ec=color)

    # Diode bar
    bar_pts = rotate_points([(x - 0.075, y - 0.08 * dir), (x + 0.075, y - 0.08 * dir)], origin, rotation)
    ax.add_line(plt.Line2D([bar_pts[0][0], bar_pts[1][0]], [bar_pts[0][1], bar_pts[1][1]], color=color, linewidth=2))

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
    ax.add_line(plt.Line2D([term_top[0][0], term_top[1][0]], [term_top[0][1], term_top[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([term_bot[0][0], term_bot[1][0]], [term_bot[0][1], term_bot[1][1]], color=color, linewidth=1.5))

def draw_forward_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=False, rotation=rotation)

def draw_reverse_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="up", is_LED=False, rotation=rotation)

def draw_LED_diode_symbol(ax, x=0, y=0, color="black", rotation=0):
    draw_diode_symbol(ax=ax, x=x, y=y, color=color, up_or_down="down", is_LED=True, rotation=rotation)

def draw_CC_symbol(ax, x=0, y=0, color="black", rotation=0):
    origin = (x, y)

    # Draw rotated circle
    circle = patches.Circle((x, y), 0.17, edgecolor=color, facecolor="white", linewidth=2)
    ax.add_patch(circle)

    # Arrow inside circle (from lower to upper)
    arrow_start = (x, y - 0.12)
    arrow_end = (x, y + 0.02)
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.1, head_length=0.1, width=0.01, fc=color, ec=color)

    # Vertical terminals (above and below the circle)
    line1 = rotate_points([(x, y + 0.18), (x, y + 0.4)], origin, rotation)
    line2 = rotate_points([(x, y - 0.18), (x, y - 0.4)], origin, rotation)
    ax.add_line(plt.Line2D([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color=color, linewidth=1.5))

def draw_resistor_symbol(ax, x=0, y=0, color="black", rotation=0):
    dx = 0.075
    dy = 0.02
    ystart = y + 0.15
    origin = (x, y)

    # Top and bottom vertical leads
    segments = [
        [(x, y+0.15), (x, y+0.4)],
        [(x, y-0.09), (x, y-0.4)],
    ]

    # Zigzag segments
    for _ in range(3):
        segments += [
            [(x, ystart), (x+dx, ystart-dy)],
            [(x+dx, ystart-dy), (x-dx, ystart-3*dy)],
            [(x-dx, ystart-3*dy), (x, ystart-4*dy)]
        ]
        ystart -= 4 * dy

    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=1.5))

def draw_earth_symbol(ax, x=0, y=0, color="black", rotation=0):
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
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=2))

def draw_pos_terminal_symbol(ax, x=0, y=0, color="black"):
    circle = patches.Circle((x, y), 0.04, edgecolor=color,facecolor="white",linewidth=2, fill=True)
    ax.add_patch(circle)

class RandomNumberGenerator():
    pass

class CappedAbsGaussian(RandomNumberGenerator):
    def __init__(self,mean,stdev,cap=None):
        self.mean = mean
        self.stdev = stdev
        self.cap = cap
    def generate(self, sample_size = 1):
        x = np.abs(np.random.normal(self.mean,self.stdev,sample_size))
        if self.cap is not None:
            x = np.minimum(x,self.cap)
        return x

def convert_ndarrays_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
