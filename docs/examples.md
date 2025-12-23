# Examples

This page highlights common modeling patterns.

---

## Make a solar cell structure

```python
from PV_Circuit_Model.device import *
from PV_Circuit_Model.device_analysis import *

# A solar cell can be made of these circuit elements.  
# # Notation: A | B means "connect A, B in parallel", and A + B means "connect A, B in series"
# IL(41e-3) = CurrentSource with IL = 41e-3A
# D1(10e-15) = ForwardDiode with I0 = 10e-15A, n=1
# D2(5e-9) = ForwardDiode with I0 = 5e-9A, n=2
# Dintrinsic_Si(180e-4) = Intrinsic_Si_diode in silicon with base thickness 180e-4 (doping, doping type set to default values)
# Drev(V_shift=10) = ReverseDiode with breakdown voltage 10V
# R(1e5), R(1/3) = Resistor(s) of 1e5ohm, 1/3ohm
circuit_group = ( 
    (IL(41e-3) | D1(10e-15) | D2(5e-9) | Dintrinsic_Si(180e-4) | Drev(V_shift=10) | R(1e5)) 
    + R(1/3)
)

# .draw will draw the circuit diagram
circuit_group.draw(display_value=True)

# .plot will plot the I-V Curve
circuit_group.plot(title="Cell Parts I-V Curve")
circuit_group.show()
```

---

## Type Casting

```python
# We can cast circuit_group as type Cell to give it additional shape and area
cell_ = circuit_group.as_type(Cell, **wafer_shape(format="M10",half_cut=True))

# Now cell has a shape and size that we can see
_ = cell_.draw_cells()

# Also, plotting a cell will show the I-V curve with the current density multiplied by the cell area
cell_.plot(title="Cell I-V Curve")
cell_.show()
```

---

## Short Cut for making a solar cell

```python
# Because cells are frequently defined, we offer a short cut definition
# Cell_ has the advantage that you can specify target I-V parameters for the diode parameters to tune to
cell = Cell_(Jsc=0.042, Voc=0.735, FF=0.82, Rs=0.3333, Rshunt=1e6, wafer_format="M10",half_cut=True)

cell.plot(title="Cell I-V Curve")
cell.show()
```

---

## Organizing parts in series and parallel connections to build a module

```python
# Let's put 24 x 2 x 3 = 144 cells together to make a module.  Note these notations below:
# A*24 = A + A .... + A = connect 24 copies of A's together in series
# tile_subgroups is optional to arrange the cells spatially, just for visualization
half_string = (cell*24 + R(0.05)).tile_subgroups(cols=2,x_gap=0.1,y_gap=0.1,turn=True)

# B**2 = B | B = connect 2 copies of B's together in parallel
# Dbypass is a bypass diode (an alias for ReverseDiode)
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
section = (half_string**2 | Dbypass()).tile_subgroups(cols = 1, y_gap = 1, yflip=True)

# C*3 = C + C + C = connect 3 copies of C's together in series
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
circuit_group = (section*3).tile_subgroups(rows=1, x_gap = 1)

# type cast to Module just for encapsulation
module_ = circuit_group.as_type(Module) 
_ = module_.draw_cells()

module_.plot(title="Module I-V Curve")
module_.show()
```

---

## Short Cut to making a module

```python
# Because modules are frequently defined, we offer a short cut definition
module = Module_(Isc=14, Voc=0.72*72, FF=0.8, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, half_cut=True, butterfly=True)

module.plot(title="Module I-V Curve")
module.show()
```

---

## Introduce cell to cell variations inside the module

```python
# Manipulate the cell properties
np.random.seed(0)
for cell in module.cells:
    cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.01)))
    cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))

module.plot(title="Module I-V Curve with inhomogenity")
module.show()

# Simulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cells Vint with inhomogenity",colour_bar=True) 

# Give one of the cells very large series resistance
module.cells[0].set_specific_Rs(40.0)
module.set_Suns(1.0) 
module.plot(title="Module I-V Curve with additional high Rs cell")
module.show()

# Resimulate cell internal voltages under electroluminescence (EL) conditions 
# No illumination, drive module at 10A forward bias
module.set_Suns(0.0) 
module.set_operating_point(I=10)
_ = module.draw_cells(title="Cell Vint with additional high Rs cell",colour_bar=True)

# By the way, one can always use the attribute .subgroups or .children or .parts (they're all the same)
# to access the child components of a CircuitGroup.  For example:
count = 0
for section in module.parts:
    for part in section.parts:
        if isinstance(part,CircuitGroup): # i.e. a substring, not a bypass diode
            print(f"Substring {count+1} is passing {part.operating_point[1]:.2f} A of current")
            count += 1
# Notice how the substring that contains the high series resistance cell has a low current, and that
# causes a high current in the substring that's parallel to it

# We can even access the I-V characteristics of each substring if we like
count = 0
for section in module.parts:
    for part in section.parts:
        if isinstance(part,CircuitGroup): # i.e. a substring, not a bypass diode
            set_Suns(part,1.0)
            Voc = part.get_Voc()
            Isc = part.get_Isc()
            Pmax = part.get_Pmax()
            FF = part.get_FF()
            print(f"If Substring {count+1} were accessed individually, it would have 1-Sun Pmax={Pmax:.2f}W, Voc={Voc:.2f}V, Isc={Isc:.2f}A, FF={FF*100:.2f}%")
            count += 1
# Notice how the substring that contains the high series resistance cell has a low current, and that
# causes a high current in the substring that's parallel to it
```

---

## Large systems

PV-Circuit-Model scales to thousands of cells by exploiting hierarchical IV merging.

```python
# we series connect 10 modules together
# again, tile_subgroups is optional to arrange the subparts spatially, for ease of visualization
module = quick_module(Isc=14, Voc=0.72*72, FF=0.8, wafer_format="M10", num_strings=3, num_cells_per_halfstring=24, half_cut=True, butterfly=True)
module_string = (module*10).tile_subgroups(rows=1, x_gap = 20)
# type cast to Device just for encapsulation
module_string = module_string.as_type(Device,name="string")

# to make the simulation interesting, let's make ~1% of the cells in the module string partially shaded
cells = module_string.findElementType(Cell)
for cell in cells:
    dice = np.random.uniform(0.0, 1.0)
    if dice < 0.01:
        shading = np.random.uniform(0.2,0.5)
        cell.set_Suns(1-shading)

module_string.plot(title="Module string I-V Curve with some cells partially shaded",show_IV_parameters=False)
module_string.show()

module_string_pair = (module_string**2).tile_subgroups(cols=1, y_gap = 50)
# restore one of the strings to 1 Sun without shading
module_string_pair.parts[1].set_Suns(1.0)

module_string_pair.plot(title="String pair I-V Curve with one string partially shaded",show_IV_parameters=False)
module_string_pair.show()

_ = module_string_pair.draw_cells(min_value=0.3)
```
