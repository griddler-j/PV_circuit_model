# PV-Circuit-Model

**PV-Circuit-Model** is a fast, hierarchical photovoltaic circuit simulation framework
for cells, modules, strings, and large PV systems.

It is designed to model real-world mismatch, shading, and nonlinear effects using
a composable series/parallel circuit abstraction.

---

## Installation

```bash
pip install PV-Circuit-Model
```

---

## Minimal example

```python

from PV_Circuit_Model.circuit_model import IL, D1, D2, Dintrinsic_Si, Drev, R, CircuitGroup

# Notation: A | B means "connect A, B in parallel", and A + B means "connect A, B in series"
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

circuit_group.draw(display_value=True)
circuit_group.plot(title="Cell Parts I-V Curve")
circuit_group.show()
```

---

## Learn more

- [Examples](examples.md)
- [API Reference](api/index.md)

---

## Links:

- [PyPI](https://pypi.org/project/PV-Circuit-Model/)
- [GitHub](https://github.com/griddler-j/PV_circuit_model)
- [Binder demo](https://mybinder.org/v2/gh/arsonwong/PV_circuit_model/HEAD)
