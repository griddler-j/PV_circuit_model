# 📦 PV Circuit Model – Overview
Photovoltaic (PV) devices such as solar cells and modules are often best represented by a network of interconnected subcomponents arranged in series and parallel. This Python package provides a flexible framework to define and simulate such circuit models with ease.

![Alt text](PV_circuit_model.png)

It also includes tools to simulate and analyze standard PV measurements (e.g., I-V curves, Suns-Voc) and fit circuit models to experimental data.

# 🛠️ Getting Started
1. Install dependencies - 
Navigate to the package directory and run: `pip install -r requirements.txt`

1. Install the package in development mode: `pip install -e .`

1. Build the Cython/C++ extension (ivkernel).  The high-performance I-V engine is implemented as a Cython wrapper around a C++ kernel.
It must be compiled before use. Command line: navigate to PV_Circuit_Module and run “python setup.py build_ext --inplace” 

1. Run example code
You can test the package functionality by executing the example script inside the examples/ directory:
