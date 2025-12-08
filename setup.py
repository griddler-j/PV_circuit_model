# setup.py
from setuptools import setup, find_packages

version = {}
with open("PV_Circuit_Model/__init__.py") as f:
    exec(f.read(), version)

setup(
    name="PV_Circuit_Model",
    version=version["__version__"],   
    packages=find_packages(),
    author="Johnson Wong",
    description="PV Circuit model for cells, modules and components with additional tools for simulations and model fits to measurement data",
    license="MIT",
)
