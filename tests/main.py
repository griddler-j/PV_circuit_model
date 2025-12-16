import importlib

modules = [
    "a01_make_solar_cell",
    "a02_make_PV_module",
    "a03_make_tandem_cell",
    "a04_make_PV_string",
    "a05_simple_cell_compared_to_LT_spice",
    "a06_simple_module_compared_to_LT_spice",
    "a07_simple_uniform_module_compared_to_LT_spice",
    "a08_simple_string_compared_to_LT_spice",
    "a09_simple_uniform_string_compared_to_LT_spice",
    "a10_simple_parallel_strings_compared_to_LT_spice",
]
from utilities import *

if __name__ == "__main__":
    for i, name in enumerate(modules):
        print(f"\n--------------------------------\nRunning Test {i+1}: {name}\n--------------------------------\n")
        mod = importlib.import_module(name)
        mod.run_test()