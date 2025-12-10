from utilities import *
from pathlib import Path
import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent

np.random.seed(1)
ref_module = quick_butterfly_module(Si_intrinsic_limit=False) # no instrinic Si diodes
for cell in tqdm(ref_module.cells): # pluck out all the cell ReverseDiodes
    diode_branch = cell.diode_branch
    diode_branch.subgroups = [element for element in diode_branch.subgroups if not isinstance(element,ReverseDiode)]
ref_module = ref_module.clone() # cloning will reconstruct the module, leading to corrected num_circuit_elements

ref_module.save_to_bson(THIS_DIR / "ref_module.bson", critical_fields_only=True)
export_ltspice_netlist(ref_module, THIS_DIR / "ref_module.net")

module = ref_module.clone()
for cell in tqdm(module.cells): 
    cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
    cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
    cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
    cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
    cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
module.cells[0].set_JL(module.cells[0].JL()/2)

module.cells[1].save_to_json(THIS_DIR / "cell.json", critical_fields_only=True)
module.cells[1].save_to_bson(THIS_DIR / "cell.bson", critical_fields_only=True)
export_ltspice_netlist(module.cells[1], THIS_DIR / "cell.net")

module.save_to_bson(THIS_DIR / "module.bson", critical_fields_only=True)
export_ltspice_netlist(module, THIS_DIR / "module.net")

modules = [ref_module.clone() for _ in tqdm(range(26))]

all_cells = []
for module in tqdm(modules):
    all_cells.extend(module.cells)
    for cell in module.cells: 
        cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
        cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
        cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
    p = np.random.permutation(len(all_cells))
    for i in range(int(len(p)/50)):
        cell = all_cells[p[i]]
        cell.set_JL(cell.JL()*np.random.rand())

string = CircuitGroup(modules)

string.save_to_bson(THIS_DIR / "string.bson", critical_fields_only=True)
export_ltspice_netlist(string, THIS_DIR / "string.net")

ref_modules = [ref_module.clone() for _ in tqdm(range(26))]
ref_string = CircuitGroup(ref_modules)

ref_string.save_to_bson(THIS_DIR / "ref_string.bson", critical_fields_only=True)
export_ltspice_netlist(ref_string, THIS_DIR / "ref_string.net")

parallel_strings2 = CircuitGroup([string,ref_string],connection="parallel")
parallel_strings2.save_to_bson(THIS_DIR / "parallel_strings2.bson", critical_fields_only=True)
export_ltspice_netlist(parallel_strings2, THIS_DIR / "parallel_strings2.net")





