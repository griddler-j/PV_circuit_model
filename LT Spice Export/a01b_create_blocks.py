from utilities import *
from pathlib import Path
import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent

np.random.seed(1)

ref_module = ParamSerializable.restore_from_bson(THIS_DIR / "ref_module.bson")
modules = [ref_module.clone() for _ in tqdm(range(26))]
string = CircuitGroup(modules)

strings = [string.clone() for _ in tqdm(range(100))]

all_cells = []
for string in tqdm(strings):
    for module in tqdm(string.subgroups):
        all_cells.extend(module.cells)
        for cell in module.cells: 
            cell.set_JL(cell.JL() * min(1.0,np.random.normal(loc=1.0, scale=0.05)))
            cell.set_J01(cell.J01() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
            cell.set_J02(cell.J02() * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
            cell.set_specific_shunt_res(10000 * 10**(np.random.normal(loc=0, scale=0.5)))
            cell.set_specific_Rs(0.3 * 10**(np.random.normal(loc=0, scale=0.2)))
p = np.random.permutation(len(all_cells))
for i in range(int(len(p)/100)):
    cell = all_cells[p[i]]
    cell.set_JL(cell.JL()*np.random.rand())

for num_strings in tqdm([4,8,16,32,64,100]):
    block = CircuitGroup(strings[:num_strings],connection="parallel")
    block.save_to_bson(THIS_DIR / f"block_{num_strings}.bson", critical_fields_only=True)
    export_ltspice_netlist(block, THIS_DIR / f"block_{num_strings}.net")





