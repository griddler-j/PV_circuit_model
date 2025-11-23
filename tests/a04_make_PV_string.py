from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
from utilities import *
from tqdm import tqdm

def main(display=True):
    np.random.seed(0)
    N = 26
    ref_module = quick_butterfly_module()
    modules = []
    for _ in tqdm(range(N)):
        module = circuit_deepcopy(ref_module)
        JL_factor = max(0.5,min(1.0,np.random.normal(loc=1.0, scale=0.03)))
        J01_factor = max(1.0,np.random.normal(loc=1.0, scale=0.5))
        for cell in module.cells:
            cell.set_JL(cell.JL() * JL_factor * max(0.5,min(1.0,np.random.normal(loc=1.0, scale=0.01))))
            cell.set_J01(cell.J01() * J01_factor * max(1.0,np.random.normal(loc=1.0, scale=0.2)))
        modules.append(module)

    string = CircuitGroup(modules)

    if display:
        string.plot(title="String Cell I-V Curve")
        string.show()

    return string

if __name__ == "__main__": 
    string = main(display=False)
    run_record_or_test(string, this_file_prefix="a04")
