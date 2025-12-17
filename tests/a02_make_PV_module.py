from PV_Circuit_Model.device import *
from PV_Circuit_Model.device_analysis import *
import a01_make_solar_cell as example1
from utilities import *

def make_device(display=False):
    cell = example1.make_device()

    # butterfly module layout
    n_cells = [22,6]
    num_cells_per_halfstring = n_cells[0]
    num_half_strings = n_cells[1]

    cells = [circuit_deepcopy(cell) for _ in range(num_half_strings*num_cells_per_halfstring)]
    module = make_butterfly_module(cells, num_strings=num_half_strings // 2, num_cells_per_halfstring=num_cells_per_halfstring)
    if display:
        # draw module cells layout
        module.draw_cells(show_names=True)
        # draw its circuit model representation
        module.draw(title="Module model")
        # plot its IV curve
        module.plot(title="Module I-V Curve")
        module.show()
        # write out its constituent parts and values
        print(module)

    return module

def run_test(display=False,pytest_mode=False):
    device = make_device(display=display)
    device = record_or_compare_artifact(device, this_file_prefix="a02",pytest_mode=pytest_mode)
    run_record_or_test(device, this_file_prefix="a02",pytest_mode=pytest_mode)

if __name__ == "__main__": 
    run_test(display=True)

