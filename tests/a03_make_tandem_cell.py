from PV_Circuit_Model.device import *
from PV_Circuit_Model.device_analysis import *
import a01_make_solar_cell as example1
from utilities import *

def make_device(display=False):
    bottom_cell = example1.make_device()
    bottom_cell.set_JL(19.0e-3)

    Jsc_top_cell = 20.5e-3
    Voc_top_cell = 1.18
    J01_PC = 5e-24

    J01, J02 = estimate_cell_J01_J02(Jsc=Jsc_top_cell,Voc=Voc_top_cell,Si_intrinsic_limit=False)
    top_cell = make_solar_cell(Jsc_top_cell, J01, J02, 
                           area=bottom_cell.area, 
                           Si_intrinsic_limit=False,J01_photon_coupling=J01_PC)
    
    tandem_cell = MultiJunctionCell([bottom_cell,top_cell])

    if display:
        tandem_cell.draw(display_value=True,title="Tandem Cell Model")
        tandem_cell.plot(title="Tandem Cell I-V Curve")
        tandem_cell.show()

    return tandem_cell

def run_test(display=False,pytest_mode=False):
    device = make_device(display=display)
    device = record_or_compare_artifact(device, this_file_prefix="a03",pytest_mode=pytest_mode)
    run_record_or_test(device, this_file_prefix="a03",pytest_mode=pytest_mode)

if __name__ == "__main__": 
    run_test(display=True)
