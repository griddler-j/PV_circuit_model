from PV_Circuit_Model.cell import *
from PV_Circuit_Model.cell_analysis import *
from PV_Circuit_Model.multi_junction_cell import *
import a01_make_solar_cell as example1
from utilities import *

def main(display=True):
    bottom_cell = example1.main(display=False)
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

if __name__ == "__main__": 
    device = main(display=False)
    run_record_or_test(device, this_file_prefix="a03")
