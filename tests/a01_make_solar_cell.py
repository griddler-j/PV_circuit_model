from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
from utilities import *

def main(display=True):
    # desired cell parameters
    width = 18.2 # cm
    Jsc = 0.042 # A/cm2
    Voc = 0.735 # V
    FF = 0.82 
    breakdown_V = -10 # V
    J0_rev = 100e-15 # A/cm2
    Rshunt = 10000 # ohm-cm2
    Rs = 0.3333 # ohm-cm2
    thickness = 150e-4 #um

    # make the solar cell 
    shape, area = wafer_shape(width/2, width)
    J01, J02 = estimate_cell_J01_J02(Jsc,Voc,FF=FF,Rs=Rs,Rshunt=Rshunt,thickness=thickness)
    cell = make_solar_cell(Jsc, J01, J02, Rshunt, Rs, area, shape, breakdown_V, J0_rev, thickness=thickness)

    if display:
        # draw its circuit model representation
        cell.draw(display_value=True,title="Cell Model")
        # plot its IV curve
        cell.plot(title="Cell I-V Curve")
        cell.show()
        # write out its constituent parts and values
        print(cell)

    return cell

if __name__ == "__main__": 
    device = main(display=False)
    device = record_or_compare_artifact(device, this_file_prefix="a01")
    run_record_or_test(device, this_file_prefix="a01")

