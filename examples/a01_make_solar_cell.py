from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *

def main(display=True):
    cell = quick_solar_cell()

    if display:
        # draw the wafer shape
        print(cell.area)
        cell.draw_cells()
        cell.show()
        # draw its circuit model representation
        cell.draw(display_value=True,title="Cell Model")
        # plot its IV curve
        cell.plot(title="Cell I-V Curve")
        cell.show()
        # write out its constituent parts and values
        print(cell)

    return cell

if __name__ == "__main__": 
    main()
