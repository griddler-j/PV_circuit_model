from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
import a01_make_solar_cell as example1

def main(display=True):
    module = quick_butterfly_module()
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

if __name__ == "__main__": 
    main()
