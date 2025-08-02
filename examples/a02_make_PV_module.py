from PV_Circuit_Model.cell import *
from PV_Circuit_Model.module import *
from PV_Circuit_Model.cell_analysis import *
import a01_make_solar_cell as example1

def main(display=True):
    cell = example1.main(display=False)

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

if __name__ == "__main__": 
    main()
