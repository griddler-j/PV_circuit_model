from PV_Circuit_Model.device import *
from PV_Circuit_Model.device_analysis import *
from utilities import *
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
PARENT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

def run_test_for_a05_10(this_file_prefix, this_device_name, display=False,pytest_mode=False):
    filepath = find_latest_file(this_file_prefix+"_result","bson")
    device = Artifact.load(filepath)
    t1 = time.perf_counter()
    device.get_Pmax()
    print(f"Finished in {time.perf_counter()-t1} seconds")
    LT_spice_IV = get_LT_spice_IV(THIS_DIR / "a05_to_10_LT spice results", this_device_name)
    all_pass, Pmax1, Pmax2 = compare_artifact_against_LT_spice(device,LT_spice_IV,pytest_mode=pytest_mode)
    if all_pass:
        print(this_file_prefix + " all pass!")
    if display and Pmax1 is not None:
        plt.plot(device.IV_V,-device.IV_I,label=f"PV Circuit Model (Pmax = {Pmax1}W)",color="orange")
        plt.scatter(LT_spice_IV[0,:],-LT_spice_IV[1,:].T,label=f"LT Spice (Pmax = {Pmax2}W)",color="blue")
        minx = np.min(LT_spice_IV[0,:])
        maxx = np.max(LT_spice_IV[0,:])
        rangex = maxx-minx
        miny = np.min(-LT_spice_IV[1,:])
        maxy = np.max(-LT_spice_IV[1,:])
        rangey = maxy-miny
        plt.xlim((minx-rangex*0.1,maxx+rangex*0.1))
        plt.ylim((miny-rangey*0.1,maxy+rangey*0.1))
        plt.xlabel("V(V)")
        plt.ylabel("I(A)")
        plt.legend()
        plt.show()

def run_test(display=False,pytest_mode=False):
    run_test_for_a05_10("a05","cell",display=display,pytest_mode=pytest_mode)

if __name__ == "__main__": 
    run_test(display=True)
    
            
