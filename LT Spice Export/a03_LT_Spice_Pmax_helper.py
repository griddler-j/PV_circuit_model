import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PV_Circuit_Model.cell_analysis import *

THIS_DIR = Path(__file__).resolve().parent
file = "cell_scan2.txt"

V, I = np.loadtxt(THIS_DIR / Path(file), skiprows=1, unpack=True)
P = V*I
plt.plot(V,P)
plt.scatter(V,P)
plt.show()

Pmax, Vmp, Imp = get_Pmax(np.array([V,-I]),return_op_point=True)

print(Pmax)
print(Vmp)
print(Imp)
print(V.size)

V_range_ = V[-1]-V[0]
print(f"{Vmp-V_range_/50} {Vmp+V_range_/50} {V_range_/25/100}")





