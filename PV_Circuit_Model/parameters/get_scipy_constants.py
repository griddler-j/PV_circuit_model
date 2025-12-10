from scipy import constants as c
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

k = c.Boltzmann            
q = c.elementary_charge   
zero_C = c.zero_Celsius
VT_at_25C = k*(zero_C+25)/q

with open(THIS_DIR / "constants.json","w") as f:
    json.dump({"k":k, "q":q, "zero_C":zero_C, "VT_at_25C": VT_at_25C},f, indent=2)



