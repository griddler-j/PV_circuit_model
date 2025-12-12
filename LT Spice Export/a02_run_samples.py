from PV_Circuit_Model.cell_analysis import *
from pathlib import Path
from utilities import *

sample_names = ["cell","ref_module","module","ref_string","string","parallel_strings2"]

THIS_DIR = Path(__file__).resolve().parent

for sample_name in tqdm(sample_names):
    sample = ParamSerializable.restore_from_bson(THIS_DIR / Path("generated samples/" + sample_name + ".bson"))
    # sample = import_ltspice_netlist(THIS_DIR / Path(sample_name + ".net"))
    # sample = ParamSerializable.restore_from_json(THIS_DIR / Path(sample_name + ".json"))
    # sample.save_to_json(THIS_DIR / Path(sample_name + ".json"), critical_fields_only=True)
    sample.get_Pmax()
    sample.save_IV_curve(THIS_DIR / Path(sample_name + "_IV.txt"))
    sample.calc_uncertainty()
    sample.save_solver_summary(THIS_DIR / Path(sample_name + "_solver_summary.txt"))

    # assert(1==0)

