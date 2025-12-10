from PV_Circuit_Model.cell_analysis import *
from pathlib import Path
from utilities import *
from PV_Circuit_Model.ivkernel import set_parallel_mode
import time

sample_names = ["cell","ref_module","module","ref_string","string","parallel_strings2","block_4","block_8","block_16","block_32","block_64","block_100"]

THIS_DIR = Path(__file__).resolve().parent

for sample_name in tqdm(sample_names):
    sample = ParamSerializable.restore_from_bson(THIS_DIR / Path(sample_name + ".bson"))
    for parallel_mode_ in [True,False]:
        set_parallel_mode(parallel_mode_)
        times = []
        for reps in range(20):
            sample.null_all_IV()
            t1 = time.perf_counter()
            sample.get_Pmax()
            t2 = time.perf_counter()
            print(f"Parallel {parallel_mode_}: time = {t2-t1}")
            times.append(t2-t1)
        sample.save_IV_curve(THIS_DIR / Path(sample_name + f"_IV_parallel={parallel_mode_}.txt"))
        sample.save_solver_summary(THIS_DIR / Path(sample_name + f"_solver_summary_parallel={parallel_mode_}.txt"))
        np.savetxt(THIS_DIR / Path(sample_name + f"_times_parallel={parallel_mode_}.txt"),np.array(times))

    # assert(1==0)

