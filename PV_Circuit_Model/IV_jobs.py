from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from PV_Circuit_Model.utilities import interp_

_HAVE_IVKERNEL = False
_PARALLEL_MODE = True

def set_parallel_mode(enabled: bool):
    global _PARALLEL_MODE
    _PARALLEL_MODE = bool(enabled)

def _try_import_ivkernel():
    """Try importing the fast extension."""
    try:
        from PV_Circuit_Model import ivkernel
        return ivkernel
    except Exception:
        return None

def _try_autobuild_extension(package_root):
    """
    Programmatically run: python setup.py build_ext --inplace
    """
    setup_py = Path(package_root) / "setup.py"
    if not setup_py.exists():
        warnings.warn("No setup.py found, cannot auto-build ivkernel.", RuntimeWarning)
        return False

    # Temporarily modify sys.argv as if running: setup.py build_ext --inplace
    old_argv = sys.argv
    sys.argv = [str(setup_py), "build_ext", "--inplace"]

    try:
        # Running setup() loads setup.py, runs build_ext, writes the .pyd
        with setup_py.open("r") as f:
            code = compile(f.read(), str(setup_py), "exec")
            exec(
                code,
                {
                    "__name__": "__main__",
                    "__file__": str(setup_py),
                },
            )
        return True

    except Exception as e:
        warnings.warn(f"Programmatic build_ext failed: {e}", RuntimeWarning)
        return False

    finally:
        sys.argv = old_argv


ivkernel = None
# ---------------------------------------------------------------------
# 1) Try normal import
# ---------------------------------------------------------------------
ivkernel = _try_import_ivkernel()

# ---------------------------------------------------------------------
# 2) If missing, try auto-build
# ---------------------------------------------------------------------
if ivkernel is None:
    warnings.warn(
        "Building C++/Cython extension 'ivkernel' (need to be done only once)...",
        RuntimeWarning
    )
    pkg_root = Path(__file__).resolve().parent
    if _try_autobuild_extension(pkg_root):
        # Try import again
        ivkernel = _try_import_ivkernel()
        if ivkernel is not None:
            print("\n\n\nSucceeded building C++/Cython extension 'ivkernel'!")

# ---------------------------------------------------------------------
# 3) If still missing, fall back to Python
# ---------------------------------------------------------------------
if ivkernel is None:
    warnings.warn(
        "The optimized C++/Cython extension 'ivkernel' could not be imported, "
        "even after attempting automatic build.\n"
        "Falling back to pure-Python implementation (much slower and has a tiny numerical difference).",
        RuntimeWarning
    )
    from PV_Circuit_Model.ivkernel_python import build_component_IV_python
    _HAVE_IVKERNEL = False
else:
    _HAVE_IVKERNEL = True

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component):
        self.components = [circuit_component]
        self.children_job_ids = [[]]
        self.job_done_index = len(self.components)
        self.build()
    def build(self):
        pos = 0
        while pos < len(self.components):
            circuit_component = self.components[pos] 
            subgroups = getattr(circuit_component, "subgroups", None)
            if subgroups:
                for element in subgroups:
                    if element.IV_V is None:
                        self.components.append(element)
                        self.children_job_ids.append([])
                        self.children_job_ids[pos].append(len(self.components)-1)
            pos += 1
    def get_runnable_iv_jobs(self,forward=True):
        include_indices = []
        start_job_index = self.job_done_index
        if forward:
            for i in reversed(range(start_job_index)):
                ids = self.children_job_ids[i]
                if len(ids)>0 and min(ids)<start_job_index:
                    break
                self.job_done_index = i
                if self.components[i].IV_V is None:
                    include_indices.append(i)
        else:
            min_id = len(self.components) + 100
            for i in range(start_job_index,len(self.components)):
                if i >= min_id:
                    break
                ids = self.children_job_ids[i]
                if len(ids)>0: min_id = min(min_id, min(ids))
                self.job_done_index = i+1
                include_indices.append(i)
        return [self.components[j] for j in include_indices]
    def reset(self,forward=True):
        if forward:
            self.job_done_index = len(self.components)
        else:
            self.job_done_index = 0
    def set_operating_point(self,V=None,I=None):
        parallel = False
        if _PARALLEL_MODE and self.components[0].max_num_points is not None:
            parallel = True
        self.reset(forward=False)
        pbar = None
        if V is not None:
            self.components[0].operating_point = [V,None]
        else:
            self.components[0].operating_point = [None,I]
        if len(self.components) > 100000:
            pbar = tqdm(total=len(self.components), desc="Processing the circuit hierarchy: ")
        while self.job_done_index < len(self.components):
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs(forward=False)
            if len(components_) > 0:
                if _HAVE_IVKERNEL:
                    ivkernel.run_multiple_operating_points(components_, parallel=parallel)
                else:
                    for component in components_:
                        V = component.operating_point[0]
                        I = component.operating_point[1]
                        if V is not None:
                            component.operating_point[1] = interp_(V,component.IV_V,component.IV_I)
                        elif I is not None:
                            component.operating_point[0] = interp_(I,component.IV_I,component.IV_V)
                        if component._type_number>=5:
                            is_series = False
                            if component.connection=="series":
                                is_series = True
                            for child in component.subgroups:
                                if is_series:
                                    child.operating_point = [None, component.operating_point[1]]
                                else:
                                    child.operating_point = [component.operating_point[0], None]
            if pbar is not None:
                pbar.update(self.job_done_index-job_done_index_before)
        if pbar is not None:
            pbar.close()

    def run_IV(self, refine_mode=False):
        parallel = False
        if _PARALLEL_MODE and self.components[0].max_num_points is not None:
            parallel = True
        self.reset()
        pbar = None
        if self.job_done_index > 100000:
            pbar = tqdm(total=self.job_done_index, desc="Processing the circuit hierarchy: ")
        while self.job_done_index > 0:
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs()
            if len(components_) > 0:
                if _HAVE_IVKERNEL:
                    ivkernel.run_multiple_jobs(components_,refine_mode=refine_mode,parallel=parallel)
                else:
                    for component in components_:
                        build_component_IV_python(component,refine_mode=refine_mode)
            if pbar is not None:
                pbar.update(job_done_index_before-self.job_done_index)
        if pbar is not None:
            pbar.close()
    def refine_IV(self):
        self.run_IV(refine_mode=True)


# Collects a list of job heaps for parallel processing
class IV_Job_Pool():
    def __init__(self,heap_list):
        self.heap_list = heap_list
    def get_runnable_iv_jobs(self):
        grand_list = []
        for heap in self.heap_list:
            grand_list.extend(heap.get_runnable_iv_jobs())
        return grand_list
    def get_total_unfinished_jobs(self):
        total_unfinished_jobs = 0
        for heap in self.heap_list:
            total_unfinished_jobs += heap.job_done_index
        return total_unfinished_jobs
    def reset(self):
        for heap in self.heap_list:
            heap.reset()
    def run_IV(self, refine_mode=False):
        self.reset()
        total_unfinished_jobs = self.get_total_unfinished_jobs()
        pbar = None
        if total_unfinished_jobs > 100000:
            pbar = tqdm(total=total_unfinished_jobs, desc="Processing the circuit hierarchy: ")
        while total_unfinished_jobs > 0:
            job_done_index_before = total_unfinished_jobs
            components_ = self.get_runnable_iv_jobs()
            if len(components_) > 0:
                if _HAVE_IVKERNEL:
                    ivkernel.run_multiple_jobs(components_,refine_mode=refine_mode,parallel=True)
                else:
                    for component in components_:
                        build_component_IV_python(component,refine_mode=refine_mode)
            total_unfinished_jobs = self.get_total_unfinished_jobs()
            if pbar is not None:
                pbar.update(job_done_index_before-total_unfinished_jobs)
        if pbar is not None:
            pbar.close()
    def refine_IV(self):
        self.run_IV(refine_mode=True)