from tqdm import tqdm
import warnings
import sys
from pathlib import Path

_HAVE_IVKERNEL = False

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
        self.components = []
        self.children_job_ids = []
        self.add(circuit_component) # now job_list has one
        self.build()
    def add(self,circuit_component):
        self.components.append(circuit_component)
        self.children_job_ids.append([])
        return len(self.components)-1
    def build(self):
        pos = 0
        while pos < len(self.components):
            circuit_component = self.components[pos] 
            subgroups = getattr(circuit_component, "subgroups", None)
            if subgroups:
                for element in subgroups:
                    if element.IV_V is None:
                        new_job_id = self.add(element)
                        self.children_job_ids[pos].append(new_job_id)
            pos += 1
    def run_IV(self):
        run_iv_jobs(self.components,self.children_job_ids)
    def refine_IV(self):
        run_iv_jobs(self.components,self.children_job_ids, refine_mode=True)

def get_runnable_iv_jobs(components, children_job_ids, job_done_index):
    include_indices = []
    for i in range(job_done_index-1,-1,-1):
        ids = children_job_ids[i]
        if len(ids)>0 and min(ids)<job_done_index:
            return [components[j] for j in include_indices], i+1
        if components[i].IV_V is None:
            include_indices.append(i)
    return [components[j] for j in include_indices], 0

def run_iv_jobs(components, children_job_ids, refine_mode=False):
    if _HAVE_IVKERNEL:
        ivkernel.pin_to_p_cores_only_()
    job_done_index = len(components)
    pbar = None
    if job_done_index > 100000:
        pbar = tqdm(total=job_done_index, desc="Processing the circuit hierarchy: ")
    while job_done_index > 0:
        components_, min_index = get_runnable_iv_jobs(components, children_job_ids, job_done_index)
        if len(components_) > 0:
            if _HAVE_IVKERNEL:
                ivkernel.run_multiple_jobs(components_,refine_mode=refine_mode,parallel=True)
            else:
                for component in components_:
                    build_component_IV_python(component,refine_mode=refine_mode)
        if pbar is not None:
            pbar.update(job_done_index-min_index)
        job_done_index = min_index
    if pbar is not None:
        pbar.close()


