from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from PV_Circuit_Model.utilities import interp_

_FORCE_PYTHON = True

def _try_import_cython():
    """Try importing the fast extension."""
    try:
        from PV_Circuit_Model.IV_jobs_cython import IV_Job_Heap, set_parallel_mode
        return IV_Job_Heap, set_parallel_mode
    except Exception:
        return None, None

def _try_autobuild_extension(package_root):
    """
    Programmatically run: python setup.py build_ext --inplace
    """
    setup_file_names = ["setup_ivkernel.py","setup_IV_jobs.py"]
    for file in setup_file_names:
        setup_py = Path(package_root) / file
        if not setup_py.exists():
            warnings.warn("No setup.py found, cannot auto-build cython/c++.", RuntimeWarning)
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
        except Exception as e:
            warnings.warn(f"Programmatic build_ext failed: {e}", RuntimeWarning)
            return False

        finally:
            sys.argv = old_argv
    return True

IV_Job_Heap = None
set_parallel_mode = None
if not _FORCE_PYTHON:
    # ---------------------------------------------------------------------
    # 1) If cython import fails, try auto-build
    # ---------------------------------------------------------------------
    IV_Job_Heap, set_parallel_mode = _try_import_cython()
    if IV_Job_Heap is None:
        warnings.warn(
            "Building C++/Cython extensions (need to be done only once)...",
            RuntimeWarning
        )
        pkg_root = Path(__file__).resolve().parent
        if _try_autobuild_extension(pkg_root):
            # Try import again
            IV_Job_Heap, set_parallel_mode = _try_import_cython()
            if IV_Job_Heap is not None:
                print("\n\n\nSucceeded building C++/Cython extensions!")

# ---------------------------------------------------------------------
# 2) If still missing, fall back to Python
# ---------------------------------------------------------------------
if IV_Job_Heap is None:
    if not _FORCE_PYTHON:
        warnings.warn(
            "The optimized C++/Cython extensions could not be imported, "
            "even after attempting automatic build.\n"
            "Falling back to pure-Python implementation (much slower and has a tiny numerical difference).",
            RuntimeWarning
        )
    from PV_Circuit_Model.ivkernel_python import build_component_IV_python

    def set_parallel_mode(boolean):
        pass 

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
                            current_ = component.operating_point[1]
                            if component._type_number>=5:
                                current_ /= component.area
                            for child in component.subgroups:
                                if is_series:
                                    child.operating_point = [None, current_]
                                else:
                                    child.operating_point = [component.operating_point[0], None]
                if pbar is not None:
                    pbar.update(self.job_done_index-job_done_index_before)
            if pbar is not None:
                pbar.close()
        # def get_bottom_up_operating_points(self):
        #     self.bottom_up_operating_points = []
        #     for component in reversed(self.components):
        #         component

        def run_IV(self, refine_mode=False):
            self.reset()
            pbar = None
            if self.job_done_index > 100000:
                pbar = tqdm(total=self.job_done_index, desc="Processing the circuit hierarchy: ")
            while self.job_done_index > 0:
                job_done_index_before = self.job_done_index
                components_ = self.get_runnable_iv_jobs()
                if len(components_) > 0:
                    for component in components_:
                        build_component_IV_python(component,refine_mode=refine_mode)
                if pbar is not None:
                    pbar.update(job_done_index_before-self.job_done_index)
            if pbar is not None:
                pbar.close()
        def refine_IV(self):
            self.components[0].null_all_IV(max_num_pts_only=True)
            self.run_IV(refine_mode=True)


