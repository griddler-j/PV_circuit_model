# IV_jobs.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
# cython: cdivision=True, infer_types=True

from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from PV_Circuit_Model import ivkernel
from PV_Circuit_Model.ivkernel import solver_env_variables
import numpy as np
cimport numpy as np
import time
ctypedef np.float64_t DTYPE_t
np.import_array()

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

def _pickle_return_none():
    return None

cdef class IV_Job_Heap:
    cdef public list components
    cdef list min_child_id
    cdef public object bottom_up_operating_points, timers
    cdef Py_ssize_t job_done_index
    cdef Py_ssize_t n_components
    
    def __cinit__(self, object circuit_component):
        # initialize pointers to NULL so __dealloc__ is safe
        self.n_components   = 0
        self.components = [circuit_component]
        self.min_child_id = [-1]
        self.bottom_up_operating_points = None
        self.job_done_index = 1  # will reset in build()
        self.timers = {"build":0.0,"IV":0.0,"refine":0.0,"bounds":0.0}

    def __init__(self, object circuit_component):
        # build the heap structure
        self.build()

    def __reduce__(self): # make pickle not complain, just store as None
        return (_pickle_return_none, ())

    def __reduce_ex__(self, protocol):
        return (_pickle_return_none, ())

    cpdef void build(self):
        start_time = time.perf_counter()
        cdef Py_ssize_t pos = 0
        cdef Py_ssize_t child_idx
        cdef object circuit_component, subgroups, element
        cdef list comps = self.components
        cdef list min_child = self.min_child_id

        while pos < len(comps):
            circuit_component = comps[pos]
            if circuit_component._type_number >= 5: # is circuitgroup
                for element in circuit_component.subgroups:
                    child_idx = len(comps)
                    comps.append(element)
                    min_child.append(-1)
                    if min_child[pos] == -1 or child_idx < min_child[pos]:
                        min_child[pos] = child_idx
            pos += 1
        self.n_components = len(comps)
        self.job_done_index = self.n_components
        self.bottom_up_operating_points = np.empty((self.n_components, 6),
                                               dtype=np.float64)
        duration = time.perf_counter() - start_time
        self.timers["build"] = duration

    cpdef list get_runnable_iv_jobs(self, bint forward=True, bint refine_mode=False):
        cdef list comps = self.components
        cdef list min_child = self.min_child_id
        cdef list runnable = []
        cdef Py_ssize_t start_job_index = self.job_done_index
        cdef Py_ssize_t i, n
        cdef int child_min
        cdef Py_ssize_t min_id

        if forward:
            # walk backward until a node that depends on a future job
            i = start_job_index - 1
            while i >= 0:
                child_min = min_child[i]
                if child_min != -1 and child_min < start_job_index:
                    break
                self.job_done_index = i
                # if refine_mode, then don't bother with CircuitElements, but run even if there is IV
                if (refine_mode and child_min>=0) or comps[i].IV_V is None:
                    if comps[i]._type_number==-1: # a user defined circuitelement, run in python immediately
                        comps[i].IV_V = comps[i].get_V_range()
                        comps[i].IV_I = comps[i].calc_I(comps[i].IV_V)
                    else:
                        runnable.append(comps[i])
                i -= 1
        else:
            n = self.n_components
            # sentinel: larger than any valid index
            min_id = n + 100

            i = start_job_index
            while i < n and i < min_id:
                child_min = min_child[i]
                if child_min != -1 and child_min < min_id:
                    min_id = child_min
                self.job_done_index = i + 1
                if comps[i]._type_number==-1: # a user defined circuitelement, run in python immediately
                    if comps[i].parent is not None:
                        if comps[i].parent.connection=="series":
                            target_I = comps[i].operating_point[1]
                            if comps[i].parent._type_number==6: # cell
                                target_I /= comps[i].parent.area
                            comps[i].set_operating_point(I = target_I)
                        else:
                            comps[i].set_operating_point(V = comps[i].parent.operating_point[0])
                else:
                    runnable.append(comps[i])
                i += 1

        return runnable

    cpdef void reset(self, bint forward=True):
        if forward:
            self.job_done_index = self.n_components
        else:
            self.job_done_index = 0

    cpdef void set_operating_point(self, V=None, I=None):
        start_time = time.perf_counter()
        _PARALLEL_MODE = solver_env_variables["_PARALLEL_MODE"]
        cdef bint parallel = False
        if _PARALLEL_MODE and self.components[0].max_num_points is not None:
            parallel = True
        self.reset(forward=False)
        pbar = None

        if V is not None:
            self.components[0].operating_point = [V, None]
        else:
            self.components[0].operating_point = [None, I]

        if self.n_components > 100000:
            pbar = tqdm(total=self.n_components, desc="Processing the circuit hierarchy: ")

        while self.job_done_index < self.n_components:
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs(forward=False)
            if components_:
                ivkernel.run_multiple_operating_points(components_, parallel=parallel)
            if pbar is not None:
                pbar.update(self.job_done_index - job_done_index_before)

        if pbar is not None:
            pbar.close()

        duration = time.perf_counter() - start_time
        self.timers["refine"] = duration
                    
    cpdef bint run_IV(self, bint refine_mode=False, interp_method=0, use_existing_grid=False):
        start_time = time.perf_counter()
        cdef bint parallel = False
        cdef int error_code
        cdef int interp_method_i = <int>interp_method
        cdef bint use_existing_grid_b = <bint>use_existing_grid

        _PARALLEL_MODE = solver_env_variables["_PARALLEL_MODE"]
        _SUPER_DENSE = solver_env_variables["_SUPER_DENSE"]
        if _PARALLEL_MODE and self.components[0].max_num_points is not None:
            parallel = True
        self.reset()
        pbar = None
        if self.job_done_index > 100000:
            pbar = tqdm(total=self.job_done_index, desc="Processing the circuit hierarchy: ")

        while self.job_done_index > 0:
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs(refine_mode=refine_mode)
            if components_:
                error_code = ivkernel.run_multiple_jobs(components_, refine_mode=refine_mode, parallel=parallel, 
                interp_method=interp_method, super_dense=_SUPER_DENSE, use_existing_grid=use_existing_grid)
                if error_code==1: # some Nan numbers in circuit element parameters
                    raise FloatingPointError("Non-finite (NaN/Inf) detected in circuit element parameters")
                if error_code==2: # children have no overlapping current ranges in series connection
                    return False
            if pbar is not None:
                pbar.update(job_done_index_before - self.job_done_index)

        if pbar is not None:
            pbar.close()

        duration = time.perf_counter() - start_time
        if not refine_mode:
            self.timers["IV"] = duration
        return True

    def refine_IV(self):
        if self.components[0].IV_V is not None and self.components[0].operating_point is not None:
            start_time = time.perf_counter()
            self.run_IV(refine_mode=True)
            duration = time.perf_counter() - start_time
            self.timers["refine"] += duration

    def calc_uncertainty(self):
        if self.components[0].IV_V is not None:
            start_time = time.perf_counter()
            self.components[0].IV_V_temp = self.components[0].IV_V.copy()
            self.components[0].IV_I_temp = self.components[0].IV_I.copy()
            self.run_IV(refine_mode=True,interp_method=2,use_existing_grid=True) # get upper bounds of curve 
            self.components[0].IV_V_upper = self.components[0].IV_V.copy()
            self.components[0].IV_I_upper = self.components[0].IV_I.copy()
            self.run_IV(refine_mode=True,interp_method=3,use_existing_grid=True) # get lower bounds of curve 
            self.components[0].IV_V_lower = self.components[0].IV_V.copy()
            self.components[0].IV_I_lower = self.components[0].IV_I.copy()
            self.components[0].IV_V = self.components[0].IV_V_temp.copy()
            self.components[0].IV_I = self.components[0].IV_I_temp.copy()
            del self.components[0].IV_V_temp
            del self.components[0].IV_I_temp
            # but the children are all wrecked
            duration = time.perf_counter() - start_time
            self.timers["bounds"] = duration

    def calc_Kirchoff_law_errors(self):
        cdef double worst_V_error, worst_I_error, largest_V, smallest_V, largest_I, smallest_I
        cdef int has_started
        if self.components[0].refined_IV:
            for component in self.components:
                if component._type_number>=5: #CircuitGroup
                    largest_V = 0
                    smallest_V = 0
                    largest_I = 0
                    smallest_I = 0
                    has_started = 0
                    for i, element in enumerate(component.subgroups):
                        if element._type_number >= 5:
                            if has_started==0:
                                largest_V = element.bottom_up_operating_point[0]
                                smallest_V = element.bottom_up_operating_point[0]
                                largest_I = element.bottom_up_operating_point[1]
                                smallest_I = element.bottom_up_operating_point[1]
                                has_started = 1
                            else:
                                largest_V = max(largest_V,element.bottom_up_operating_point[0])
                                smallest_V = min(smallest_V,element.bottom_up_operating_point[0])
                                largest_I = max(largest_I,element.bottom_up_operating_point[1])
                                smallest_I = min(smallest_I,element.bottom_up_operating_point[1])
                    if component.connection=="series": # require same I
                        worst_I_error = max(worst_I_error, largest_I-smallest_I)
                    else: # require same V
                        worst_V_error = max(worst_V_error, largest_V-smallest_V)
            return worst_V_error, worst_I_error
        
                    
        
