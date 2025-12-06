# IV_jobs.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
# cython: cdivision=True, infer_types=True

from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from PV_Circuit_Model import ivkernel
from PV_Circuit_Model.utilities import ParameterSet 
import numpy as np
cimport numpy as np
import time
ctypedef np.float64_t DTYPE_t
np.import_array()

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

cdef bint _PARALLEL_MODE = True
cdef bint _REPORT_UNCERTAINTY = False
cdef int _SUPER_DENSE = 0    # don't change!  for debugging only

solver_env_variables = None
try:
    ParameterSet(name="solver_env_variables",filename=PARAM_DIR / "solver_env_variables.json")
    solver_env_variables = ParameterSet.get_set("solver_env_variables")
    _PARALLEL_MODE = solver_env_variables["_PARALLEL_MODE"]
    _REPORT_UNCERTAINTY = solver_env_variables["_REPORT_UNCERTAINTY"]
except Exception:
    ParameterSet(name="solver_env_variables",data={})
    solver_env_variables.set("_PARALLEL_MODE", _PARALLEL_MODE)
    solver_env_variables.set("_REPORT_UNCERTAINTY", _REPORT_UNCERTAINTY)
    solver_env_variables.set("_SUPER_DENSE", _SUPER_DENSE)

def set_parallel_mode(enabled: bool):
    global _PARALLEL_MODE, solver_env_variables
    _PARALLEL_MODE = bool(enabled)
    solver_env_variables.set("_PARALLEL_MODE", _PARALLEL_MODE)

def set_report_uncertainty(enabled: bool):
    global _REPORT_UNCERTAINTY, solver_env_variables
    _REPORT_UNCERTAINTY = bool(enabled)
    solver_env_variables.set("_REPORT_UNCERTAINTY", _REPORT_UNCERTAINTY)

def set_super_dense(num_points):
    global _SUPER_DENSE
    _SUPER_DENSE = int(num_points)
    solver_env_variables.set("_SUPER_DENSE", _SUPER_DENSE)

cdef class IV_Job_Heap:
    cdef public list components
    cdef list min_child_id
    cdef public object bottom_up_operating_points, timers, errors
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
        self.errors = {"V":0.0,"I":0.0}

    def __init__(self, object circuit_component):
        # build the heap structure
        self.build()

    cpdef void build(self):
        start_time = time.time()
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
        duration = time.time() - start_time
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
                runnable.append(comps[i])
                i += 1

        return runnable

    cpdef void reset(self, bint forward=True):
        if forward:
            self.job_done_index = self.n_components
        else:
            self.job_done_index = 0

    cpdef void set_operating_point(self, V=None, I=None):
        start_time = time.time()

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

        duration = time.time() - start_time
        self.timers["refine"] = duration

    cpdef void get_bottom_up_operating_points(self):
        cdef Py_ssize_t n = self.n_components  # already tracked in your class
        cdef Py_ssize_t i, j, count, n_sub
        cdef double V, I, max_V, min_V, max_I, min_I, area
        cdef double[:, :] bop = self.bottom_up_operating_points
        cdef list comps = self.components
        # Walk bottom-up via indices instead of reversed(self.components)
        for i in range(n - 1, -1, -1):
            component = comps[i]
            area = 1
            if component._type_number < 5:
                # CircuitElement: operating_point is already "exact"
                # component.operating_point is a Python sequence [V, I]
                bop[i, 0] = <double> component.operating_point[0]
                bop[i, 1] = <double> component.operating_point[1]
                component.bottom_up_operating_point = component.operating_point
            else:
                if component._type_number==6: # cell
                    area = component.area
                # CircuitGroup: aggregate children
                V = 0.0
                I = 0.0
                subgroups = component.subgroups
                n_sub = len(subgroups)
                for j in range(n_sub):
                    element = subgroups[j]
                    V_ = <double> element.bottom_up_operating_point[0]
                    I_ = <double> element.bottom_up_operating_point[1]*area
                    V += V_
                    I += I_
                    if j==0 or V_>max_V:
                        max_V = V_
                    if j==0 or V_<min_V:
                        min_V = V_
                    if j==0 or I_>max_I:
                        max_I = I_
                    if j==0 or I_<min_I:
                        min_I = I_

                if component.connection == "series":
                    I /= n_sub
                else:
                    V /= n_sub
                bop[i, 0] = V
                bop[i, 1] = I
                bop[i, 2] = max_V
                bop[i, 3] = min_V
                bop[i, 4] = max_I
                bop[i, 5] = min_I
                component.bottom_up_operating_point = [V,I]
                    
    cpdef void run_IV(self, bint refine_mode=False, interp_method=0, use_existing_grid=False):
        start_time = time.time()
        cdef bint parallel = False
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
                ivkernel.run_multiple_jobs(components_, refine_mode=refine_mode, parallel=parallel, 
                interp_method=interp_method, super_dense=_SUPER_DENSE, use_existing_grid=use_existing_grid)
            if pbar is not None:
                pbar.update(job_done_index_before - self.job_done_index)

        if pbar is not None:
            pbar.close()

        duration = time.time() - start_time
        if refine_mode:
            self.timers["refine"] += duration  # added to the operating point time
        else:
            self.timers["IV"] = duration

    def refine_IV(self):
        self.run_IV(refine_mode=True)
        if _REPORT_UNCERTAINTY:
            start_time = time.time()
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
            duration = time.time() - start_time
            self.timers["bounds"] = duration

    def calc_Kirchoff_law_errors(self):
        cdef double worst_V_error, worst_I_error, largest_V, smallest_V, largest_I, smallest_I
        if hasattr(self.components[0],"refined_IV") and self.components[0].refined_IV:
            for component in self.components:
                if component._type_number>=5: #CircuitGroup
                    for i, element in enumerate(component.subgroups):
                        if i==0:
                            largest_V = element.bottom_up_operating_point[0]
                            smallest_V = element.bottom_up_operating_point[0]
                            largest_I = element.bottom_up_operating_point[1]
                            smallest_I = element.bottom_up_operating_point[1]
                        else:
                            largest_V = max(largest_V,element.bottom_up_operating_point[0])
                            smallest_V = min(smallest_V,element.bottom_up_operating_point[0])
                            largest_I = max(largest_I,element.bottom_up_operating_point[1])
                            smallest_I = min(smallest_I,element.bottom_up_operating_point[1])
                    if component.connection=="series": # require same I
                        worst_I_error = max(worst_I_error, largest_I-smallest_I)
                    else: # require same V
                        worst_V_error = max(worst_V_error, largest_V-smallest_V)
            self.errors["V"] = worst_V_error
            self.errors["I"] = worst_I_error
        
                    
        
