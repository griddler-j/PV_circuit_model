# IV_jobs.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False

from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
from cpython.ref cimport PyObject

from tqdm import tqdm
import warnings
import sys
from pathlib import Path
from PV_Circuit_Model import ivkernel

cdef bint _PARALLEL_MODE = True

def set_parallel_mode(enabled: bool):
    global _PARALLEL_MODE
    _PARALLEL_MODE = bool(enabled)

cdef class IV_Job_Heap:
    cdef public list components
    cdef list min_child_id
    cdef Py_ssize_t job_done_index
    cdef Py_ssize_t n_components

    def __cinit__(self, object circuit_component):
        # initialize pointers to NULL so __dealloc__ is safe
        self.n_components   = 0
        self.components = [circuit_component]
        self.min_child_id = [-1]
        self.job_done_index = 1  # will reset in build()

    def __init__(self, object circuit_component):
        # build the heap structure
        self.build()

    cpdef void build(self):
        cdef Py_ssize_t pos = 0
        cdef Py_ssize_t child_idx
        cdef object circuit_component, subgroups, element

        while pos < len(self.components):
            circuit_component = self.components[pos]
            if circuit_component._type_number >= 5: # is circuitgroup
                for element in circuit_component.subgroups:
                    child_idx = len(self.components)
                    self.components.append(element)
                    self.min_child_id.append(-1)
                    if self.min_child_id[pos] == -1 or child_idx < self.min_child_id[pos]:
                        self.min_child_id[pos] = child_idx
            pos += 1
        self.n_components = len(self.components)
        self.job_done_index = self.n_components

    cpdef list get_runnable_iv_jobs(self, bint forward=True):
        cdef list runnable = []
        cdef Py_ssize_t start_job_index = self.job_done_index
        cdef Py_ssize_t i, n
        cdef int child_min
        cdef Py_ssize_t min_id

        if forward:
            # walk backward until a node that depends on a future job
            i = start_job_index - 1
            while i >= 0:
                child_min = self.min_child_id[i]
                if child_min != -1 and child_min < start_job_index:
                    break
                self.job_done_index = i
                if self.components[i].IV_V is None:
                    runnable.append(self.components[i])
                i -= 1
        else:
            n = self.n_components
            # sentinel: larger than any valid index
            min_id = n + 100

            i = start_job_index
            while i < n and i < min_id:
                child_min = self.min_child_id[i]
                if child_min != -1 and child_min < min_id:
                    min_id = child_min
                self.job_done_index = i + 1
                runnable.append(self.components[i])
                i += 1

        return runnable

    cpdef void reset(self, bint forward=True):
        if forward:
            self.job_done_index = self.n_components
        else:
            self.job_done_index = 0

    cpdef void set_operating_point(self, V=None, I=None):
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

    cpdef void run_IV(self, bint refine_mode=False):
        cdef bint parallel = False
        if _PARALLEL_MODE and self.components[0].max_num_points is not None:
            parallel = True
        self.reset()
        pbar = None
        if self.job_done_index > 100000:
            pbar = tqdm(total=self.job_done_index, desc="Processing the circuit hierarchy: ")

        while self.job_done_index > 0:
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs()
            if components_:
                ivkernel.run_multiple_jobs(components_, refine_mode=refine_mode, parallel=parallel)
            if pbar is not None:
                pbar.update(job_done_index_before - self.job_done_index)

        if pbar is not None:
            pbar.close()

    def refine_IV(self):
        self.components[0].null_all_IV(max_num_pts_only=True)
        self.run_IV(refine_mode=True)
