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
    cdef Py_ssize_t job_done_index
    cdef Py_ssize_t n_components

    # C child layout
    cdef Py_ssize_t* child_offsets
    cdef int*        child_counts
    cdef int*        child_ids
    cdef int*        min_child_id
    cdef Py_ssize_t  total_children

    def __cinit__(self, object circuit_component):
        # initialize pointers to NULL so __dealloc__ is safe
        self.child_offsets = NULL
        self.child_counts  = NULL
        self.child_ids     = NULL
        self.min_child_id  = NULL
        self.total_children = 0
        self.n_components   = 0

        self.components = [circuit_component]
        self.job_done_index = 1  # will reset in build()

    def __init__(self, object circuit_component):
        # build the heap structure
        self.build()

    def __dealloc__(self):
        if self.child_offsets != NULL:
            free(<void*> self.child_offsets)
        if self.child_counts != NULL:
            free(<void*> self.child_counts)
        if self.child_ids != NULL:
            free(<void*> self.child_ids)
        if self.min_child_id != NULL:
            free(<void*> self.min_child_id)

    cpdef void build(self):
        cdef Py_ssize_t pos = 0
        cdef object circuit_component, subgroups, element

        # First, BFS and collect children as Python list-of-lists
        cdef list children_lists = [[]]  # children_lists[i] -> list of child indices

        while pos < len(self.components):
            circuit_component = self.components[pos]
            if circuit_component._type_number >= 5: # is circuitgroup
                for element in circuit_component.subgroups:
                    self.components.append(element)
                    children_lists.append([])
                    children_lists[pos].append(len(self.components) - 1)
            pos += 1

        self.n_components = len(self.components)
        self.job_done_index = self.n_components

        # Free any old C arrays if build() is called again
        if self.child_offsets != NULL:
            free(<void*> self.child_offsets)
            self.child_offsets = NULL
        if self.child_counts != NULL:
            free(<void*> self.child_counts)
            self.child_counts = NULL
        if self.child_ids != NULL:
            free(<void*> self.child_ids)
            self.child_ids = NULL
        if self.min_child_id != NULL:
            free(<void*> self.min_child_id)
            self.min_child_id = NULL

        # Flatten children_lists into C arrays
        self._build_child_arrays(children_lists)

    cdef void _build_child_arrays(self, list children_lists):
        cdef Py_ssize_t i, j, n = self.n_components
        cdef list lst
        cdef Py_ssize_t total = 0
        cdef int child_idx, minv
        cdef Py_ssize_t offset

        # Compute total number of children
        for i in range(n):
            lst = children_lists[i]
            total += len(lst)

        self.total_children = total

        # Allocate arrays
        self.child_offsets = <Py_ssize_t*> malloc(n * sizeof(Py_ssize_t))
        self.child_counts  = <int*>        malloc(n * sizeof(int))
        self.child_ids     = <int*>        malloc(total * sizeof(int))
        self.min_child_id  = <int*>        malloc(n * sizeof(int))

        if (self.child_offsets == NULL or self.child_counts == NULL or
            self.child_ids == NULL or self.min_child_id == NULL):
            if self.child_offsets != NULL: free(<void*> self.child_offsets)
            if self.child_counts  != NULL: free(<void*> self.child_counts)
            if self.child_ids     != NULL: free(<void*> self.child_ids)
            if self.min_child_id  != NULL: free(<void*> self.min_child_id)
            self.child_offsets = self.child_counts = NULL
            self.child_ids = self.min_child_id = NULL
            raise MemoryError()

        # Fill arrays and precompute min_child_id
        offset = 0
        for i in range(n):
            lst = children_lists[i]
            self.child_offsets[i] = offset
            self.child_counts[i]  = <int> len(lst)

            if len(lst) == 0:
                self.min_child_id[i] = -1
            else:
                # copy children into flat array
                minv = 2147483647  # big int (2^31-1)
                for j in range(len(lst)):
                    child_idx = <int> lst[j]
                    self.child_ids[offset + j] = child_idx
                    if child_idx < minv:
                        minv = child_idx
                self.min_child_id[i] = minv

            offset += len(lst)

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
