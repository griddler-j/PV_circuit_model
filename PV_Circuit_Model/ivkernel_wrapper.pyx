# cython: language_level=3
# distutils: language = c++
# distutils: sources = ivkernel.cpp

import numpy as np
cimport numpy as np
from cython cimport nogil
from libc.stdlib cimport malloc, free 
from libc.string cimport memset

cdef extern from "ivkernel.h":

    cdef struct IVView:
        const double* V      # pointer to V array
        const double* I      # pointer to I array
        int length           # Ni
        double scale
        int type_number

    cdef struct IVJobDesc:
        int connection
        int circuit_component_type_number
        int n_children
        const IVView* children_IVs
        const IVView* children_pc_IVs
        IVView dark_IV
        double total_IL
        double cap_current
        int max_num_points
        double area
        int abs_max_num_points
        int all_children_are_CircuitElement   # maps to C++ bool
        const double* circuit_element_parameters
        double* out_V
        double* out_I
        int* out_len

    double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs) nogil

def run_multiple_jobs(jobs):
    cdef Py_ssize_t n_jobs = len(jobs)
    if n_jobs == 0:
        return [], 0.0

    # --- allocate IVJobDesc array ---
    cdef IVJobDesc* jobs_c = <IVJobDesc*> malloc(n_jobs * sizeof(IVJobDesc))
    if jobs_c == NULL:
        raise MemoryError()
    memset(jobs_c, 0, n_jobs * sizeof(IVJobDesc))

    # out_len for each job
    cdef np.ndarray[np.int32_t, ndim=1] out_len_array = np.empty(n_jobs, dtype=np.int32)
    cdef np.int32_t[::1] mv_out_len = out_len_array
    cdef int* c_out_len_all = <int*>&mv_out_len[0]

    # keep output IV arrays alive
    out_IV_list = [None] * n_jobs

    # type mapping (same semantics as before)
    cdef dict type_numbers = {
        "CurrentSource": 0,
        "Resistor": 1,
        "Diode": 2,
        "ForwardDiode": 2,
        "PhotonCouplingDiode": 2,
        "ReverseDiode": 3,
        "Intrinsic_Si_diode": 4,
        "CircuitGroup": 5,
    }

    # ---- count total children / pc-children to allocate IVView buffers ----
    cdef Py_ssize_t total_children = 0
    cdef Py_ssize_t i, j

    for i in range(n_jobs):
        circuit_component = jobs[i]["circuit_component"]
        if hasattr(circuit_component, "subgroups"):
            total_children += len(circuit_component.subgroups)

    cdef IVView* children_views = <IVView*> malloc(total_children * sizeof(IVView))
    if children_views == NULL:
        free(jobs_c)
        raise MemoryError()

    # This list keeps all numpy buffers alive until we return from this function
    owned_buffers = []   # type: list
    memset(children_views, 0, total_children * sizeof(IVView))

    cdef IVView* pc_children_views = <IVView*> malloc(total_children * sizeof(IVView))
    if pc_children_views == NULL:
        free(jobs_c)
        free(children_views)
        raise MemoryError()
    memset(pc_children_views, 0, total_children * sizeof(IVView))

    # Cython views
    cdef np.float64_t[:, ::1] mv_child
    cdef np.float64_t[:, ::1] mv_child_pc
    cdef np.float64_t[:, ::1] mv_dark
    cdef np.float64_t[::1] mv_params
    cdef np.float64_t[::1] mv_outV
    cdef np.float64_t[::1] mv_outI

    cdef int circuit_component_type_number, type_number
    cdef int n_children, Ni
    cdef int abs_max_num_points
    cdef double total_IL, cap_current, area
    cdef int max_num_points
    cdef int all_children_are_CircuitElement

    cdef Py_ssize_t child_base = 0
    cdef double kernel_ms
    cdef int olen

    try:
        for i in range(n_jobs):
            job = jobs[i]
            circuit_component = job["circuit_component"]
            type_name = type(circuit_component).__name__

            # ----- top-level component type number -----
            circuit_component_type_number = 5  # default CircuitGroup
            if type_name in type_numbers:
                circuit_component_type_number = type_numbers[type_name]

            # ----- build circuit_element_parameters (matches your C++ expectations) -----
            max_I = 0.2
            if hasattr(circuit_component, "max_I"):
                max_I = circuit_component.max_I

            if circuit_component_type_number == 0:      # CurrentSource
                params = np.array([circuit_component.IL], dtype=np.float64)
            elif circuit_component_type_number == 1:    # Resistor
                params = np.array([circuit_component.cond], dtype=np.float64)
            elif circuit_component_type_number in (2, 3):  # diodes
                params = np.array([
                    circuit_component.I0,
                    circuit_component.n,
                    circuit_component.VT,
                    circuit_component.V_shift,
                    max_I,
                ], dtype=np.float64)
            elif circuit_component_type_number == 4:    # Intrinsic_Si_diode
                base_type_number = 0.0  # p
                try:
                    if circuit_component.base_type == "n":
                        base_type_number = 1.0
                except AttributeError:
                    pass
                params = np.array([
                    circuit_component.base_doping,1,circuit_component.VT,
                    circuit_component.base_thickness,
                    max_I,
                    circuit_component.ni,
                    base_type_number,
                ], dtype=np.float64)
            else:
                # CircuitGroup or unknown
                params = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

            owned_buffers.append(params)
            params = np.ascontiguousarray(params, dtype=np.float64)
            mv_params = params
            

            # ----- scalar fields -----
            total_IL = 0.0
            area = 1.0
            if hasattr(circuit_component,"shape") and hasattr(circuit_component,"area") and circuit_component.area is not None:
                area = circuit_component.area
            cap_current = -1.0
            if hasattr(circuit_component,"cap_current"):
                cap_current = circuit_component.cap_current if circuit_component.cap_current is not None else -1.0
            max_num_points = -1
            if hasattr(circuit_component,"max_num_points"):
                max_num_points = circuit_component.max_num_points if circuit_component.max_num_points is not None else -1
            abs_max_num_points = 0

            # ----- fill IVJobDesc scalars -----
            jobs_c[i].connection = -1
            if hasattr(circuit_component, "connection"):
                jobs_c[i].connection = 0  # series default
                if circuit_component.connection == "parallel":
                    jobs_c[i].connection = 1

            jobs_c[i].circuit_component_type_number = circuit_component_type_number
            jobs_c[i].cap_current        = cap_current
            jobs_c[i].max_num_points     = max_num_points
            jobs_c[i].area               = area
            jobs_c[i].circuit_element_parameters = &mv_params[0]

            # ----- children_IVs → IVView[] (zero-copy views) -----
            n_children = 0
            all_children_are_CircuitElement = 0
            if hasattr(circuit_component,"subgroups"):
                n_children = len(circuit_component.subgroups)
                all_children_are_CircuitElement = 1

            jobs_c[i].n_children = n_children
            if n_children > 0:
                jobs_c[i].children_IVs = children_views + child_base
            else:
                jobs_c[i].children_IVs = <IVView*> 0

            for j in range(n_children):
                type_number = 5
                cname = type(circuit_component.subgroups[j]).__name__
                if cname in type_numbers:
                    type_number = type_numbers[cname]
                else: # not CircuitElement
                    all_children_are_CircuitElement = 0

                children_views[child_base + j].type_number = type_number
                if type_number==0: # current source
                    total_IL -= circuit_component.subgroups[j].IL
                    children_views[child_base + j].V      = <const double*> 0
                    children_views[child_base + j].I      = <const double*> 0
                    children_views[child_base + j].length = 0
                else:
                    # ensure IV_table is C-contiguous float64 (2, Ni)
                    arr = circuit_component.subgroups[j].IV_table
                    if (not arr.flags["C_CONTIGUOUS"]) or (arr.dtype != np.float64):
                        arr = np.ascontiguousarray(arr, dtype=np.float64)
                        circuit_component.subgroups[j].IV_table = arr
                    mv_child = circuit_component.subgroups[j].IV_table  
                    Ni = mv_child.shape[1]
                    abs_max_num_points += Ni
                    children_views[child_base + j].V           = &mv_child[0, 0]
                    children_views[child_base + j].I           = &mv_child[1, 0]
                    children_views[child_base + j].length      = Ni

            jobs_c[i].total_IL = total_IL
            jobs_c[i].all_children_are_CircuitElement = all_children_are_CircuitElement

            # ----- photon-coupled children → IVView[] -----
            if n_children > 0:
                jobs_c[i].children_pc_IVs = pc_children_views + child_base
            else:
                jobs_c[i].children_pc_IVs = <IVView*> 0

            for j in range(n_children):
                element_area = 0
                element = circuit_component.subgroups[j]
                Ni = 0
                if hasattr(element,"photon_coupling_diodes") and len(element.photon_coupling_diodes)>0:
                    abs_max_num_points += circuit_component.subgroups[j].IV_table.shape[1]
                    # ensure IV_table is C-contiguous float64 (2, Ni)
                    arr = circuit_component.subgroups[j].IV_table
                    if (not arr.flags["C_CONTIGUOUS"]) or (arr.dtype != np.float64):
                        arr = np.ascontiguousarray(arr, dtype=np.float64)
                        circuit_component.subgroups[j].IV_table = arr
                    mv_child_pc = element.photon_coupling_diodes[0].IV_table
                    element_area = element.area
                    Ni = mv_child_pc.shape[1]
                if Ni > 0:
                    pc_children_views[child_base + j].V      = &mv_child_pc[0, 0]
                    pc_children_views[child_base + j].I      = &mv_child_pc[1, 0]
                    pc_children_views[child_base + j].length = Ni
                else:
                    pc_children_views[child_base + j].V      = <const double*> 0
                    pc_children_views[child_base + j].I      = <const double*> 0
                    pc_children_views[child_base + j].length = 0
                pc_children_views[child_base + j].scale       = element_area

            child_base += n_children

            max_num_points_ = np.ceil(100/0.2*max_I) + 5
            abs_max_num_points = max(abs_max_num_points, max_num_points_)
            abs_max_num_points = max(abs_max_num_points, max_num_points)
            abs_max_num_points = int(abs_max_num_points)

            jobs_c[i].abs_max_num_points = abs_max_num_points

            # ----- allocate per-job output buffer (2 x abs_max_num_points) -----
            out_IV = np.empty((2, abs_max_num_points), dtype=np.float64)
            out_IV_list[i] = out_IV
            mv_outV = out_IV[0]
            mv_outI = out_IV[1]

            jobs_c[i].out_V   = &mv_outV[0]
            jobs_c[i].out_I   = &mv_outI[0]
            jobs_c[i].out_len = &c_out_len_all[i]

            # ----- dark_IV -----
            if hasattr(circuit_component,"dark_IV_table") and circuit_component.dark_IV_table is not None:
                # ensure IV_table is C-contiguous float64 (2, Ni)
                arr = circuit_component.dark_IV_table
                if (not arr.flags["C_CONTIGUOUS"]) or (arr.dtype != np.float64):
                    arr = np.ascontiguousarray(arr, dtype=np.float64)
                    circuit_component.dark_IV_table = arr
                mv_dark = circuit_component.dark_IV_table   # (2, Ni_dark)
                Ni = mv_dark.shape[1]
                jobs_c[i].dark_IV.V           = &mv_dark[0, 0]
                jobs_c[i].dark_IV.I           = &mv_dark[1, 0]
                jobs_c[i].dark_IV.length      = Ni
            else:
                jobs_c[i].dark_IV.V           = <const double*> 0
                jobs_c[i].dark_IV.I           = <const double*> 0
                jobs_c[i].dark_IV.length      = 0

        # ----- call C++ batched kernel (no Python inside) -----
        with nogil:
           kernel_ms = combine_iv_jobs_batch(<int> n_jobs, jobs_c)

        # ----- unpack outputs -----
        for i in range(n_jobs):
            circuit_component = jobs[i]["circuit_component"]
            olen = c_out_len_all[i]
            if olen < 0:
                raise ValueError(f"Negative out_len for job {i}")
            circuit_component.IV_table = out_IV_list[i][:, :olen]
            if jobs_c[i].all_children_are_CircuitElement==1:
                circuit_component.dark_IV_table = circuit_component.IV_table.copy()
                circuit_component.dark_IV_table[1,:] -= jobs_c[i].total_IL*jobs_c[i].area

    finally:
        free(jobs_c)
        free(children_views)
        free(pc_children_views)

    return kernel_ms
