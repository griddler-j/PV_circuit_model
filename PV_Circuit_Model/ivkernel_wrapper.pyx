# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
# distutils: language = c++
# distutils: sources = ivkernel.cpp

import numpy as np
cimport numpy as np
from cython cimport nogil
from libc.stdlib cimport malloc, free 
from libc.string cimport memset
import time

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
        int has_photon_coupling
        double op_pt_V
        int refine_mode
        int max_num_points
        double area
        int abs_max_num_points 
        const double* circuit_element_parameters
        double* out_V
        double* out_I
        int* out_len

    double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, int parallel) nogil

    void pin_to_p_cores_only()

def pin_to_p_cores_only_():
    pin_to_p_cores_only()

def run_multiple_jobs(components,refine_mode=False,parallel=False):

    parallel_ = 0
    if parallel: parallel_ = 1
    cdef Py_ssize_t n_jobs = len(components)

    cdef int PARAMS_LEN = 8
    params_all = np.zeros((n_jobs, PARAMS_LEN), dtype=np.float64)
    cdef np.float64_t[:, ::1] mv_params_all = params_all

    # --- allocate IVJobDesc array ---
    cdef IVJobDesc* jobs_c = <IVJobDesc*> malloc(n_jobs * sizeof(IVJobDesc))
    if jobs_c == NULL:
        raise MemoryError()
    memset(jobs_c, 0, n_jobs * sizeof(IVJobDesc))

    # out_len for each job
    cdef np.ndarray[np.int32_t, ndim=1] out_len_array = np.empty(n_jobs, dtype=np.int32)
    cdef np.int32_t[::1] mv_out_len = out_len_array
    cdef int* c_out_len_all = <int*>&mv_out_len[0]

    # ---- count total children / pc-children to allocate IVView buffers ----
    cdef Py_ssize_t total_children = 0
    cdef Py_ssize_t i, j

    for i in range(n_jobs):
        circuit_component = components[i]
        if circuit_component._type_number>=5:
            total_children += len(circuit_component.subgroups)

    cdef IVView* children_views = <IVView*> malloc(total_children * sizeof(IVView))
    if children_views == NULL:
        free(jobs_c)
        raise MemoryError()

    # This list keeps all numpy buffers alive until we return from this function
    memset(children_views, 0, total_children * sizeof(IVView))

    cdef IVView* pc_children_views = <IVView*> malloc(total_children * sizeof(IVView))
    if pc_children_views == NULL:
        free(jobs_c)
        free(children_views)
        raise MemoryError()
    memset(pc_children_views, 0, total_children * sizeof(IVView))

    # Cython views
    cdef np.float64_t[::1] mv_child_v
    cdef np.float64_t[::1] mv_child_pc_v
    cdef np.float64_t[::1] mv_child_i
    cdef np.float64_t[::1] mv_child_pc_i
    cdef np.float64_t[::1] mv_params
    cdef np.float64_t[::1] mv_outV
    cdef np.float64_t[::1] mv_outI

    cdef int circuit_component_type_number, type_number
    cdef int n_children, Ni
    cdef int abs_max_num_points
    cdef double area

    cdef Py_ssize_t child_base = 0
    cdef double kernel_ms
    cdef int olen
    cdef double tmp
    cdef np.float64_t[::1] mv_big_v 
    cdef np.float64_t[::1] mv_big_i 
    cdef double* base 
    cdef Py_ssize_t stride_job 
    cdef Py_ssize_t offset
    cdef int sum_abs_max_num_points

    try:
        t1 = time.time()
        sum_abs_max_num_points = 0
        for i in range(n_jobs):
            circuit_component = components[i]
            subgroups = None
            if circuit_component._type_number>=5:
                subgroups = circuit_component.subgroups
            circuit_component_type_number = circuit_component._type_number  # default CircuitGroup

            # ----- build circuit_element_parameters (matches your C++ expectations) -----
            mv_params = mv_params_all[i]  # shape (PARAMS_LEN,)
            jobs_c[i].circuit_element_parameters = &mv_params[0]
            max_I = circuit_component.max_I
            if not max_I:
                max_I = 0.2
            if circuit_component_type_number == 0:      # CurrentSource
                mv_params[0] = circuit_component.IL
            elif circuit_component_type_number == 1:    # Resistor
                mv_params[0] = circuit_component.cond
            elif circuit_component_type_number in (2, 3):  # diodes
                mv_params[0] = circuit_component.I0
                mv_params[1] = circuit_component.n
                mv_params[2] = circuit_component.VT
                mv_params[3] = circuit_component.V_shift
                mv_params[4] = max_I
            elif circuit_component_type_number == 4:    # Intrinsic_Si_diode
                base_type_number = 0.0  # p
                try:
                    if circuit_component.base_type == "n":
                        base_type_number = 1.0
                except AttributeError:
                    pass

                mv_params[0] = circuit_component.base_doping
                mv_params[1] = 1.0
                mv_params[2] = circuit_component.VT
                mv_params[3] = circuit_component.base_thickness
                mv_params[4] = max_I
                mv_params[5] = circuit_component.ni
                mv_params[6] = base_type_number
            else:
                # CircuitGroup or unknown
                pass 

            # ----- scalar fields -----
            area = 1.0
            if circuit_component._type_number==6 and circuit_component.area is not None:
                area = circuit_component.area
            max_num_points = circuit_component.max_num_points
            if not max_num_points:
                max_num_points = -1
            # ----- fill IVJobDesc scalars -----
            jobs_c[i].connection = -1
            if circuit_component._type_number >= 5:
                connection = circuit_component.connection
                jobs_c[i].connection = 0  # series default
                if connection == "parallel":
                    jobs_c[i].connection = 1

            jobs_c[i].circuit_component_type_number = circuit_component_type_number
            jobs_c[i].max_num_points     = max_num_points
            jobs_c[i].area               = area
            jobs_c[i].circuit_element_parameters = &mv_params[0]
            if refine_mode:
                jobs_c[i].refine_mode = 1
            else:
                jobs_c[i].refine_mode = 0

            # ----- children_IVs → IVView[] (zero-copy views) -----
            abs_max_num_points = 0
            n_children = 0
            if subgroups:
                n_children = len(subgroups)

            jobs_c[i].n_children = n_children
            if n_children > 0:
                jobs_c[i].children_IVs = children_views + child_base
            else:
                jobs_c[i].children_IVs = <IVView*> 0

            for j in range(n_children):
                element = subgroups[j]
                type_number = element._type_number

                children_views[child_base + j].type_number = type_number
                # ensure IV_table is C-contiguous float64 (2, Ni)
                mv_child_v = element.IV_V
                mv_child_i = element.IV_I
                Ni = mv_child_v.shape[0]
                abs_max_num_points += Ni
                children_views[child_base + j].V           = &mv_child_v[0]
                children_views[child_base + j].I           = &mv_child_i[0]
                children_views[child_base + j].length      = Ni

            # ----- photon-coupled children → IVView[] -----
            if n_children > 0:
                jobs_c[i].children_pc_IVs = pc_children_views + child_base
            else:
                jobs_c[i].children_pc_IVs = <IVView*> 0

            abs_max_num_points_multipier = 1
            for j in range(n_children):
                element_area = 0
                element = subgroups[j]
                Ni = 0
                if element._type_number == 6: # cell
                    photon_coupling_diodes = element.photon_coupling_diodes
                    if photon_coupling_diodes and len(photon_coupling_diodes)>0:
                        abs_max_num_points_multipier += 1
                        mv_child_pc_v = photon_coupling_diodes[0].IV_V
                        mv_child_pc_i = photon_coupling_diodes[0].IV_I
                        element_area = element.area
                        Ni = mv_child_pc_v.shape[0]
                    if Ni > 0:
                        pc_children_views[child_base + j].V      = &mv_child_pc_v[0]
                        pc_children_views[child_base + j].I      = &mv_child_pc_i[0]
                        pc_children_views[child_base + j].length = Ni
                    else:
                        pc_children_views[child_base + j].V      = <const double*> 0
                        pc_children_views[child_base + j].I      = <const double*> 0
                        pc_children_views[child_base + j].length = 0
                    pc_children_views[child_base + j].scale       = element_area

            if abs_max_num_points_multipier == 1:
                jobs_c[i].has_photon_coupling = 0;
            else:
                jobs_c[i].has_photon_coupling = 1;

            abs_max_num_points = abs_max_num_points_multipier*abs_max_num_points

            child_base += n_children

            if circuit_component_type_number==0: # current source
                abs_max_num_points = 1
            elif circuit_component_type_number==1: # resistor
                abs_max_num_points = 2
            elif circuit_component_type_number>=2 and circuit_component_type_number<=4: # diode
                tmp = 100.0 / 0.2 * max_I + 5.0
                abs_max_num_points = <int>(tmp + 0.999999)  # cheap ceil

            abs_max_num_points = max(abs_max_num_points, max_num_points)
            abs_max_num_points = int(abs_max_num_points)

            jobs_c[i].abs_max_num_points = abs_max_num_points
            op_pt_V = 0
            op_pt = circuit_component.operating_point
            if op_pt:
                op_pt_V = op_pt[0]
            jobs_c[i].op_pt_V = op_pt_V 

            # ----- allocate per-job output buffer (2 x abs_max_num_points) -----
            sum_abs_max_num_points += abs_max_num_points

        big_out_V = np.empty(sum_abs_max_num_points, dtype=np.float64)
        big_out_I = np.empty(sum_abs_max_num_points, dtype=np.float64)
        mv_big_v = big_out_V  # anchors the array
        mv_big_i = big_out_I  # anchors the array
        base_v = &mv_big_v[0]              # base pointer into contiguous buffer
        base_i = &mv_big_i[0]              # base pointer into contiguous buffer

        offset = 0
        for i in range(n_jobs):
            jobs_c[i].out_V   = base_v + offset
            jobs_c[i].out_I   = base_i + offset 
            jobs_c[i].out_len = &c_out_len_all[i]
            offset += jobs_c[i].abs_max_num_points

        packing_time = time.time()-t1
        # ----- call C++ batched kernel (no Python inside) -----
        with nogil:
            kernel_ms = combine_iv_jobs_batch(<int> n_jobs, jobs_c, parallel_)

        # ----- unpack outputs -----
        t1 = time.time()
        offset = 0
        for i in range(n_jobs):
            circuit_component = components[i]
            olen = c_out_len_all[i]
            if olen < 0:
                raise ValueError(f"Negative out_len for job {i}")
            circuit_component.IV_V = big_out_V[offset:offset+olen]
            circuit_component.IV_I = big_out_I[offset:offset+olen]
            offset += jobs_c[i].abs_max_num_points
        unpacking_time = time.time()-t1

    finally:
        free(jobs_c)
        free(children_views)
        free(pc_children_views)

    return kernel_ms, packing_time, unpacking_time

