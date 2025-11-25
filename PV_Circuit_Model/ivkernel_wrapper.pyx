# cython: language_level=3
# distutils: language = c++
# distutils: sources = ivkernel.cpp

import numpy as np
cimport numpy as np
from cython cimport nogil
from libc.stdlib cimport malloc, free 
from libc.string cimport memset

cdef extern from "ivkernel.h":
    cdef struct IVJobDesc:
        int connection
        int circuit_component_type_number
        int n_children
        const int* children_type_numbers
        const double* children_Vs
        const double* children_Is
        const int* children_offsets
        const int* children_lengths
        int children_Vs_size
        const double* children_pc_Vs
        const double* children_pc_Is
        const int* children_pc_offsets
        const int* children_pc_lengths
        int children_pc_Vs_size
        double total_IL
        double cap_current
        int max_num_points
        double area
        int abs_max_num_points
        const double* circuit_element_parameters
        double* out_V
        double* out_I
        int* out_len

    double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs) nogil

    double combine_iv_job(
        int connection,
        int circuit_component_type_number,
        int n_children,
        const int* children_type_numbers,
        const double* children_Vs,
        const double* children_Is,
        const int* children_offsets,
        const int* children_lengths,
        int children_Vs_size,
        const double* children_pc_Vs,
        const double* children_pc_Is,
        const int* children_pc_offsets,
        const int* children_pc_lengths,
        int children_pc_Vs_size,
        double total_IL,
        double cap_current,
        int max_num_points,
        double area,
        int abs_max_num_points,
        const double* circuit_element_parameters,
        double* out_V,
        double* out_I,
        int* out_len
    ) 

cdef inline double _now_ms():
    """
    Tiny helper to get time in milliseconds using Python's time.perf_counter().
    Overhead is negligible compared to your kernel.
    """
    from time import perf_counter
    return perf_counter() * 1000.0

cpdef tuple run_single_job(
    int connection,
    int circuit_component_type_number,
    np.ndarray[np.int32_t, ndim=1] children_type_numbers,
    np.ndarray[np.float64_t, ndim=1] children_Vs,
    np.ndarray[np.float64_t, ndim=1] children_Is,
    np.ndarray[np.int32_t, ndim=1] children_offsets,
    np.ndarray[np.int32_t, ndim=1] children_lengths,
    np.ndarray[np.float64_t, ndim=1] children_pc_Vs,
    np.ndarray[np.float64_t, ndim=1] children_pc_Is,
    np.ndarray[np.int32_t, ndim=1] children_pc_offsets,
    np.ndarray[np.int32_t, ndim=1] children_pc_lengths,
    double total_IL,
    double cap_current,
    int max_num_points,
    double area,
    int abs_max_num_points,
    np.ndarray[np.float64_t, ndim=1] circuit_element_parameters,
    int abs_max_num_points_out
):
    """
    Cython wrapper around the flat C++ combine_iv_job call.

    Returns:
        (V, I, kernel_ms)
    Where V, I are 1D float64 numpy arrays.
    """
    cdef int n_children = children_type_numbers.shape[0]
    cdef int children_Vs_size = children_Vs.shape[0]
    cdef int children_pc_Vs_size = children_pc_Vs.shape[0]

    # Output buffers
    cdef np.ndarray[np.float64_t, ndim=2] out_IV = np.empty((2, abs_max_num_points_out), dtype=np.float64)
    cdef double[::1] mv_out_V = out_IV[0]
    cdef double[::1] mv_out_I = out_IV[1]
    cdef double* c_out_V = &mv_out_V[0]
    cdef double* c_out_I = &mv_out_I[0]
    cdef int out_len = 0


    # Get raw pointers (match C++ expected types)
    cdef int* c_children_type_numbers = <int*> children_type_numbers.data
    cdef double* c_children_Vs = <double*> children_Vs.data
    cdef double* c_children_Is = <double*> children_Is.data
    cdef int* c_children_offsets = <int*> children_offsets.data
    cdef int* c_children_lengths = <int*> children_lengths.data

    # photon-coupling
    cdef double* c_children_pc_Vs = <double*> children_pc_Vs.data
    cdef double* c_children_pc_Is = <double*> children_pc_Is.data
    cdef int* c_children_pc_offsets = <int*> children_pc_offsets.data
    cdef int* c_children_pc_lengths = <int*> children_pc_lengths.data

    cdef double* c_circuit_element_parameters = &(<np.float64_t*>circuit_element_parameters.data)[0]
    cdef int* c_out_len = &out_len

    cdef double kernel_ms
    kernel_ms = combine_iv_job(
        connection,
        circuit_component_type_number,
        n_children,
        c_children_type_numbers,
        c_children_Vs,
        c_children_Is,
        c_children_offsets,
        c_children_lengths,
        children_Vs_size,
        c_children_pc_Vs,
        c_children_pc_Is,
        c_children_pc_offsets,
        c_children_pc_lengths,
        children_pc_Vs_size,
        total_IL,
        cap_current,
        max_num_points,
        area,
        abs_max_num_points,
        c_circuit_element_parameters,
        c_out_V,
        c_out_I,
        c_out_len
    )

    if out_len < 0 or out_len > abs_max_num_points_out:
        raise ValueError(f"Invalid out_len {out_len} (abs_max_num_points_out={abs_max_num_points_out})")

    # Slice to actual used length
    IV = out_IV[:,:out_len] # try not copy!   #.copy(order='C')
    return IV, kernel_ms

def run_multiple_jobs_in_parallel(job_args_list):
    """
    job_args_list: list of tuples with args for one job
    Returns:
        results: list of (V, I)
        batch_wall_ms: total wall-clock time reported by C++
    """
    cdef Py_ssize_t n_jobs
    cdef IVJobDesc* jobs
    cdef np.ndarray[np.int32_t, ndim=1] out_len_array
    cdef int* c_out_len_all
    cdef int i, olen
    cdef np.ndarray[np.float64_t, ndim=1] out_V, out_I
    cdef double batch_wall_ms

    # memoryviews (declared ONCE, assigned inside loop)
    cdef np.int32_t[::1] mv_ctn, mv_coffs, mv_clens, mv_pcoffs, mv_pclens, mv_out_len
    cdef np.float64_t[::1] mv_cVs, mv_cIs, mv_pcVs, mv_pcIs, mv_params, mv_outV, mv_outI

    job_args_list = list(job_args_list)
    n_jobs = len(job_args_list)
    if n_jobs == 0:
        return [], 0.0

    jobs = <IVJobDesc*> malloc(n_jobs * sizeof(IVJobDesc))
    if jobs == NULL:
        raise MemoryError()
    memset(jobs, 0, n_jobs * sizeof(IVJobDesc))

    out_len_array = np.empty(n_jobs, dtype=np.int32)
    mv_out_len = out_len_array
    c_out_len_all = <int*>&mv_out_len[0]

    out_IV_list = [None] * n_jobs

    try:
        for i in range(n_jobs):
            (connection,
             circuit_component_type_number,
             children_type_numbers,
             children_Vs,
             children_Is,
             children_offsets,
             children_lengths,
             children_pc_Vs,
             children_pc_Is,
             children_pc_offsets,
             children_pc_lengths,
             total_IL,
             cap_current,
             max_num_points,
             area,
             abs_max_num_points,
             circuit_element_parameters,
             abs_max_num_points_out) = job_args_list[i]

            # allocate per-job outputs
            out_IV = np.empty((2, abs_max_num_points_out), dtype=np.float64)
            out_IV_list[i] = out_IV

            # assign memoryviews (NO cdef here)
            mv_ctn    = children_type_numbers
            mv_cVs    = children_Vs
            mv_cIs    = children_Is
            mv_coffs  = children_offsets
            mv_clens  = children_lengths

            mv_pcVs   = children_pc_Vs
            mv_pcIs   = children_pc_Is
            mv_pcoffs = children_pc_offsets
            mv_pclens = children_pc_lengths

            mv_params = circuit_element_parameters
            mv_outV   = out_IV[0]
            mv_outI   = out_IV[1]

            jobs[i].connection = connection
            jobs[i].circuit_component_type_number = circuit_component_type_number
            jobs[i].n_children = mv_ctn.shape[0]

            # children_type_numbers / offsets / lengths (int arrays)
            if mv_ctn.shape[0] > 0:
                jobs[i].children_type_numbers = <const int*>&mv_ctn[0]
            else:
                jobs[i].children_type_numbers = <const int*>0

            if mv_coffs.shape[0] > 0:
                jobs[i].children_offsets = <const int*>&mv_coffs[0]
            else:
                jobs[i].children_offsets = <const int*>0

            if mv_clens.shape[0] > 0:
                jobs[i].children_lengths = <const int*>&mv_clens[0]
            else:
                jobs[i].children_lengths = <const int*>0

            # children_Vs / Is (double arrays)
            if mv_cVs.shape[0] > 0:
                jobs[i].children_Vs = &mv_cVs[0]
            else:
                jobs[i].children_Vs = <double*>0

            if mv_cIs.shape[0] > 0:
                jobs[i].children_Is = &mv_cIs[0]
            else:
                jobs[i].children_Is = <double*>0

            jobs[i].children_Vs_size = mv_cVs.shape[0]

            # photon-coupling arrays (may also be empty)
            if mv_pcVs.shape[0] > 0:
                jobs[i].children_pc_Vs = &mv_pcVs[0]
                jobs[i].children_pc_Vs_size = mv_pcVs.shape[0]
            else:
                jobs[i].children_pc_Vs = <double*>0
                jobs[i].children_pc_Vs_size = 0

            if mv_pcIs.shape[0] > 0:
                jobs[i].children_pc_Is = &mv_pcIs[0]
            else:
                jobs[i].children_pc_Is = <double*>0

            if mv_pcoffs.shape[0] > 0:
                jobs[i].children_pc_offsets = <const int*>&mv_pcoffs[0]
            else:
                jobs[i].children_pc_offsets = <const int*>0

            if mv_pclens.shape[0] > 0:
                jobs[i].children_pc_lengths = <const int*>&mv_pclens[0]
            else:
                jobs[i].children_pc_lengths = <const int*>0

            jobs[i].total_IL              = total_IL
            jobs[i].cap_current           = cap_current
            jobs[i].max_num_points        = max_num_points
            jobs[i].area                  = area
            jobs[i].abs_max_num_points    = abs_max_num_points

            jobs[i].circuit_element_parameters = &mv_params[0]
            jobs[i].out_V                       = &mv_outV[0]
            jobs[i].out_I                       = &mv_outI[0]
            jobs[i].out_len                     = &c_out_len_all[i]

        with nogil:
            batch_wall_ms = combine_iv_jobs_batch(<int> n_jobs, jobs)

    finally:
        free(jobs)

    results = []
    for i in range(n_jobs):
        olen = c_out_len_all[i]
        IV = out_IV_list[i][:,:olen]
        results.append(IV)

    return results, batch_wall_ms