# cython: language_level=3
# distutils: language = c++
# distutils: sources = ivkernel.cpp

import numpy as np
cimport numpy as np

cdef extern from "ivkernel.h":
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
    cdef np.ndarray[np.float64_t, ndim=1] out_V = \
        np.empty(abs_max_num_points_out, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] out_I = \
        np.empty(abs_max_num_points_out, dtype=np.float64)
    cdef int out_len = 0


    # Get raw pointers (match C++ expected types)
    cdef int* c_children_type_numbers = <int*> children_type_numbers.data
    cdef double* c_children_Vs = <double*> children_Vs.data
    cdef double* c_children_Is = <double*> children_Is.data
    cdef int* c_children_offsets = <int*> children_offsets.data
    cdef int* c_children_lengths = <int*> children_lengths.data

    # photon-coupling
    c_children_pc_Vs = <double*> children_pc_Vs.data
    c_children_pc_Is = <double*> children_pc_Is.data
    c_children_pc_offsets = <int*> children_pc_offsets.data
    c_children_pc_lengths = <int*> children_pc_lengths.data

    cdef double* c_circuit_element_parameters = &(<np.float64_t*>circuit_element_parameters.data)[0]
    cdef double* c_out_V = &(<np.float64_t*>out_V.data)[0]
    cdef double* c_out_I = &(<np.float64_t*>out_I.data)[0]
    cdef int* c_out_len = &out_len

    cdef double kernel_ms = combine_iv_job(
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
    V = out_V[:out_len].copy()
    I = out_I[:out_len].copy()
    return V, I, kernel_ms
