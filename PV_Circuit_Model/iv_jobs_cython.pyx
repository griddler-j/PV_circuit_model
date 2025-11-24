# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np

# Build flat children_Vs / children_Is / offsets / lengths from list of 2×Ni arrays
#
# children_IVs: list of arrays, each shape (2, Ni), dtype float64
#
# Returns:
#   children_Vs      : 1D np.float64 array of length sum(Ni)
#   children_Is      : 1D np.float64 array of length sum(Ni)
#   children_offsets : 1D np.int32   array of length n_children
#   children_lengths : 1D np.int32   array of length n_children
#
def build_children_buffers(children_IVs):
    cdef Py_ssize_t n_children = len(children_IVs)

    # -------------------------------------------
    #  If no children, return empty valid arrays
    # -------------------------------------------
    if n_children == 0:
        empty_f = np.zeros((0,), dtype=np.float64)
        empty_i = np.zeros((0,), dtype=np.int32)

        return empty_f, empty_f, empty_i, empty_i
    
    
    cdef Py_ssize_t i, j, total_len = 0

    # lengths per child
    cdef np.ndarray[np.int32_t, ndim=1] children_lengths = \
        np.empty(n_children, dtype=np.int32)

    cdef np.ndarray[np.float64_t, ndim=2] child

    # First pass: get lengths + total_len
    for i in range(n_children):
        child = np.asarray(children_IVs[i], dtype=np.float64)
        if child.ndim != 2 or child.shape[0] != 2:
            raise ValueError("Each child IV must have shape (2, Ni)")

        children_lengths[i] = <np.int32_t> child.shape[1]
        total_len += child.shape[1]

    # Offsets
    cdef np.ndarray[np.int32_t, ndim=1] children_offsets = \
        np.empty(n_children, dtype=np.int32)

    cdef Py_ssize_t offset = 0
    for i in range(n_children):
        children_offsets[i] = <np.int32_t> offset
        offset += children_lengths[i]

    # Flat Vs/Is
    cdef np.ndarray[np.float64_t, ndim=1] children_Vs = \
        np.empty(total_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] children_Is = \
        np.empty(total_len, dtype=np.float64)

    # Memoryviews for fast assignment
    cdef double[:] children_Vs_mem = children_Vs
    cdef double[:] children_Is_mem = children_Is
    cdef int[:] children_offsets_mem = children_offsets
    cdef int[:] children_lengths_mem = children_lengths

    # Second pass: copy data
    offset = 0
    cdef Py_ssize_t base
    for i in range(n_children):
        child = np.asarray(children_IVs[i], dtype=np.float64)
        base = offset
        for j in range(children_lengths_mem[i]):
            children_Vs_mem[base + j] = child[0, j]
            children_Is_mem[base + j] = child[1, j]
        offset += children_lengths_mem[i]

    return children_Vs, children_Is, children_offsets, children_lengths
