# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
# distutils: language = c++

import numpy as np
cimport numpy as np
from cython cimport nogil
from libc.stdlib cimport malloc, free 
from libc.string cimport memset
from PV_Circuit_Model.utilities import ParameterSet 
from PV_Circuit_Model.utilities_silicon import bandgap_narrowing_RT
from pathlib import Path
from libcpp cimport bool as cbool
from libc.math cimport isfinite  # C-level, fast
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

REFINE_V_HALF_WIDTH = 0.005
_SUPER_DENSE = 0    # don't change!  for debugging only
MAX_TOLERABLE_RADIANS_CHANGE = 0.008726638 # half a degree
REMESH_POINTS_DENSITY = 500
REFINEMENT_POINTS_DENSITY = 125
REMESH_NUM_ELEMENTS_THRESHOLD = 50

solver_env_variables = None
try:
    ParameterSet(name="solver_env_variables",filename=PARAM_DIR / "solver_env_variables.json")
    solver_env_variables = ParameterSet.get_set("solver_env_variables")
    _PARALLEL_MODE = solver_env_variables["_PARALLEL_MODE"]
    REFINE_V_HALF_WIDTH = solver_env_variables["REFINE_V_HALF_WIDTH"]
    MAX_TOLERABLE_RADIANS_CHANGE = solver_env_variables["MAX_TOLERABLE_RADIANS_CHANGE"]
    REMESH_POINTS_DENSITY = solver_env_variables["REMESH_POINTS_DENSITY"]
    REFINEMENT_POINTS_DENSITY = solver_env_variables["REFINEMENT_POINTS_DENSITY"]
    REMESH_NUM_ELEMENTS_THRESHOLD = solver_env_variables["REMESH_NUM_ELEMENTS_THRESHOLD"]
except Exception:
    ParameterSet(name="solver_env_variables",data={})
    solver_env_variables = ParameterSet.get_set("solver_env_variables")
    solver_env_variables.set("_PARALLEL_MODE", _PARALLEL_MODE)
    solver_env_variables.set("REFINE_V_HALF_WIDTH", REFINE_V_HALF_WIDTH)
    solver_env_variables.set("MAX_TOLERABLE_RADIANS_CHANGE", MAX_TOLERABLE_RADIANS_CHANGE)
    solver_env_variables.set("REMESH_POINTS_DENSITY", REMESH_POINTS_DENSITY)
    solver_env_variables.set("REFINEMENT_POINTS_DENSITY", REFINEMENT_POINTS_DENSITY)
    solver_env_variables.set("REMESH_NUM_ELEMENTS_THRESHOLD", REMESH_NUM_ELEMENTS_THRESHOLD)

solver_env_variables.set("_SUPER_DENSE", _SUPER_DENSE)
solver_env_variables.set("_USE_CYTHON", True)

def set_parallel_mode(enabled: bool):
    global _PARALLEL_MODE, solver_env_variables
    _PARALLEL_MODE = bool(enabled)
    solver_env_variables.set("_PARALLEL_MODE", _PARALLEL_MODE)

def get_parallel_mode():
    global solver_env_variables
    return solver_env_variables["_PARALLEL_MODE"]

if os.environ.get("PV_CIRCUIT_NO_OPENMP", "").strip()=="1":
    set_parallel_mode(False)

if ivkernel_has_openmp() == 0:
    set_parallel_mode(False)

def set_super_dense(num_points):
    global _SUPER_DENSE
    _SUPER_DENSE = int(num_points)
    solver_env_variables.set("_SUPER_DENSE", _SUPER_DENSE)

constants = ParameterSet.get_set("constants")
_q = constants["q"]

cdef extern from "ivkernel.h":

    int ivkernel_has_openmp()

    cdef struct IVView:
        const double* V      # pointer to V array
        const double* I      # pointer to I array
        int length           # Ni
        double scale
        cbool left_extrapolation_allowed
        cbool right_extrapolation_allowed
        double extrapolation_dI_dV[2]
        cbool has_lower_I_domain_limit
        cbool has_upper_I_domain_limit
        int type_number
        double element_params[8]

    cdef struct IVJobDesc:
        int connection
        int circuit_component_type_number
        int n_children
        const IVView* this_IV
        const IVView* children_IVs
        const IVView* children_pc_IVs
        int has_photon_coupling
        double operating_point[3] # V, bottom_up_operating_point_V,normalized_operating_point_V,I
        int max_num_points
        int refinement_points
        double area
        int abs_max_num_points 
        const double* circuit_element_parameters
        double* out_V
        double* out_I
        int* out_len;
        cbool* out_extrapolation_allowed;
        double* out_extrapolation_dI_dV;
        cbool* out_has_I_domain_limit;
        int all_children_are_elements;

    bint combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, 
    int parallel, int refine_mode, int interp_method, int use_existing_grid, 
    double refine_V_half_width, double max_tolerable_radians_change, 
    int has_any_intrinsic_diode, int has_any_photon_coupling, int largest_abs_max_num_points) nogil

    void interp_monotonic_inc_scalar(
        const double** xs,
        const double** ys,
        const int* ns,
        const double* xqs,
        double** yqs,
        int n_jobs,
        int parallel,
        const double (*element_params)[8],
        int* circuit_type_number
    ) nogil

    void ivkernel_set_bandgap_table(const double* x, const double* y, int n)

    void ivkernel_set_q(double q_value)
    

# Keep arrays alive so C++ can safely hold pointers into them
cdef object _bgn_x_store = None
cdef object _bgn_y_store = None
cdef bint _tables_initialized = False
cdef bint _q_initialized = False

cdef void _init_ivkernel_tables():
    global _bgn_x_store, _bgn_y_store, _tables_initialized
    if _tables_initialized:
        return

    # bandgap_narrowing_RT is your Python list-of-lists
    cdef np.ndarray[np.float64_t, ndim=2] arr = \
        np.asarray(bandgap_narrowing_RT, dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] x_arr = \
        np.ascontiguousarray(arr[:, 0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] y_arr = \
        np.ascontiguousarray(arr[:, 1], dtype=np.float64)

    # Keep references alive at module scope, so their buffers don't get freed
    _bgn_x_store = x_arr
    _bgn_y_store = y_arr

    cdef int n = x_arr.shape[0]
    if n == 0:
        raise ValueError("bandgap_narrowing_RT is empty")

    ivkernel_set_bandgap_table(&x_arr[0], &y_arr[0], n)

    _tables_initialized = True

cdef void _init_q():
    global _q, _q_initialized
    if _q_initialized:
        return
    ivkernel_set_q(_q)
    _q_initialized = True

def init_ivkernel_tables():
    _init_ivkernel_tables()

def init_q():
    _init_q()

_init_ivkernel_tables()
_init_q()

cdef inline bint _setp(double[:] p, int k, double v) nogil:
    if not isfinite(v):
        return 0
    p[k] = v
    return 1

def run_multiple_jobs(components,refine_mode=False,parallel=False,interp_method=0,super_dense=10000,use_existing_grid=False):

    if parallel and ivkernel_has_openmp() == 0:
        parallel = False

    cdef int parallel_ = 1 if parallel else 0
    cdef int has_any_photon_coupling = 0
    cdef int has_any_intrinsic_diode = 0
    cdef Py_ssize_t n_jobs = len(components)
    cdef int n_jobs_c = <int> n_jobs

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

    cdef IVView* this_view = <IVView*> malloc(n_jobs*sizeof(IVView))
    if this_view == NULL:
        free(jobs_c)
        free(children_views)
        raise MemoryError()

    # This list keeps all numpy buffers alive until we return from this function
    memset(this_view, 0, n_jobs*sizeof(IVView))

    cdef IVView* pc_children_views = <IVView*> malloc(total_children * sizeof(IVView))
    if pc_children_views == NULL:
        free(jobs_c)
        free(this_view)
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
    cdef np.float64_t[::1] mv_this_v
    cdef np.float64_t[::1] mv_this_i

    cdef int circuit_component_type_number, type_number
    cdef int n_children, Ni
    cdef int abs_max_num_points
    cdef double area
    cdef double REFINE_V_HALF_WIDTH_ = REFINE_V_HALF_WIDTH
    cdef double max_tolerable_radians_change = MAX_TOLERABLE_RADIANS_CHANGE

    cdef Py_ssize_t child_base = 0
    cdef double kernel_ms
    cdef int olen
    cdef double tmp, tmp2
    cdef np.float64_t[::1] mv_big_v 
    cdef np.float64_t[::1] mv_big_i 
    cdef np.uint8_t[::1] mv_big_extrapolation_allowed
    cdef np.float64_t[::1] mv_big_extrapolation_dI_dV
    cdef np.uint8_t[::1] mv_big_has_I_domain_limit
    cdef double* base 
    cdef Py_ssize_t stride_job 
    cdef Py_ssize_t offset, offset2
    cdef int sum_abs_max_num_points
    cdef int largest_abs_max_num_points = 0
    cdef double normalized_operating_point_V
    cdef double normalized_operating_point_I
    cdef double bottom_up_operating_point_V
    cdef double bottom_up_operating_point_I
    cdef int all_children_are_elements, refine_mode_, interp_method_, use_existing_grid_, refinement_points_density
    cdef double* base_v
    cdef double* base_i
    cdef double* base_extrapolation_dI_dV
    cdef cbool* base_extrapolation_allowed
    cdef cbool* base_has_I_domain_limit
    cdef int abs_max_num_points_multipier

    if refine_mode:
        refine_mode_ = 1
    else:
        refine_mode_ = 0

    if use_existing_grid:
        use_existing_grid_ = 1
    else:
        use_existing_grid_ = 0

    interp_method_ = interp_method
    refinement_points_density = REFINEMENT_POINTS_DENSITY

    try:
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
            if max_I is None:
                max_I = 0.2
            if circuit_component_type_number == 0:      # CurrentSource
                if not _setp(mv_params, 0, circuit_component.IL): return 1
            elif circuit_component_type_number == 1:    # Resistor
                if not _setp(mv_params, 0, circuit_component.cond): return 1
            elif circuit_component_type_number in (2, 3):  # diodes
                if not _setp(mv_params, 0, circuit_component.I0): return 1
                if not _setp(mv_params, 1, circuit_component.n): return 1
                if not _setp(mv_params, 2, circuit_component.VT): return 1
                if not _setp(mv_params, 3, circuit_component.V_shift): return 1
                if not _setp(mv_params, 4, max_I): return 1
            elif circuit_component_type_number == 4:    # Intrinsic_Si_diode
                has_any_intrinsic_diode = 1
                base_type_number = 0.0  # p
                try:
                    if circuit_component.base_type == "n":
                        base_type_number = 1.0
                except AttributeError:
                    pass

                if not _setp(mv_params, 0, circuit_component.base_doping): return 1
                mv_params[1] = 1.0
                if not _setp(mv_params, 2, circuit_component.VT): return 1
                if not _setp(mv_params, 3, circuit_component.base_thickness): return 1
                if not _setp(mv_params, 4, max_I): return 1
                if not _setp(mv_params, 5, circuit_component.ni): return 1
                if not _setp(mv_params, 6, base_type_number): return 1
            else:
                # CircuitGroup or unknown
                pass 

            # ----- scalar fields -----
            area = 1.0
            if circuit_component._type_number==6 and circuit_component.area is not None:
                area = circuit_component.area
            max_num_points = circuit_component.max_num_points
            if max_num_points is None:
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

            if use_existing_grid:
                mv_this_v = circuit_component.IV_V
                mv_this_i = circuit_component.IV_I
                Ni = mv_this_v.shape[0]
                this_view[i].V           = &mv_this_v[0]
                this_view[i].I           = &mv_this_i[0]
                this_view[i].length      = Ni
                jobs_c[i].this_IV = &this_view[i]
            else:
                jobs_c[i].this_IV = <IVView*> 0

            if refine_mode:
                bottom_up_operating_point_V = 0;
                bottom_up_operating_point_I = 0;
                normalized_operating_point_V = 0;
                normalized_operating_point_I = 0;

            all_children_are_elements = 1
            for j in range(n_children):
                element = subgroups[j]
                type_number = element._type_number

                if refine_mode and not use_existing_grid:
                    if type_number < 5:
                        bottom_up_operating_point_V += element.operating_point[0]
                        bottom_up_operating_point_I += element.operating_point[1]
                        normalized_operating_point_V += 1
                        normalized_operating_point_I += 1
                    else:
                        all_children_are_elements = 0
                        if element.bottom_up_operating_point is not None:
                            bottom_up_operating_point_V += element.bottom_up_operating_point[0]
                            bottom_up_operating_point_I += element.bottom_up_operating_point[1]
                            normalized_operating_point_V += element.normalized_operating_point[0]
                            normalized_operating_point_I += element.normalized_operating_point[1]

                children_views[child_base + j].type_number = type_number
                # ensure IV_table is C-contiguous float64 (2, Ni)
                mv_child_v = element.IV_V
                mv_child_i = element.IV_I
                Ni = mv_child_v.shape[0]
                abs_max_num_points += Ni

                children_views[child_base + j].left_extrapolation_allowed = element.extrapolation_allowed[0]
                children_views[child_base + j].right_extrapolation_allowed = element.extrapolation_allowed[1]
                children_views[child_base + j].extrapolation_dI_dV[0] = element.extrapolation_dI_dV[0]
                children_views[child_base + j].extrapolation_dI_dV[1] = element.extrapolation_dI_dV[1]
                children_views[child_base + j].has_lower_I_domain_limit = element.has_I_domain_limit[0]
                children_views[child_base + j].has_upper_I_domain_limit = element.has_I_domain_limit[1]

                children_views[child_base + j].V           = &mv_child_v[0]
                children_views[child_base + j].I           = &mv_child_i[0]
                children_views[child_base + j].length      = Ni

                if type_number in (2, 3):  # diodes
                    children_views[child_base + j].element_params[0] = element.I0
                    children_views[child_base + j].element_params[1] = element.n
                    children_views[child_base + j].element_params[2] = element.VT
                    children_views[child_base + j].element_params[3] = element.V_shift
                elif type_number == 4:    # Intrinsic_Si_diode
                    has_any_intrinsic_diode = 1
                    base_type_number = 0.0  # p
                    if element.base_type == "n":
                        base_type_number = 1.0
                    children_views[child_base + j].element_params[0] = element.base_doping
                    children_views[child_base + j].element_params[1] = element.VT
                    children_views[child_base + j].element_params[2] = element.base_thickness
                    children_views[child_base + j].element_params[3] = element.ni
                    children_views[child_base + j].element_params[4] = base_type_number

            if refine_mode and not use_existing_grid:
                if jobs_c[i].connection == 0:  # series 
                    bottom_up_operating_point_I /= n_children
                    normalized_operating_point_I /= n_children
                else:
                    bottom_up_operating_point_V /= n_children
                    normalized_operating_point_V /= n_children
                if circuit_component._type_number == 6: # cell
                     bottom_up_operating_point_I *= area
                jobs_c[i].operating_point[0] = circuit_component.operating_point[0]
                jobs_c[i].operating_point[1] = bottom_up_operating_point_V
                jobs_c[i].operating_point[2] = normalized_operating_point_V
                jobs_c[i].refinement_points = int(refinement_points_density*np.sqrt(circuit_component.num_circuit_elements))
                circuit_component.bottom_up_operating_point = [bottom_up_operating_point_V,bottom_up_operating_point_I]
                circuit_component.normalized_operating_point = [normalized_operating_point_V,normalized_operating_point_I]
                jobs_c[i].all_children_are_elements = all_children_are_elements

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
                jobs_c[i].has_photon_coupling = 0
            else:
                jobs_c[i].has_photon_coupling = 1
                has_any_photon_coupling = 1

            abs_max_num_points = abs_max_num_points_multipier*abs_max_num_points

            child_base += n_children

            if super_dense > 0: 
                if jobs_c[i].circuit_component_type_number < 5: # if element, make lots of points
                    jobs_c[i].max_num_points = super_dense
                else: # else, don't ever remesh
                    jobs_c[i].max_num_points = -1

            if circuit_component_type_number==0: # current source
                abs_max_num_points = 1
            elif circuit_component_type_number==1: # resistor
                abs_max_num_points = 2
            elif circuit_component_type_number>=2 and circuit_component_type_number<=4: # diode
                tmp2 = max(jobs_c[i].max_num_points,100.0)
                tmp = tmp2 / 0.2 * max_I + 6.0
                abs_max_num_points = <int>(tmp + 0.999999)  # cheap ceil

            abs_max_num_points = int(abs_max_num_points)

            if refine_mode and all_children_are_elements and not use_existing_grid:
                abs_max_num_points += jobs_c[i].refinement_points

            if use_existing_grid:
                abs_max_num_points = this_view[i].length

            jobs_c[i].abs_max_num_points = abs_max_num_points

            sum_abs_max_num_points += abs_max_num_points
            if abs_max_num_points > largest_abs_max_num_points:
                largest_abs_max_num_points = abs_max_num_points

        big_out_V = np.empty(sum_abs_max_num_points, dtype=np.float64)
        big_out_I = np.empty(sum_abs_max_num_points, dtype=np.float64)
        big_out_extrapolation_dI_dV = np.empty(n_jobs*2, dtype=np.float64)
        big_out_extrapolation_allowed = np.empty(n_jobs*2, dtype=np.uint8)
        big_out_has_I_domain_limit = np.empty(n_jobs*2, dtype=np.uint8)
        
        # anchors the array
        mv_big_v = big_out_V  
        mv_big_i = big_out_I  
        mv_big_extrapolation_allowed = big_out_extrapolation_allowed 
        mv_big_extrapolation_dI_dV = big_out_extrapolation_dI_dV
        mv_big_has_I_domain_limit = big_out_has_I_domain_limit 

        # base pointer into contiguous buffer
        base_v = &mv_big_v[0]              
        base_i = &mv_big_i[0]              
        base_extrapolation_allowed = <cbool*>&mv_big_extrapolation_allowed[0]
        base_extrapolation_dI_dV = &mv_big_extrapolation_dI_dV[0]
        base_has_I_domain_limit = <cbool*>&mv_big_has_I_domain_limit[0]

        offset = 0
        offset2 = 0
        for i in range(n_jobs):
            jobs_c[i].out_V   = base_v + offset
            jobs_c[i].out_I   = base_i + offset 
            jobs_c[i].out_extrapolation_allowed = base_extrapolation_allowed + offset2
            jobs_c[i].out_extrapolation_dI_dV = base_extrapolation_dI_dV + offset2
            jobs_c[i].out_has_I_domain_limit = base_has_I_domain_limit + offset2
            jobs_c[i].out_len = &c_out_len_all[i]
            offset += jobs_c[i].abs_max_num_points
            offset2 += 2

        # ----- call C++ batched kernel (no Python inside) -----
        with nogil:
            success = combine_iv_jobs_batch(n_jobs_c, jobs_c, parallel_, refine_mode_, interp_method_, 
            use_existing_grid_, REFINE_V_HALF_WIDTH_, max_tolerable_radians_change, has_any_intrinsic_diode, has_any_photon_coupling, largest_abs_max_num_points)

        # ----- unpack outputs -----
        if not success:
            return 2

        offset = 0
        for i in range(n_jobs):
            circuit_component = components[i]
            olen = c_out_len_all[i]
            if olen < 0:
                raise ValueError(f"Negative out_len for job {i}")
            circuit_component.IV_V = big_out_V[offset:offset+olen]
            circuit_component.IV_I = big_out_I[offset:offset+olen]
            if not use_existing_grid:
                circuit_component.extrapolation_allowed = [bool(big_out_extrapolation_allowed[2*i]),bool(big_out_extrapolation_allowed[2*i+1])]
                circuit_component.extrapolation_dI_dV = [big_out_extrapolation_dI_dV[2*i],big_out_extrapolation_dI_dV[2*i+1]]
                # if circuit_component._type_number >= 5:
                #     print("Yoz: circuit_component.extrapolation_dI_dV = ", circuit_component.extrapolation_dI_dV)
                circuit_component.has_I_domain_limit = [bool(big_out_has_I_domain_limit[2*i]),bool(big_out_has_I_domain_limit[2*i+1])]
            offset += jobs_c[i].abs_max_num_points

        return 0

    finally:
        free(jobs_c)
        free(children_views)
        free(pc_children_views)
        free(this_view)
        

def run_multiple_operating_points(components, bint parallel=False):
    cdef Py_ssize_t n_jobs = len(components)
    if n_jobs == 0:
        return np.empty(0, dtype=np.float64)

    cdef int parallel_ = 1 if parallel else 0

    # --------------------------------------------------------
    # Allocate output array (one solved value per component)
    # --------------------------------------------------------
    cdef np.ndarray[np.float64_t, ndim=1] yqs_arr = np.empty(n_jobs, dtype=np.float64)
    cdef double[:] yqs_mv = yqs_arr  # memoryview over output

    # Query values
    cdef np.ndarray[np.float64_t, ndim=1] xqs_arr = np.empty(n_jobs, dtype=np.float64)
    cdef double[:] xqs_mv = xqs_arr

    # Allocate C pointer tables
    cdef const double** xs = <const double**> malloc(n_jobs * sizeof(const double*))
    cdef const double** ys = <const double**> malloc(n_jobs * sizeof(const double*))
    cdef double[8] *element_params
    element_params = <double[8] *> malloc(n_jobs * sizeof(double[8]))
    cdef int* circuit_type_number = <int*> malloc(n_jobs * sizeof(int))
    cdef int* ns           = <int*> malloc(n_jobs * sizeof(int))
    cdef double** yqs      = <double**> malloc(n_jobs * sizeof(double*))
    cdef bint* known_is_V = <bint*> malloc(n_jobs * sizeof(bint))
    cdef object circuit_component
    cdef object operating_point

    if xs == NULL or ys == NULL or ns == NULL or yqs == NULL or known_is_V == NULL or element_params == NULL or circuit_type_number==NULL:
        if xs != NULL:       free(<void*> xs)
        if ys != NULL:       free(<void*> ys)
        if ns != NULL:       free(<void*> ns)
        if yqs != NULL:      free(<void*> yqs)
        if known_is_V != NULL: free(<void*> known_is_V)
        if element_params != NULL: free(<void*> element_params)
        if circuit_type_number != NULL: free(<void*> circuit_type_number)
        raise MemoryError()

    cdef Py_ssize_t i
    cdef double[:] xmv, ymv,dpmv

    try:
        # --------------------------------------------------------
        # Build pointer lists for each job
        # --------------------------------------------------------
        for i in range(n_jobs):
            circuit_component = components[i]
            circuit_type_number[i] = <int>circuit_component._type_number
            
            # Select which axis is X and which is Y
            if circuit_component.operating_point[0] is not None:
                # X = V, Y = I, query in V -> solve I(V)
                known_is_V[i] = 1
                xmv = circuit_component.IV_V
                ymv = circuit_component.IV_I
                xqs_mv[i] = <double> circuit_component.operating_point[0]
            else:
                # X = I, Y = V, query in I -> solve V(I)
                known_is_V[i] = 0
                circuit_type_number[i] = -1
                xmv = circuit_component.IV_I
                ymv = circuit_component.IV_V
                xqs_mv[i] = <double> circuit_component.operating_point[1]

            if (circuit_type_number[i]>=2 and circuit_type_number[i]<=3):
                element_params[i][0] = circuit_component.I0
                element_params[i][1] = circuit_component.n
                element_params[i][2] = circuit_component.VT
                element_params[i][3] = circuit_component.V_shift
            elif (circuit_type_number[i]==4):
                base_type_number = 0.0  # p
                if circuit_component.base_type == "n":
                    base_type_number = 1.0
                element_params[i][0] = circuit_component.base_doping
                element_params[i][1] = circuit_component.VT
                element_params[i][2] = circuit_component.base_thickness
                element_params[i][3] = circuit_component.ni
                element_params[i][4] = base_type_number

            # Fill metadata for this job
            ns[i]  = <int> xmv.shape[0]
            xs[i]  = &xmv[0]
            ys[i]  = &ymv[0]
            yqs[i] = &yqs_mv[i]

        with nogil:
            interp_monotonic_inc_scalar(
                xs, ys, ns,
                &xqs_mv[0],
                yqs,
                <int> n_jobs,
                parallel_,
                element_params,
                circuit_type_number
            )

        for i in range(n_jobs):
            circuit_component = components[i]
            operating_point = circuit_component.operating_point
            if known_is_V[i]:
                operating_point[1] = yqs_arr[i]
            else:
                operating_point[0] = yqs_arr[i]
            if circuit_component._type_number >= 5: 
                is_series = False
                if circuit_component.connection=="series":
                    is_series = True
                current_ = operating_point[1]
                if circuit_component._type_number == 6: # cell
                     current_ /= circuit_component.area
                for child in circuit_component.subgroups:
                    if is_series:
                        child.operating_point = [None, current_]
                    else:
                        child.operating_point = [operating_point[0], None]

    finally:
        free(<void*> xs)
        free(<void*> ys)
        free(<void*> ns)
        free(<void*> yqs)
        free(<void*> known_is_V)
        free(<void*> element_params)
        free(<void*> circuit_type_number)