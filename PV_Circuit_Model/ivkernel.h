// ivkernel.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct IVView {
    const double* V;   // pointer to V array
    const double* I;   // pointer to I array
    int length;        // Ni
    double scale;  
    bool left_extrapolation_allowed;
    bool right_extrapolation_allowed;
    bool has_lower_I_domain_limit;
    bool has_upper_I_domain_limit;
    int type_number;
    double element_params[5];
};

struct IVJobDesc {
    int connection;
    int circuit_component_type_number;
    int n_children;
    const IVView* this_IV;
    const IVView* children_IVs;
    const IVView* children_pc_IVs;
    int has_photon_coupling;
    double operating_point[3]; 
    int max_num_points;
    int refinement_points;
    double area;
    int abs_max_num_points;
    const double* circuit_element_parameters;
    double* out_V;
    double* out_I;
    int* out_len;
    bool* out_extrapolation_allowed;
    bool* out_has_I_domain_limit;
    int all_children_are_elements;
};

double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, int num_threads, int refine_mode, int interp_method, 
    int use_existing_grid, double refine_V_half_width, double max_tolerable_radians_change);

void interp_monotonic_inc_scalar(
    const double** xs,   // size n, strictly increasing
    const double** ys,   // size n
    const int* ns,
    const double* xqs,         // single query points
    double** yqs,        // output (single values)
    int n_jobs,
    int parallel,
    const double (*element_params)[5],
    int* circuit_type_number
);

void ivkernel_set_bandgap_table(const double* x, const double* y, int n);

#ifdef __cplusplus
}
#endif
