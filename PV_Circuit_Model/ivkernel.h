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
    int type_number; 
};

struct IVJobDesc {
    int connection;
    int circuit_component_type_number;
    int n_children;
    const IVView* children_IVs;
    const IVView* children_pc_IVs;
    double op_pt_V;
    int refine_mode;
    int max_num_points;
    double area;
    int abs_max_num_points;
    const double* circuit_element_parameters;
    double* out_V;
    double* out_I;
    int* out_len;
};

double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, int num_threads);

void pin_to_p_cores_only();

#ifdef __cplusplus
}
#endif
