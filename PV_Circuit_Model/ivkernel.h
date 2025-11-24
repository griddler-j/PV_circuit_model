// ivkernel.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct JobDesc {
    int connection;
    int circuit_component_type_number;
    int n_children;
    const int* children_type_numbers;
    const double* children_Vs;
    const double* children_Is;
    const int* children_offsets;
    const int* children_lengths;
    int children_Vs_size;
    const double* children_pc_Vs;
    const double* children_pc_Is;
    const int* children_pc_offsets;
    const int* children_pc_lengths;
    int children_pc_Vs_size;
    double total_IL;
    double cap_current;
    int max_num_points;
    double area;
    int abs_max_num_points;
    const double* circuit_element_parameters;
    double* out_V;
    double* out_I;
    int* out_len;
};

double combine_iv_job(int connection,
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
    int* out_len);
double combine_iv_job_batch(const JobDesc* jobs, int n_jobs);

#ifdef __cplusplus
}
#endif
