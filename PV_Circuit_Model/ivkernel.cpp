// ivkernel.cpp
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <omp.h> 
#include <chrono>
#include "ivkernel.h"
#include <numeric>
#include <queue>
#include <intrin.h>
#include <fstream>


extern "C" {

    struct MergeNode {
    const double* cur;  // pointer to current element in this child
    const double* end;  // one-past-last pointer
};

struct MergeNodeCmp {
    bool operator()(const MergeNode& a, const MergeNode& b) const {
        // priority_queue is max-heap; flip for min-heap
        return *(a.cur) > *(b.cur);
    }
};

void merge_children_kway_ptr(
    const IVView* children_IVs,
    int sort_V_or_I,
    int n_children,
    double* out_Vs,     // output buffer
    int& out_len,       // number written
    int capacity        // max capacity of out_Vs
) {
    using PQ = std::priority_queue<
        MergeNode,
        std::vector<MergeNode>,
        MergeNodeCmp
    >;

    PQ pq;

    // Fill heap with the first element of each non-empty child
    for (int i = 0; i < n_children; ++i) {
        int len = children_IVs[i].length;
        if (len == 0) continue;

        const double* p =
            (sort_V_or_I == 0)
            ? children_IVs[i].V
            : children_IVs[i].I;

        MergeNode node{p, p + len};
        pq.push(node);
    }

    int count = 0;

    // K-way merge
    while (!pq.empty() && count < capacity) {
        MergeNode node = pq.top();
        pq.pop();

        out_Vs[count++] = *node.cur;

        ++node.cur;
        if (node.cur != node.end) {
            pq.push(node);
        }
    }

    out_len = count;
}

void build_current_source_iv(
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len,
    int abs_max_num_points
) {
    double IL = circuit_element_parameters[0];
    int n = 1;
    for (int i=0; i < n; ++i) {
        out_V[i] = 0;
        out_I[i] = -IL;
    }
    *out_len = n;
}

void build_resistor_iv(
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len,
    int abs_max_num_points
) {
    double cond = circuit_element_parameters[0];
    int n = 2;
    for (int i=0; i < n; ++i) {
        out_V[i] = -0.1 + i*0.2;
        out_I[i] = cond*out_V[i];
    }
    *out_len = n;
}

void interp_monotonic_inc(
    const double* x,   // size n, increasing
    const double* y,   // size n
    int n,
    const double* xq,  // size m, increasing
    int m,
    double* yq,         // size m, output
    bool additive        // true: adds to yq
) {
    if (!x || !y || !xq || !yq || n <= 0 || m <= 0) return;
    if (n == 1) {
        // Degenerate: constant function (like IL)
        for (int j = 0; j < m; ++j) {
            yq[j] = (additive ? yq[j] : 0.0) + y[0];
        }
        return;
    } else if (n==2) {
        // Degenerate: slope (like R)
        double slope = (y[1] - y[0])/(x[1] - x[0]);
        for (int j = 0; j < m; ++j) {
            yq[j] = (additive ? yq[j] : 0.0) + y[0] + (xq[j]-x[0])*slope;
        }
        return;
    }
    int j_left = 0;
    // --- Left extrapolation: xq <= x[0] ---
    double slope_left;
    if (xq[j_left] <= x[0]) {
        slope_left = (y[1] - y[0])/(x[1] - x[0]);
        while (j_left < m && xq[j_left] <= x[0]) {
            yq[j_left] = (additive? yq[j_left]:0.0) + y[0] + (xq[j_left] - x[0])*slope_left;
            ++j_left;
        }
    }
    int j_right = m-1;
    // --- Right extrapolation: xq >= x[n-1] ---
    double slope_right;
    if (xq[j_right] >= x[n-1]) {
        slope_right = (y[n-1] - y[n-2])/(x[n-1] - x[n-2]);
        while (j_left <= j_right && xq[j_right] >= x[n-1]) {
            yq[j_right] = (additive? yq[j_right]:0.0) + y[n-1] + (xq[j_right] - x[n-1])*slope_right;
            --j_right;
        }
    }
    // --- Main interpolation region: x[0] < xq < x[n-1] ---
    int i = 0; // segment index for x
    int prev_i = -1;
    double slope;
    for (int j = j_left; j <= j_right; ++j) {
        double xj = xq[j];
        // Advance i until x[i] <= xj <= x[i+1]
        while (i + 1 < n - 1 && x[i+1] < xj) 
            ++i;
        if (i > prev_i) {
            slope = (y[i+1]-y[i])/(x[i+1]-x[i]);
        }
        prev_i = i;
        yq[j] = (additive? yq[j]:0.0) + y[i] + slope*(xj - x[i]);
    }
}

void interp_monotonic_inc_scalar(
    const double** xs,   // size n, strictly increasing
    const double** ys,   // size n
    const int* ns,
    const double* xqs,         // single query points
    double** yqs,        // output (single values)
    int n_jobs,
    int parallel
) {
    int max_threads = omp_get_max_threads();
    int num_threads = max_threads;
    if (n_jobs < max_threads*2) num_threads = max_threads/2;
    if (n_jobs < max_threads) num_threads = max_threads/4;
    if (n_jobs < max_threads/4) num_threads = n_jobs;

    if (parallel==1) {
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < n_jobs; i ++) {
            const double* x = xs[i];
            const double* y = ys[i];
            int n = ns[i];
            double xq = xqs[i];
            double* yq = yqs[i];

            if (!x || !y || !yq || n <= 0) continue;

            // n == 1: constant function (like IL)
            if (n == 1) {
                *yq = y[0];
                continue;
            }

            // n == 2: simple line (like R)
            if (n == 2) {
                double slope = (y[1] - y[0]) / (x[1] - x[0]);
                *yq = (y[0] + (xq - x[0]) * slope);
                continue;
            }

            // n >= 3 from here on

            // --- Left extrapolation: xq <= x[0] ---
            if (xq <= x[0]) {
                double slope_left = (y[1] - y[0]) / (x[1] - x[0]);
                *yq = (y[0] + (xq - x[0]) * slope_left);
                continue;
            }

            // --- Right extrapolation: xq >= x[n-1] ---
            if (xq >= x[n-1]) {
                double slope_right = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
                *yq = (y[n-1] + (xq - x[n-1]) * slope_right);
                continue;
            }

            // --- Main interpolation region: x[0] < xq < x[n-1] ---
            // Binary search for i such that x[i] <= xq <= x[i+1]
            int low = 0;
            int high = n - 1;

            // Invariant: x[low] <= xq < x[high]
            while (high - low > 1) {
                int mid = (low + high) / 2;
                if (x[mid] <= xq) {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            int j = low;
            double slope = (y[j+1] - y[j]) / (x[j+1] - x[j]);
            *yq = (y[j] + slope * (xq - x[j]));
        }
    }
    else {
        for (int i = 0; i < n_jobs; i ++) {
            const double* x = xs[i];
            const double* y = ys[i];
            int n = ns[i];
            double xq = xqs[i];
            double* yq = yqs[i];

            if (!x || !y || !yq || n <= 0) continue;

            // n == 1: constant function (like IL)
            if (n == 1) {
                *yq = y[0];
                continue;
            }

            // n == 2: simple line (like R)
            if (n == 2) {
                double slope = (y[1] - y[0]) / (x[1] - x[0]);
                *yq = (y[0] + (xq - x[0]) * slope);
                continue;
            }

            // n >= 3 from here on

            // --- Left extrapolation: xq <= x[0] ---
            if (xq <= x[0]) {
                double slope_left = (y[1] - y[0]) / (x[1] - x[0]);
                *yq = (y[0] + (xq - x[0]) * slope_left);
                continue;
            }

            // --- Right extrapolation: xq >= x[n-1] ---
            if (xq >= x[n-1]) {
                double slope_right = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2]);
                *yq = (y[n-1] + (xq - x[n-1]) * slope_right);
                continue;
            }

            // --- Main interpolation region: x[0] < xq < x[n-1] ---
            // Binary search for i such that x[i] <= xq <= x[i+1]
            int low = 0;
            int high = n - 1;

            // Invariant: x[low] <= xq < x[high]
            while (high - low > 1) {
                int mid = (low + high) / 2;
                if (x[mid] <= xq) {
                    low = mid;
                } else {
                    high = mid;
                }
            }

            int j = low;
            double slope = (y[j+1] - y[j]) / (x[j+1] - x[j]);
            *yq = (y[j] + slope * (xq - x[j]));
        }

    }
}

void calc_intrinsic_Si_I(
    const double* V,          // input voltages, length n_V
    int n_V,
    double ni,
    double  VT,
    double base_doping,
    int base_type_number,   // 0-p, 1-n
    double base_thickness,
    double area,               // area is actually always 1
    double* out_I             // output: I(V) or dI/dV, length n_V
) {
    const double q = 1.602e-19;
    double N_doping = base_doping;

    // Equilibrium n0, p0 depend only on doping & ni, not on V
    double n0, p0;
    {
        double root = std::sqrt(N_doping * N_doping + 4.0 * ni * ni);
        if (base_type_number == 0) {
            n0 = 0.5 * (-N_doping + root);
            p0 = 0.5 * ( N_doping + root);
        } else {
            p0 = 0.5 * (-N_doping + root);
            n0 = 0.5 * ( N_doping + root);
        }
    }

    // gee / geh depend only on n0, p0
    double geeh = 1.0 + 13.0 * (1.0 - std::tanh(std::pow(n0 / 3.3e17, 0.66)));
    double gehh = 1.0 + 7.5  * (1.0 - std::tanh(std::pow(p0 / 7e17, 0.63)));

    const double Brel = 1.0;
    const double Blow = 4.73e-15;

    std::vector<double> delta_n(n_V);
    std::vector<double> BGN(n_V);
    std::vector<double> pn(n_V);

    // Compute delta_n first
    for (int i = 0; i < n_V; ++i) {
        double expv = std::exp(V[i]/VT);
        pn[i] = ni * ni * expv;
        delta_n[i] = 0.5 * (-N_doping + std::sqrt(N_doping*N_doping + 4.0*ni*ni*expv));
    }

    static const double bgn_x[] = {
        1e10, 1e14, 3e14, 1e15, 3e15, 1e16, 3e16,
        1e17, 3e17, 1e18, 3e18, 1e19, 3e19, 1e20
    };

    static const double bgn_y[] = {
        1.41e-03,
        0.00145608,
        0.00155279,
        0.00187385,
        0.00258644,
        0.00414601,
        0.00664397,
        0.0112257,
        0.018247,
        0.0295337,
        0.0421825,
        0.0597645,
        0.0811658,
        0.113245
    };

    int n_bgn = 14;

    // Vector interpolation:
    interp_monotonic_inc(
        bgn_x,
        bgn_y,
        n_bgn,
        delta_n.data(),
        n_V,
        BGN.data(),
        false        // overwrite
    );

    double termA = 2.5e-31 * geeh * n0;
    double termB = 8.5e-32 * gehh * p0;
    for (int i = 0; i < n_V; ++i) {
        // ni_eff = ni * exp(BGN/(2*VT))
        double ni_eff = ni * std::exp(BGN[i] / (2.0 * VT));

        // Recombination prefactor
        double termC = 3e-29   * std::pow(delta_n[i], 0.92);
        double coeff = termA + termB + termC + Brel * Blow;

        // intrinsic_recomb = (pn - ni_eff^2)*coeff
        double intrinsic_recomb = (pn[i] - ni_eff*ni_eff) * coeff;

        // I(V) = q * intrinsic_recomb * thickness * area
        out_I[i] = q * intrinsic_recomb * base_thickness * area;
    }
}

std::vector<double> get_V_range(const double* circuit_element_parameters,int max_num_points,bool intrinsic_Si_calc,double op_pt_V,bool is_reverse_diode) {
    // circuit_component.base_doping, circuit_component.n, circuit_component.VT, circuit_component.base_thickness, max_I, circuit_component.ni, base_type_number
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = 0;
    if (!intrinsic_Si_calc) V_shift = circuit_element_parameters[3];
    double max_I = circuit_element_parameters[4];

    double max_num_points_ = (double)max_num_points;
    if (max_num_points <= 0) max_num_points_ = 100.0;
    max_num_points_ = max_num_points_/0.2*max_I;

    double Voc = 10.0;
    if (intrinsic_Si_calc) {
        double base_thickness = circuit_element_parameters[3];
        if (base_thickness > 0) {
            Voc = 0.7;
            for (int i=0; i<10; ++i) {
                std::vector<double> V(1);
                V[0] = Voc;
                std::vector<double> I(1);
                double ni = circuit_element_parameters[5];
                int base_type_number = static_cast<int>(circuit_element_parameters[6]);
                double base_doping = circuit_element_parameters[0];
                double base_thickness = circuit_element_parameters[3];
                calc_intrinsic_Si_I(V.data(),V.size(),ni,VT,base_doping,base_type_number,base_thickness,1.0,I.data());
                if (I[0] >= max_I && I[0] <= max_I*1.1) break;
                Voc += VT*std::log(max_I/I[0]);
            }
        }
    }
    else {
        double I0 = circuit_element_parameters[0];
        if (I0 > 0) Voc = n*VT*std::log(max_I/I0);
    }
    

    std::vector<double> V;
    int N = (int)std::floor(max_num_points_);
    V.reserve(N + 3);

    // First 5 fixed points
    V.push_back(V_shift - 1.1);
    V.push_back(V_shift - 1.0);
    V.push_back(V_shift);

    // Now generate the log-spaced part
    // Python: np.arange(1, max_num_points)
    for (int k = 1; k < max_num_points_; ++k) {
        double frac = std::log((double)k) / std::log(max_num_points_ - 1);
        double v = V_shift + Voc * frac;
        V.push_back(v);
    }

    return V;
}

void build_forward_diode_iv(
    const double* circuit_element_parameters,
    int max_num_points,
    double* out_V,
    double* out_I,
    int* out_len,
    int abs_max_num_points,
    double op_pt_V
) {
    // circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I
    double I0 = circuit_element_parameters[0];
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = circuit_element_parameters[3];

    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,false,op_pt_V,false);

    // Now compute diode I = I0*(exp((V-V_shift)/(n*VT)) - 1)
    std::vector<double> I(V.size());
    for (size_t i = 0; i < V.size(); ++i) {
        double dv = (V[i] - V_shift) / (n * VT);
        I[i] = I0 * (std::exp(dv) - 1.0);
    }

    // Write to output buffer
    int outN = (int)V.size();
    for (int i = 0; i < outN; ++i) {
        out_V[i] = V[i];
        out_I[i] = I[i];
    }
    *out_len = outN;

}


void build_reverse_diode_iv(
    const double* circuit_element_parameters,
    int max_num_points,
    double* out_V,
    double* out_I,
    int* out_len,
    int abs_max_num_points,
    double op_pt_V)
 {
    // circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I
    double I0 = circuit_element_parameters[0];
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = circuit_element_parameters[3];

    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,false,op_pt_V,true);

    // Now compute diode I = I0*(exp((V-V_shift)/(n*VT)) - 1)
    std::vector<double> I(V.size());
    for (size_t i = 0; i < V.size(); ++i) {
        double dv = (V[i] - V_shift) / (n * VT);
        I[i] = I0 * std::exp(dv);
    }

    // Write to output buffer
    int outN = (int)V.size();
    for (int i = 0; i < outN; ++i) {
        out_V[i] = -V[outN-i-1];
        out_I[i] = -I[outN-i-1];
    }
    *out_len = outN;
}

void build_Si_intrinsic_diode_iv(
    const double* circuit_element_parameters,
    int max_num_points,
    double* out_V,
    double* out_I,
    int* out_len,
    int abs_max_num_points,
    double op_pt_V
) {
    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,true,op_pt_V,false);
    std::vector<double> I(V.size());
    double ni = circuit_element_parameters[5];
    int base_type_number = static_cast<int>(circuit_element_parameters[6]);
    double base_doping = circuit_element_parameters[0];
    double base_thickness = circuit_element_parameters[3];
    double VT = circuit_element_parameters[2];
    calc_intrinsic_Si_I(V.data(),(int)V.size(),ni,VT,base_doping,base_type_number,base_thickness,1.0,I.data());

    // Write to output buffer
    int outN = (int)V.size();
    for (int i = 0; i < outN; ++i) {
        out_V[i] = V[i];
        out_I[i] = I[i];
    }
    *out_len = outN;
}

double combine_iv_job(int connection,
    int circuit_component_type_number,
    double op_pt_V,
    int n_children,
    const IVView* children_IVs,
    const IVView* children_pc_IVs,
    int has_photon_coupling,
    int max_num_points,
    double area,
    int abs_max_num_points,
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // std::printf("cpp: combine_iv_job: n_children = %d, connection = %d\n",
    //         n_children, connection);
    // std::fflush(stdout); 

    int children_Vs_size = 0;
    for (int i=0; i < n_children; i++) children_Vs_size += children_IVs[i].length;

    if (connection == -1 && circuit_component_type_number <=4) { // CircuitElement; the two conditions are actually redundant as connection == -1 iff circuit_component_type_number <=4
       
        switch (circuit_component_type_number) {
            case 0: // CurrentSource
                build_current_source_iv(circuit_element_parameters, out_V, out_I, out_len,abs_max_num_points);
                break;
            case 1: // Resistor
                build_resistor_iv(circuit_element_parameters, out_V, out_I, out_len,abs_max_num_points);;
                break;
            case 2: // ForwardDiode
                build_forward_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len,abs_max_num_points,op_pt_V);
                break;
            case 3: // ReverseDiode
                build_reverse_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len,abs_max_num_points,op_pt_V);
                break;
            case 4: // Intrinsic Si Diode
                build_Si_intrinsic_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len,abs_max_num_points,op_pt_V);
                break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ms;
    }

    double* Vs = out_V;
    double* Is = out_I;
    int vs_len = children_Vs_size;
    // --- Series connection branch (connection == 0) ---
    if (connection == 0) {
        double I_range = 0;
        for (int i=0; i < n_children; i++) {
            int Ni = children_IVs[i].length;
            double sub_I_range = children_IVs[i].I[Ni-1]-children_IVs[i].I[0];
            if (sub_I_range > I_range) I_range = sub_I_range;
        }
        // add voltage
        if (children_Vs_size <= n_children*100) {
            int pos = 0;
            for (int i=0; i < n_children; i++) {
                int Ni = children_IVs[i].length;
                memcpy(Is + pos, children_IVs[i].I, Ni * sizeof(double));
                pos += Ni;
            }
        }
        std::vector<double> extra_Is;
        for (int iteration = 0; iteration < 2; ++iteration) {
            if (iteration == 1 && !extra_Is.empty()) {
                memcpy(Is + vs_len, extra_Is.data(), extra_Is.size() * sizeof(double));
                vs_len += extra_Is.size();
            }
            
            // auto t0 = std::chrono::high_resolution_clock::now();
            if (iteration==0) 
                if (children_Vs_size > n_children*100) // found to be optimal 
                    merge_children_kway_ptr(children_IVs, 1, n_children,Is,vs_len,abs_max_num_points);
                else {
                    std::sort(Is, Is + vs_len);
                }
            else
                std::sort(Is, Is + vs_len);

            double eps = I_range/1e8;
            auto new_end = std::unique(Is, Is + vs_len, [eps](double a, double b) {
                return std::fabs(a - b) < eps;
            });
            // auto new_end = std::unique(Is, Is + vs_len);
            vs_len = int(new_end - Is);

            std::fill(Vs, Vs + vs_len, 0.0);
            if (has_photon_coupling==0) {
                for (int i = 0; i < n_children; ++i) {  
                    int len = children_IVs[i].length;
                    if (len > 0) {
                        interp_monotonic_inc(children_IVs[i].I, children_IVs[i].V, len, Is, vs_len, Vs, true); // keeps adding
                    }
                }  
                break;
            } else {
                std::vector<double> this_V(vs_len);
                // do reverse order to allow for photon coupling
                for (int i = n_children-1; i >= 0; --i) {
                    int len = children_IVs[i].length;
                    if (len > 0) { 
                        const double* IV_table_V = children_IVs[i].V;  
                        const double* IV_table_I = children_IVs[i].I;  
                        if (i<n_children-1 && children_pc_IVs[i+1].length>0 && children_IVs[i+1].length>0) {  // need to add the current transferred by the subcell above via pc                        
                            const double* pc_IV_table_V = children_pc_IVs[i+1].V;  
                            const double* pc_IV_table_I = children_pc_IVs[i+1].I; 
                            double scale =  children_pc_IVs[i+1].scale;
                            int pc_len = children_pc_IVs[i+1].length;
                            std::vector<double> added_I(vs_len);
                            // the first time this is reached, i<n_children-1 (at least second iteration through the loop)
                            // children_lengths[i+1]>0 which means in the previous iteration, this_V.data() would have been filled already!
                            interp_monotonic_inc(pc_IV_table_V, pc_IV_table_I, pc_len, this_V.data(), (int)this_V.size(), added_I.data(), false); 
                            for (int j=0; j < added_I.size(); j++) added_I[j] *= -1*scale;
                            std::vector<double> xq(vs_len);
                            std::transform(Is, Is + vs_len,added_I.begin(),xq.begin(),std::minus<double>());
                            interp_monotonic_inc(IV_table_I, IV_table_V, len, xq.data(), (int)xq.size(), this_V.data(),false); 
                            std::vector<double> new_points(vs_len);
                            std::transform(Is, Is + vs_len,added_I.begin(),new_points.begin(),std::plus<double>());
                            extra_Is.insert(extra_Is.end(), new_points.begin(), new_points.end());
                        } else {
                            interp_monotonic_inc(IV_table_I, IV_table_V, len, Is, vs_len, this_V.data(), false); 
                        }
                        std::transform(Vs, Vs+vs_len,this_V.begin(),Vs,std::plus<double>());
                    }
                }
                if (extra_Is.empty()) break;
            }
        }
    }
    // --- parallel connection branch (connection == 1) ---
    else if (connection == 1) {
        // add current
        double left_limit = -100;
        double right_limit = 100;
        for (int i=0; i < n_children; i++) {
            int Ni = children_IVs[i].length;
            if (Ni > 0) {
                left_limit = std::min(left_limit, children_IVs[i].V[0]-100);
                right_limit = std::max(right_limit, children_IVs[i].V[Ni-1]+100);
            }
        }
        if (children_Vs_size > n_children*100) // found to be optimal 
            merge_children_kway_ptr(children_IVs, 0, n_children, Vs,vs_len,abs_max_num_points);
        else {
            int pos = 0;
            for (int i=0; i < n_children; i++) {
                int Ni = children_IVs[i].length;
                memcpy(Vs + pos, children_IVs[i].V, Ni * sizeof(double));
                pos += Ni;
            }
            std::sort(Vs, Vs+vs_len); // replace this with smart k way merge since children V's are each sorted 
        }

        for (int i=0; i < n_children; ++i) {
            const double* IV_table_V = children_IVs[i].V;
            int len = children_IVs[i].length;
            if (children_IVs[i].type_number==2) { //  forward diode
                right_limit = std::min(right_limit, IV_table_V[len-1]);
            } else if (children_IVs[i].type_number==3) {  // rev diode
                left_limit = std::max(left_limit, IV_table_V[0]);
            }
        }
        double* new_end = std::remove_if(
            Vs, Vs + vs_len,
            [left_limit, right_limit](double v) {
                return v < left_limit || v > right_limit;
            }
        );
        vs_len = int(new_end - Vs);
        double eps = 1e-6; // microvolt
        new_end = std::unique(Vs, Vs+vs_len, [eps](double a, double b) {
            return std::fabs(a - b) < eps;
        });
        // new_end = std::unique(Vs, Vs+vs_len);
        vs_len = int(new_end - Vs);

        std::fill(Is, Is + vs_len, 0.0);
        for (int i = 0; i < n_children; ++i) {  
            int len = children_IVs[i].length;
            if (len > 0) {
                interp_monotonic_inc(children_IVs[i].V, children_IVs[i].I, len, Vs, vs_len, Is, true); // keeps adding 
            }
        }  
    }
    
    if (area != 1) {
        for (int i=0; i<vs_len; i++) Is[i] *= area;
    }
    *out_len = vs_len;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
} 

// change remesh_IV to doing individual points


void remesh_IV(
    double op_pt_V,
    int refine_mode,
    int max_num_points,
    double* out_V,
    double* out_I,
    int* out_len) {

    double* Vs = out_V;
    double* Is = out_I;
    int *vs_len = out_len;

    if (*vs_len <= max_num_points)
        return;
    
    int mpp_points = 0;
    if (refine_mode==1) mpp_points = max_num_points*0.1;

    int n = static_cast<int>(*vs_len);
    std::vector<double> accum_abs_dir_change(n-1); 
    std::vector<double> accum_abs_dir_change_near_mpp(n-1); 
    accum_abs_dir_change.push_back(0.0);
    double V_range = Vs[n-1];
    double left_V = 0.05*Vs[0];
    double right_V = 0.05*Vs[n-1];
    double I_range = std::min(std::abs(Is[n-1]),std::abs(Is[0]));
    double V_closest_to_SC = 1000000;
    int idx_V_closest_to_SC = 0;
    double V_closest_to_SC_right = 1000000;
    int idx_V_closest_to_SC_right = 0;
    double V_closest_to_SC_left = 1000000;
    int idx_V_closest_to_SC_left = 0;
    double sqrt_half = std::sqrt(0.5);
    double last_unit_vector_x, last_unit_vector_y;
    for (int i = 0; i < n - 1; ++i) {
        if (V_closest_to_SC > std::abs(Vs[i])) {
            V_closest_to_SC = std::abs(Vs[i]);
            idx_V_closest_to_SC = i;
        }
        double absdiff = std::abs(Vs[i]-right_V);
        if (V_closest_to_SC_right > absdiff) {
            V_closest_to_SC_right = absdiff;
            idx_V_closest_to_SC_right = i;
        }
        absdiff = std::abs(Vs[i]-left_V);
        if (V_closest_to_SC_left > absdiff) {
            V_closest_to_SC_left = absdiff;
            idx_V_closest_to_SC_left = i;
        }
        double unit_vector_x = (Vs[i+1] - Vs[i])/V_range;
        double unit_vector_y = (Is[i+1] - Is[i])/I_range;
        double mag = std::sqrt((unit_vector_x*unit_vector_x + unit_vector_y*unit_vector_y));
        unit_vector_x /= mag;
        unit_vector_y /= mag;
        int bad_point = false;
        if (!std::isfinite(unit_vector_x) || !std::isfinite(unit_vector_y)) { // catches NaN, +inf, -inf
            bad_point = true;
            unit_vector_x = sqrt_half;
            unit_vector_y = sqrt_half;
        }
        if (i > 0) {
            double change = 0;
            double fudge_factor1 = 1;
            if (!bad_point) {
                double dx = unit_vector_x - last_unit_vector_x;
                double dy = unit_vector_y - last_unit_vector_y;
                fudge_factor1 = 1;
                if (Vs[i] < 0) fudge_factor1 = 0.1; // we care much less about reverse characteristics
                change = std::sqrt(dx*dx + dy*dy);
            }
            accum_abs_dir_change[i] = accum_abs_dir_change[i-1] + fudge_factor1*change;
            if (refine_mode==1 && mpp_points > 0) {
                double change_ = 0;
                if (std::abs(op_pt_V-Vs[i])<std::abs(op_pt_V)*0.05) change_ = change;
                accum_abs_dir_change_near_mpp[i] = accum_abs_dir_change_near_mpp[i-1] + change_;
            }
        }
        last_unit_vector_x = unit_vector_x;
        last_unit_vector_y = unit_vector_y;
    }
    double variation_segment = accum_abs_dir_change[n-2]/(max_num_points-2);
    double variation_segment_mpp = 0.0;
    if (refine_mode == 1 && mpp_points > 0) {
        variation_segment_mpp =
            accum_abs_dir_change_near_mpp[n - 2] / mpp_points;
    }
    std::vector<int> idx;
    idx.reserve(max_num_points+100+mpp_points);
    idx.push_back(0);
    int count = 1;
    int countmpp = 1;
    for (int i = 1; i < n - 1; ++i) {
        if (accum_abs_dir_change[i] >= count * variation_segment) {
            int last_index = idx.size()-1;
            if (last_index >= 0 && i > idx_V_closest_to_SC_left && idx_V_closest_to_SC_left > idx[last_index]) 
                idx.push_back(idx_V_closest_to_SC_left);  // just also capture points closest to SC to keep Isc accurate
            if (last_index >= 0 && i > idx_V_closest_to_SC && idx_V_closest_to_SC > idx[last_index]) 
                idx.push_back(idx_V_closest_to_SC);  // just also capture points closest to SC to keep Isc accurate
            if (last_index >= 0 && i > idx_V_closest_to_SC_right && idx_V_closest_to_SC_right > idx[last_index]) 
                idx.push_back(idx_V_closest_to_SC_right);  // just also capture points closest to SC to keep Isc accurate
            idx.push_back(i);
            while (accum_abs_dir_change[i] >= count * variation_segment)
                ++count;
        } else if (refine_mode==1 && mpp_points > 0 && variation_segment_mpp>0 && accum_abs_dir_change_near_mpp[i] >= countmpp * variation_segment_mpp) {
            idx.push_back(i);
            while (accum_abs_dir_change_near_mpp[i] >= countmpp * variation_segment_mpp)
                ++countmpp;
        }
    }
    idx.push_back(n-1);

    // remesh
    int n_out = (int)idx.size();
    for (int i = 0; i < n_out; ++i) {
        int k = idx[i];
        out_V[i] = Vs[k];
        double I_val = Is[k];
        out_I[i] = I_val;
    }
    *out_len = n_out;
} 


double combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, int parallel) {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (parallel==1 && n_jobs>1) {
        int max_threads = omp_get_max_threads();
        int num_threads = max_threads;
        if (n_jobs < max_threads*2) num_threads = max_threads/2;
        if (n_jobs < max_threads) num_threads = max_threads/4;
        if (n_jobs < max_threads/4) num_threads = n_jobs;

        #pragma omp parallel for num_threads(num_threads)
        for (int j = 0; j < n_jobs; ++j) {
            IVJobDesc& job = jobs[j];
            combine_iv_job(
                job.connection,
                job.circuit_component_type_number,
                job.op_pt_V,
                job.n_children,
                job.children_IVs,
                job.children_pc_IVs,
                job.has_photon_coupling,
                job.max_num_points,
                job.area,
                job.abs_max_num_points,
                job.circuit_element_parameters,
                job.out_V,
                job.out_I,
                job.out_len
            );
        }
    }
    else 
        for (int j = 0; j < n_jobs; ++j) {
            IVJobDesc& job = jobs[j];
            combine_iv_job(
                job.connection,
                job.circuit_component_type_number,
                job.op_pt_V,
                job.n_children,
                job.children_IVs,
                job.children_pc_IVs,
                job.has_photon_coupling,
                job.max_num_points,
                job.area,
                job.abs_max_num_points,
                job.circuit_element_parameters,
                job.out_V,
                job.out_I,
                job.out_len
            );
        }

    int num_jobs_need_remesh = 0;
    for (int j = 0; j < n_jobs; ++j) 
        if (jobs[j].max_num_points > 2) 
            num_jobs_need_remesh++;
    if (num_jobs_need_remesh > 0) {
        if (parallel==1 && num_jobs_need_remesh>1) {
            int num_threads = 32;
            if (num_jobs_need_remesh < 16*4) num_threads = 16;
            if (num_jobs_need_remesh < 8*4) num_threads = 8;
            if (num_jobs_need_remesh < 8) num_threads = num_jobs_need_remesh;
            // if (n_jobs > 8) num_threads = 16;
            // if (n_jobs > 16) num_threads = 32;

            #pragma omp parallel for num_threads(num_threads)
            for (int j = 0; j < n_jobs; ++j) {
                IVJobDesc& job = jobs[j];
                if (jobs[j].max_num_points > 2) 
                    remesh_IV(
                        job.op_pt_V,
                        job.refine_mode,
                        job.max_num_points,
                        job.out_V,
                        job.out_I,
                        job.out_len
                    );
            }
        }
        else 
            for (int j = 0; j < n_jobs; ++j) {
                IVJobDesc& job = jobs[j];
                if (jobs[j].max_num_points > 2) 
                    remesh_IV(
                        job.op_pt_V,
                        job.refine_mode,
                        job.max_num_points,
                        job.out_V,
                        job.out_I,
                        job.out_len
                    );
            }
    }
    

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;
}

}// extern "C"
