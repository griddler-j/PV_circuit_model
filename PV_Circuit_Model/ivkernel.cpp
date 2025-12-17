// ivkernel.cpp
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdio>
// ---- Optional OpenMP support ----
#ifdef _OPENMP
  #include <omp.h>
#else
  // Build without OpenMP: provide minimal stubs so the code links.
  static inline int omp_get_max_threads() { return 1; }
  static inline int omp_get_thread_num()  { return 0; }
#endif
#include <chrono>
#include "ivkernel.h"
#include <numeric>
#include <queue>
#include <fstream>

extern "C" {

static std::vector<double> g_bgn_x;
static std::vector<double> g_bgn_y;
static double q;
static bool g_bgn_loaded = false;
static bool constants_loaded = false;

int ivkernel_has_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

void ivkernel_set_bandgap_table(const double* x, const double* y, int n)
{
    if (!x || !y || n <= 0) {
        // you can throw, assert, or just ignore
        return;
    }
    g_bgn_x.assign(x, x + n);
    g_bgn_y.assign(y, y + n);
    g_bgn_loaded = true;
}

void ivkernel_set_q(double q_value) {
    q = q_value;
}

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
    const double* __restrict circuit_element_parameters,
    double* __restrict out_V,
    double* __restrict out_I,
    int* out_len
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
    const double* __restrict circuit_element_parameters,
    double* __restrict out_V,
    double* __restrict out_I,
    int* out_len
) {
    double cond = circuit_element_parameters[0];
    int n = 2;
    double step = 1e-3;
    if (cond > 1)
        step /= cond;
    for (int i=0; i < n; ++i) {
        out_V[i] = -step + i*step*2;
        out_I[i] = cond*out_V[i];
    }
    *out_len = n;
}

void interp_monotonic_inc(
    const double* __restrict x,   // size n, increasing
    const double* __restrict y,   // size n
    int n,
    const double* __restrict xq,  // size m, increasing
    int m,
    double* __restrict yq,         // size m, output
    bool additive,        // true: adds to yq
    int method, // 0-linear, 1-smooth slope, 2-take max, 3-take min
    double extrapolation_left_slope,
    double extrapolation_right_slope
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
        slope_left = extrapolation_left_slope;
        if (slope_left < 0) // -1 means use data slope
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
        slope_right = extrapolation_right_slope;
        if (slope_right < 0) // -1 means use data slope
            slope_right = (y[n-1] - y[n-2])/(x[n-1] - x[n-2]);
        while (j_left <= j_right && xq[j_right] >= x[n-1]) {
            yq[j_right] = (additive? yq[j_right]:0.0) + y[n-1] + (xq[j_right] - x[n-1])*slope_right;
            --j_right;
        }
    }

    // --- Main interpolation region: x[0] < xq < x[n-1] ---
    int i = 0; // segment index for x
    int prev_i = -1;
    double slope, left_slope, right_slope;
    int this_slope_i = -1, right_slope_i = -1;
    for (int j = j_left; j <= j_right; ++j) {
        double xj = xq[j];
        // Advance i until x[i] <= xj <= x[i+1]
        while (i + 1 < n - 1 && x[i+1] < xj) 
            ++i;
        if (i > prev_i) {
            if (method>0) {
                if (this_slope_i==i-1 && std::isfinite(slope))
                    left_slope = slope;
                else {
                    if (i > 0) 
                        left_slope = (y[i]-y[i-1])/(x[i]-x[i-1]);
                    else
                        left_slope = extrapolation_left_slope;
                }

                if (right_slope_i==i && std::isfinite(right_slope))
                    slope = right_slope;
                else
                    slope = (y[i+1]-y[i])/(x[i+1]-x[i]);
                
                this_slope_i = i;

                if (i < n-2) 
                    right_slope = (y[i+2]-y[i+1])/(x[i+2]-x[i+1]);
                else
                    right_slope = extrapolation_right_slope;
                right_slope_i = i + 1;
            } else 
                slope = (y[i+1]-y[i])/(x[i+1]-x[i]);

            if (!std::isfinite(left_slope)) left_slope = slope;
            if (!std::isfinite(right_slope)) right_slope = slope;
        }
        if (!std::isfinite(slope)) {
            yq[j] = (additive? yq[j]:0.0) + y[i];
            if (method==2) 
                yq[j] = (additive? yq[j]:0.0) + y[i+1];
            else
                yq[j] = (additive? yq[j]:0.0) + y[i];
        }
        else if (method==1) {
            double delta_x = 0.5*(x[i+1]-x[i]);
            double delta_x_left = xj - x[i];
            double half_slope_left = 0.5*(slope + left_slope);
            double half_slope_right = 0.5*(slope + right_slope);
            double y_ref_left = 0.5*(slope - half_slope_left)/delta_x*(delta_x_left*delta_x_left)+half_slope_left*delta_x_left+y[i];
            double delta_x_right = xj - x[i+1];
            double y_ref_right = 0.5*(right_slope - half_slope_right)/delta_x*(delta_x_right*delta_x_right)+half_slope_right*delta_x_right+y[i+1];
            double yadd = (-y_ref_left*delta_x_right+y_ref_right*delta_x_left)/(2*delta_x);
            if (yadd < y[i]) yadd = y[i];
            if (yadd > y[i+1]) yadd = y[i+1];
            yq[j] = (additive? yq[j]:0.0) + yadd;
            if (!std::isfinite(yq[j]) || (j>0 && yq[j]<yq[j-1])) yq[j] = yq[j-1];
        } else if (method==2 || method==3) {
            double yadd;
            if (!std::isfinite(slope) || !std::isfinite(left_slope) || !std::isfinite(right_slope)) {
                if (method==2)
                    yadd = y[i+1];
                else
                    yadd = y[i];
            } else {
                double y_ref_mid = y[i] + slope*(xj - x[i]);
                double y_ref_left = y[i] + left_slope*(xj - x[i]);
                double y_ref_right = y[i+1] + right_slope*(xj - x[i+1]);
                if (method==2) {
                    if (slope > left_slope && slope > right_slope) { // has an inflection point
                        yadd = std::max(y_ref_mid, y_ref_right);
                    } else {
                        yadd = std::min(y_ref_left, y_ref_right);
                        yadd = std::max(y_ref_mid, yadd);
                    }
                }
                else {
                    if (slope > left_slope && slope > right_slope) { // has an inflection point
                        yadd = std::min(y_ref_mid, y_ref_left);
                    } else {
                        yadd = std::max(y_ref_left, y_ref_right);
                        yadd = std::min(y_ref_mid, yadd);
                    }
                }
            }
            if (yadd < y[i]) yadd = y[i];
            if (yadd > y[i+1]) yadd = y[i+1];
            yq[j] = (additive? yq[j]:0.0) + yadd;
        } else 
            yq[j] = (additive? yq[j]:0.0) + y[i] + slope*(xj - x[i]);

        prev_i = i;
    }
}

void calc_intrinsic_Si_I(
    const double* __restrict V,          // input voltages, length n_V
    int n_V,
    double ni,
    double  VT,
    double base_doping,
    int base_type_number,   // 0-p, 1-n
    double base_thickness,
    double area,               // area is actually always 1
    double* __restrict out_I,             // output: I(V) or dI/dV, length n_V
    bool additive,
    double* delta_n, double* BGN, double* pn
) {
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

    double termA = 2.5e-31 * geeh * n0;
    double termB = 8.5e-32 * gehh * p0;

    if (n_V==1) { // just one point 
        double expv = std::exp(V[0]/VT);
        double pn0 = ni * ni * expv;
        double delta_n0 = 0.5 * (-N_doping + std::sqrt(N_doping*N_doping + 4.0*ni*ni*expv));
        double BGN0;
        interp_monotonic_inc(
            g_bgn_x.data(),
            g_bgn_y.data(),
            (int)g_bgn_x.size(),
            &delta_n0,
            n_V,
            &BGN0,
            false,        // overwrite
            0,0,0
        );
        // ni_eff = ni * exp(BGN/(2*VT))
        double ni_eff = ni * std::exp(BGN0 / (2.0 * VT));

        // Recombination prefactor
        double termC = 3e-29   * std::pow(delta_n0, 0.92);
        double coeff = termA + termB + termC + Brel * Blow;

        // intrinsic_recomb = (pn - ni_eff^2)*coeff
        double intrinsic_recomb = (pn0 - ni_eff*ni_eff) * coeff;

        // I(V) = q * intrinsic_recomb * thickness * area
        out_I[0] = (additive? out_I[0]:0.0) + q * intrinsic_recomb * base_thickness * area;

    } else {
        // Compute delta_n first
        for (int i = 0; i < n_V; ++i) {
            double expv = std::exp(V[i]/VT);
            pn[i] = ni * ni * expv;
            delta_n[i] = 0.5 * (-N_doping + std::sqrt(N_doping*N_doping + 4.0*ni*ni*expv));
        }

        // Vector interpolation:
        interp_monotonic_inc(
            g_bgn_x.data(),
            g_bgn_y.data(),
            (int)g_bgn_x.size(),
            delta_n,
            n_V,
            BGN,
            false,        // overwrite
            0,0,0
        );

        for (int i = 0; i < n_V; ++i) {
            // ni_eff = ni * exp(BGN/(2*VT))
            double ni_eff = ni * std::exp(BGN[i] / (2.0 * VT));

            // Recombination prefactor
            double termC = 3e-29   * std::pow(delta_n[i], 0.92);
            double coeff = termA + termB + termC + Brel * Blow;

            // intrinsic_recomb = (pn - ni_eff^2)*coeff
            double intrinsic_recomb = (pn[i] - ni_eff*ni_eff) * coeff;

            // I(V) = q * intrinsic_recomb * thickness * area
            out_I[i] = (additive? out_I[i]:0.0) + q * intrinsic_recomb * base_thickness * area;
        }
    }
}

int get_V_range(const double* __restrict circuit_element_parameters,
    int max_num_points,bool intrinsic_Si_calc,bool is_reverse_diode,double* out_V,double* delta_n, double* BGN, double* pn) {
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
                double V = Voc;
                double I;
                double ni = circuit_element_parameters[5];
                int base_type_number = static_cast<int>(circuit_element_parameters[6]);
                double base_doping = circuit_element_parameters[0];
                double base_thickness = circuit_element_parameters[3];
                calc_intrinsic_Si_I(&V,1,ni,VT,base_doping,base_type_number,base_thickness,1.0,&I,false,delta_n, BGN, pn);
                if (I >= max_I && I <= max_I*1.1) break;
                Voc += VT*std::log(max_I/I);
            }
        }
    }
    else {
        double I0 = circuit_element_parameters[0];
        if (I0 > 0) Voc = n*VT*std::log(max_I/I0);
    }
    
    int N = (int)std::floor(max_num_points_);

    if (is_reverse_diode) { // reverse order
        out_V[N+4] = -V_shift + 11;
        out_V[N+3] = -V_shift + 10.1;
        out_V[N+2] = -V_shift + 1.1;
        out_V[N+1] = -V_shift + 1.0;
        out_V[N] = -V_shift;
        int pos = N-1;
        for (int k = 1; k < N + 1; ++k) {
            double frac = std::log((double)k) / std::log(max_num_points_ - 1);
            out_V[pos] = -V_shift - Voc * frac;
            --pos;
        }
    } else {
        out_V[0] = V_shift - 11;
        out_V[1] = V_shift - 10.1;
        out_V[2] = V_shift - 1.1;
        out_V[3] = V_shift - 1.0;
        out_V[4] = V_shift;
        int pos = 5;
        for (int k = 1; k < N + 1; ++k) {
            double frac = std::log((double)k) / std::log(max_num_points_ - 1);
            out_V[pos] = V_shift + Voc * frac;
            ++pos;
        }
    }
    return N+5;
}

inline double calc_forward_diode_I(double I0, double n, double VT, double V_shift, double V) {
    if ((V - V_shift) < -10) // approx
        return 0.0;
    return I0 * (std::exp((V - V_shift) / (n * VT)) - 1.0);
}

inline double calc_reverse_diode_I(double I0, double n, double VT, double V_shift, double V) {
    if ((-V - V_shift) < -10) // approx
        return 0.0;
    return -I0 * std::exp((-V - V_shift) / (n * VT));
}

// modified such that diodes get exact current rather than interpolation
void interp_monotonic_inc_scalar(
    const double** __restrict xs,   // size n, strictly increasing
    const double** __restrict ys,   // size n
    const int* __restrict ns,
    const double* __restrict xqs,         // single query points
    double** __restrict yqs,        // output (single values)
    int n_jobs,
    int parallel,
    const double (*element_params)[8],
    int* __restrict circuit_type_number
) {
    int max_threads = omp_get_max_threads();
    int num_threads = max_threads;
    if (n_jobs < max_threads*2) num_threads = max_threads/2;
    if (n_jobs < max_threads) num_threads = max_threads/4;
    if (n_jobs < max_threads/4) num_threads = n_jobs;

    #pragma omp parallel for num_threads(num_threads) if(parallel && n_jobs>1)
    for (int i = 0; i < n_jobs; i ++) {
        const double* __restrict x = xs[i];
        const double* __restrict y = ys[i];
        const double* __restrict diode_param = element_params[i];
        int type_number = circuit_type_number[i];
        int n = ns[i];
        double xq = xqs[i];
        double* __restrict yq = yqs[i];

        if (!x || !y || !yq || n <= 0) continue;

        if (type_number==2 || type_number==3) {
            double I0 = diode_param[0];
            double n = diode_param[1];
            double VT = diode_param[2];
            double V_shift = diode_param[3];
            if (type_number==2) 
                *yq = calc_forward_diode_I(I0, n, VT, V_shift, xq);
            else
                *yq = calc_reverse_diode_I(I0, n, VT, V_shift, xq);
            continue;
        } else if (type_number==4) {
            double ni = diode_param[5];
            int base_type_number = static_cast<int>(diode_param[6]);
            double base_doping = diode_param[0];
            double base_thickness = diode_param[3];
            double VT = diode_param[2];
            double V = xq;
            calc_intrinsic_Si_I(&V,1,ni,VT,base_doping,base_type_number,base_thickness,1,yq,false,nullptr,nullptr,nullptr);
            continue;
        }

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

void build_diode_iv(
    const double* __restrict circuit_element_parameters,
    int max_num_points,
    double* __restrict out_V,
    double* __restrict out_I,
    int* out_len,
    bool is_reverse_diode
) {
    // circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I
    double I0 = circuit_element_parameters[0];
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = circuit_element_parameters[3];

    int size = get_V_range(circuit_element_parameters, max_num_points,false,is_reverse_diode,out_V,nullptr,nullptr,nullptr);

    if (!is_reverse_diode) {
        for (size_t i = 0; i < size; ++i) 
            out_I[i] = calc_forward_diode_I(I0, n, VT, V_shift, out_V[i]);
    } else {
        for (size_t i = 0; i < size; ++i) 
            out_I[i] = calc_reverse_diode_I(I0, n, VT, V_shift, out_V[i]);
    }
    *out_len = size;
}

void build_Si_intrinsic_diode_iv(
    const double* __restrict circuit_element_parameters,
    int max_num_points,
    double* __restrict out_V,
    double* __restrict out_I,
    int* out_len,
    double* delta_n, double* BGN, double* pn
) {
    int size = get_V_range(circuit_element_parameters, max_num_points,true,false,out_V,delta_n,BGN,pn);
    double ni = circuit_element_parameters[5];
    int base_type_number = static_cast<int>(circuit_element_parameters[6]);
    double base_doping = circuit_element_parameters[0];
    double base_thickness = circuit_element_parameters[3];
    double VT = circuit_element_parameters[2];
    calc_intrinsic_Si_I(out_V,size,ni,VT,base_doping,base_type_number,base_thickness,1.0,out_I,false,delta_n, BGN, pn);
    *out_len = size;
}

void combine_iv_job(int connection,
    int circuit_component_type_number,
    int n_children,
    const IVView* this_IV,
    const IVView* children_IVs,
    const IVView* children_pc_IVs,
    int has_photon_coupling,
    int max_num_points,
    double area,
    int abs_max_num_points,
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len,
    bool* out_extrapolation_allowed,
    double* out_extrapolation_dI_dV,
    bool* out_has_I_domain_limit,
    int refine_mode,
    int all_children_are_elements,
    double op_pt_V,
    double bottom_up_op_pt_V,
    double normalized_op_pt_V,
    int refinement_points,
    int interp_method, int use_existing_grid, double refine_V_half_width,
    double* delta_n, double* BGN, double* pn, 
    double* extra_Is, double* this_V, double* added_I, 
    double* xq, double* new_points) {

    out_extrapolation_allowed[0] = true; 
    out_extrapolation_allowed[1] = true;
    out_has_I_domain_limit[0] = false; 
    out_has_I_domain_limit[1] = false;
    out_extrapolation_dI_dV[0] = 0;
    out_extrapolation_dI_dV[1] = 0;

    int children_Vs_size = 0;
    for (int i=0; i < n_children; i++) children_Vs_size += children_IVs[i].length;

    if (connection == -1 && circuit_component_type_number <=4) { // CircuitElement; the two conditions are actually redundant as connection == -1 iff circuit_component_type_number <=4
        switch (circuit_component_type_number) {
            case 0: // CurrentSource
                build_current_source_iv(circuit_element_parameters, out_V, out_I, out_len);
                out_has_I_domain_limit[0] = true;
                out_has_I_domain_limit[1] = true;
                out_extrapolation_dI_dV[0] = 0;
                out_extrapolation_dI_dV[1] = 0;
                break;
            case 1: // Resistor
                build_resistor_iv(circuit_element_parameters, out_V, out_I, out_len);
                out_extrapolation_dI_dV[0] = circuit_element_parameters[0]; // cond
                out_extrapolation_dI_dV[1] = circuit_element_parameters[0];
                break;
            case 2: // ForwardDiode
                build_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len,false);
                out_has_I_domain_limit[0] = true;
                out_extrapolation_allowed[1] = false;
                out_extrapolation_dI_dV[0] = 0;
                out_extrapolation_dI_dV[1] = (out_I[*out_len-1]-out_I[*out_len-2])/(out_V[*out_len-1]-out_V[*out_len-2]);
                break;
            case 3: // ReverseDiode
                build_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len,true);
                out_has_I_domain_limit[1] = true;
                out_extrapolation_allowed[0] = false;
                out_extrapolation_dI_dV[1] = 0;
                out_extrapolation_dI_dV[0] = (out_I[1]-out_I[0])/(out_V[1]-out_V[0]);
                break;
            case 4: // Intrinsic Si Diode
                build_Si_intrinsic_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len, delta_n, BGN, pn);
                out_has_I_domain_limit[0] = true;
                out_extrapolation_allowed[1] = false;
                out_extrapolation_dI_dV[0] = 0;
                out_extrapolation_dI_dV[1] = (out_I[*out_len-1]-out_I[*out_len-2])/(out_V[*out_len-1]-out_V[*out_len-2]);
                break;
        }
        return;
    }

    double* Vs = out_V;
    double* Is = out_I;
    int vs_len = children_Vs_size;
    // --- Series connection branch (connection == 0) ---
    if (connection == 0) {
        if (use_existing_grid==1) {
            int N = (*this_IV).length;
            memcpy(Is, (*this_IV).I, N * sizeof(double));
            if (area != 1) {
                for (int i=0; i<N; i++) Is[i] /= area;
            }
            vs_len = N;
        }
        double I_range = 0;
        if (use_existing_grid!=1) {
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
        }
        int extra_Is_length = 0;
        for (int iteration = 0; iteration < 2; ++iteration) {
            if (use_existing_grid!=1) {
                if (iteration == 1 && extra_Is_length>0) {
                    memcpy(Is + vs_len, extra_Is, extra_Is_length * sizeof(double));
                    vs_len += extra_Is_length;
                }
                
                if (iteration==0) 
                    if (children_Vs_size > n_children*100) // found to be optimal 
                        merge_children_kway_ptr(children_IVs, 1, n_children,Is,vs_len,abs_max_num_points);
                    else {
                        std::sort(Is, Is + vs_len);
                    }
                else
                    std::sort(Is, Is + vs_len);

                double left_limit = Is[0]-1;
                double right_limit = Is[vs_len-1]+1;
                for (int i=0; i < n_children; ++i) {
                    const double* IV_table_I = children_IVs[i].I;
                    int len = children_IVs[i].length;
                    if (!children_IVs[i].right_extrapolation_allowed) {
                        right_limit = std::min(right_limit, IV_table_I[len-1]);
                        out_extrapolation_allowed[1] = false;
                    }  
                    if (out_extrapolation_dI_dV[1]==0)
                        out_extrapolation_dI_dV[1] = children_IVs[i].extrapolation_dI_dV[1];
                    else
                        out_extrapolation_dI_dV[1] = 1/(1/out_extrapolation_dI_dV[1] + 1/children_IVs[i].extrapolation_dI_dV[1]);
                    if (!children_IVs[i].left_extrapolation_allowed) {
                        left_limit = std::max(left_limit, IV_table_I[0]);
                        out_extrapolation_allowed[0] = false;
                    } 
                    if (out_extrapolation_dI_dV[0]==0)
                        out_extrapolation_dI_dV[0] = children_IVs[i].extrapolation_dI_dV[0];
                    else
                        out_extrapolation_dI_dV[0] = 1/(1/out_extrapolation_dI_dV[0] + 1/children_IVs[i].extrapolation_dI_dV[0]);
                    if (children_IVs[i].has_upper_I_domain_limit) {
                        right_limit = std::min(right_limit, IV_table_I[len-1]);
                        out_has_I_domain_limit[1] = true;
                    }
                    if (children_IVs[i].has_lower_I_domain_limit) {
                        left_limit = std::max(left_limit, IV_table_I[0]);
                        out_has_I_domain_limit[0] = true;
                    }
                }

                // guard against no domain
                if (left_limit >= right_limit) {
                    *out_len = 0;
                    return;
                }

                double* new_end = std::remove_if(
                Is, Is + vs_len,
                    [left_limit, right_limit](double i_) {
                        return i_ < left_limit || i_ > right_limit;
                    }
                );
                vs_len = int(new_end - Is);

                // double eps = I_range/1e9; // nano amps
                // new_end = std::unique(Is, Is + vs_len, [eps](double a, double b) {
                //     return std::fabs(a - b) < eps;
                // });
                new_end = std::unique(Is, Is + vs_len);
                vs_len = int(new_end - Is);
            }

            std::fill(Vs, Vs + vs_len, 0.0);
            if (has_photon_coupling==0) {
                for (int i = 0; i < n_children; ++i) {  
                    double left_inv_slope = 1;
                    if (children_IVs[i].extrapolation_dI_dV[0]>0)
                        left_inv_slope =  1/children_IVs[i].extrapolation_dI_dV[0];
                    double right_inv_slope = 1;
                    if (children_IVs[i].extrapolation_dI_dV[1]>0)
                        right_inv_slope =  1/children_IVs[i].extrapolation_dI_dV[1];

                    int len = children_IVs[i].length;
                    if (len > 0) {
                        interp_monotonic_inc(children_IVs[i].I, children_IVs[i].V, len,
                            Is, vs_len, Vs, true, interp_method, left_inv_slope, right_inv_slope); // keeps adding
                    }
                }  
                break;
            } else {
                // do reverse order to allow for photon coupling
                for (int i = n_children-1; i >= 0; --i) {
                    int len = children_IVs[i].length;
                    if (len > 0) { 
                        double left_inv_slope = 1;
                        if (children_IVs[i].extrapolation_dI_dV[0]>0)
                            left_inv_slope =  1/children_IVs[i].extrapolation_dI_dV[0];
                        double right_inv_slope = 1;
                        if (children_IVs[i].extrapolation_dI_dV[1]>0)
                            right_inv_slope =  1/children_IVs[i].extrapolation_dI_dV[1];
                        const double* IV_table_V = children_IVs[i].V;  
                        const double* IV_table_I = children_IVs[i].I;  
                        if (i<n_children-1 && children_pc_IVs[i+1].length>0 && children_IVs[i+1].length>0) {  // need to add the current transferred by the subcell above via pc                        
                            const double* pc_IV_table_V = children_pc_IVs[i+1].V;  
                            const double* pc_IV_table_I = children_pc_IVs[i+1].I; 
                            double scale =  children_pc_IVs[i+1].scale;
                            int pc_len = children_pc_IVs[i+1].length;
                            // the first time this is reached, i<n_children-1 (at least second iteration through the loop)
                            // children_lengths[i+1]>0 which means in the previous iteration, this_V.data() would have been filled already!

                            interp_monotonic_inc(pc_IV_table_V, pc_IV_table_I, pc_len, 
                                this_V, vs_len, added_I, false, 0, -1,-1); 
                            for (int j=0; j < vs_len; j++) added_I[j] *= -1*scale;
                            for (int j=0; j < vs_len; j++) xq[j] = Is[j] - added_I[j];
                            interp_monotonic_inc(IV_table_I, IV_table_V, len, 
                                 xq, vs_len, this_V,false, interp_method, left_inv_slope, right_inv_slope); 
                            if (use_existing_grid!=1 && iteration==0) {
                                for (int j=0; j < vs_len; j++) new_points[j] = Is[j] + added_I[j];
                                memcpy(extra_Is + extra_Is_length, new_points, vs_len * sizeof(double));
                                extra_Is_length += vs_len;
                            }
                        } else {
                            interp_monotonic_inc(IV_table_I, IV_table_V, len, 
                                Is, vs_len, this_V,false, interp_method, left_inv_slope, right_inv_slope); 
                        }
                        for (int j=0; j < vs_len; j++) Vs[j] += this_V[j];
                    }
                }
                if (extra_Is_length == 0) break;
                if (use_existing_grid==1) break;
            }
        }
    }
    // --- parallel connection branch (connection == 1) ---
    else if (connection == 1) {
        // current is negated when plotting power, so upperbound becomes lowerbound and vice versa
        int interp_method_ = interp_method;
        if (interp_method==3) 
            interp_method_ = 2;
        else if (interp_method==2)
            interp_method_ = 3;

        if (use_existing_grid==1) {
            int Ni = (*this_IV).length;
            memcpy(Vs, (*this_IV).V, Ni * sizeof(double));
            vs_len = Ni;
        } else {
            if (children_Vs_size > n_children*100 && (refine_mode==0 || all_children_are_elements==0)) // found to be optimal 
                merge_children_kway_ptr(children_IVs, 0, n_children, Vs,vs_len,abs_max_num_points);
            else {
                int pos = 0;
                for (int i=0; i < n_children; i++) {
                    int Ni = children_IVs[i].length;
                    memcpy(Vs + pos, children_IVs[i].V, Ni * sizeof(double));
                    pos += Ni;
                }
                if (refine_mode==1 && all_children_are_elements==1) {  // add more points near the operating point
                    double left_V = op_pt_V;
                    double right_V = bottom_up_op_pt_V;
                    if (left_V > right_V) {
                        left_V = bottom_up_op_pt_V;
                        right_V = op_pt_V;
                    }
                    left_V -= normalized_op_pt_V*refine_V_half_width;
                    right_V += normalized_op_pt_V*refine_V_half_width;
                    double step = (right_V - left_V)/(refinement_points-1);
                    for (int i=0; i<refinement_points; ++i) 
                        Vs[pos+i] = left_V + step*i;
                    vs_len += refinement_points;
                }
                std::sort(Vs, Vs+vs_len); // replace this with smart k way merge since children V's are each sorted 
            }

            out_has_I_domain_limit[0] = true;
            out_has_I_domain_limit[1] = true;
            double left_limit = Vs[0]-1;
            double right_limit = Vs[vs_len-1]+1;
            for (int i=0; i < n_children; ++i) {
                const double* IV_table_V = children_IVs[i].V;
                int len = children_IVs[i].length;
                if (!children_IVs[i].right_extrapolation_allowed) {
                    right_limit = std::min(right_limit, IV_table_V[len-1]);
                    out_extrapolation_allowed[1] = false;
                } 
                out_extrapolation_dI_dV[1] += children_IVs[i].extrapolation_dI_dV[1];
                if (!children_IVs[i].left_extrapolation_allowed) {
                    left_limit = std::max(left_limit, IV_table_V[0]);
                    out_extrapolation_allowed[0] = false;
                } 
                out_extrapolation_dI_dV[0] += children_IVs[i].extrapolation_dI_dV[0];
                if (!children_IVs[i].has_lower_I_domain_limit) 
                    out_has_I_domain_limit[0] = false; // relief
                if (!children_IVs[i].has_upper_I_domain_limit) 
                    out_has_I_domain_limit[1] = false; // relief
            }

            double* new_end = std::remove_if(
                Vs, Vs + vs_len,
                [left_limit, right_limit](double v) {
                    return v < left_limit || v > right_limit;
                }
            );
            vs_len = int(new_end - Vs);

            if (vs_len > 1) {
                double eps = 1e-9; // nanovolt
                double last_val = Vs[vs_len - 1];
                double* new_end = std::unique(Vs, Vs+vs_len - 1,
                    [eps](double a, double b) {
                        return std::fabs(a - b) < eps;
                    });
                // always keep the original last point
                if (std::fabs(last_val - new_end[-1]) >= eps)
                    *new_end++ = last_val;
                vs_len = int(new_end - Vs);
            }
        }

        std::fill(Is, Is + vs_len, 0.0);
        for (int i = 0; i < n_children; ++i) {  
            int len = children_IVs[i].length;
            if (len > 0) {
                if (children_IVs[i].type_number==2 || children_IVs[i].type_number==3) { // diode
                    double I0 = children_IVs[i].element_params[0];
                    double n = children_IVs[i].element_params[1];
                    double VT = children_IVs[i].element_params[2];
                    double V_shift = children_IVs[i].element_params[3];
                    for (int k = 0; k < vs_len; ++k) 
                        if (children_IVs[i].type_number==2) 
                            Is[k] += calc_forward_diode_I(I0,n,VT,V_shift,Vs[k]); // forward diode
                        else
                            Is[k] += calc_reverse_diode_I(I0,n,VT,V_shift,Vs[k]); // reverse diode
                } else if (children_IVs[i].type_number==4) { // intrinsic silicon diode
                    double base_doping = children_IVs[i].element_params[0];
                    double VT = children_IVs[i].element_params[1];
                    double base_thickness = children_IVs[i].element_params[2];
                    double ni = children_IVs[i].element_params[3];
                    double base_type_number = children_IVs[i].element_params[4];
                    calc_intrinsic_Si_I(Vs,vs_len,ni,VT,base_doping,base_type_number,base_thickness,1,Is,true,delta_n, BGN, pn);
                }
                else
                    interp_monotonic_inc(children_IVs[i].V, children_IVs[i].I, len, 
                        Vs, vs_len, Is, true, interp_method_, children_IVs[i].extrapolation_dI_dV[0], children_IVs[i].extrapolation_dI_dV[1]); // keeps adding 
            }
        }  
    }
    
    if (area != 1) {
        for (int i=0; i<vs_len; i++) Is[i] *= area;
        out_extrapolation_dI_dV[0] *= area;
        out_extrapolation_dI_dV[1] *= area;
    }
    *out_len = vs_len;
    
    return;
} 

void remesh_IV(
    double op_pt_V,
    double bottom_up_op_pt_V,
    double normalized_op_pt_V,
    int refine_mode,
    int max_num_points,
    int refinement_points,
    double* out_V,
    double* out_I,
    int* out_len, 
    double refine_V_half_width, 
    double max_tolerable_radians_change,
    double* accum_abs_dir_change, 
    double* accum_abs_dir_change_near_mpp, 
    int* idx) {

    double* Vs = out_V;
    double* Is = out_I;
    int *vs_len = out_len;

    if (*vs_len <= max_num_points)
        return;

    double op_left_V, op_right_V;
    if (refine_mode==1) {  
        op_left_V = op_pt_V;
        op_right_V = bottom_up_op_pt_V;
        if (op_left_V > op_right_V) {
            op_left_V = bottom_up_op_pt_V;
            op_right_V = op_pt_V;
        }
        op_left_V -= normalized_op_pt_V*refine_V_half_width;
        op_right_V += normalized_op_pt_V*refine_V_half_width;
    }

    int n = static_cast<int>(*vs_len);
    accum_abs_dir_change[0] = 0.0;
    if (refine_mode)
        accum_abs_dir_change_near_mpp[0] = 0.0;
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
    last_unit_vector_x = 2;
    last_unit_vector_y = 2;
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
        bool bad_point = false;
        if (!std::isfinite(unit_vector_x) || !std::isfinite(unit_vector_y) || mag<1e-8)  // catches NaN, +inf, -inf
            bad_point = true;
        if (i > 0) {
            double change = 0;
            if (!bad_point && std::abs(last_unit_vector_x)<=1.0) {
                double dx = unit_vector_x - last_unit_vector_x;
                double dy = unit_vector_y - last_unit_vector_y;
                change = std::sqrt(dx*dx + dy*dy);
            }
            accum_abs_dir_change[i] = accum_abs_dir_change[i-1] + change;
            if (refine_mode==1) {
                double change_ = 0;
                if (Vs[i]>=op_left_V && Vs[i]<=op_right_V) change_ = change;
                accum_abs_dir_change_near_mpp[i] = accum_abs_dir_change_near_mpp[i-1] + change_;
            }
        }
        if (!bad_point) {
            last_unit_vector_x = unit_vector_x;
            last_unit_vector_y = unit_vector_y;
        }
    }
    if (max_tolerable_radians_change > 0) {
        double at_least_max_num_points = accum_abs_dir_change[n-2]/max_tolerable_radians_change+2;
        if (*vs_len <= at_least_max_num_points)
            return;
        if (max_num_points < at_least_max_num_points)
            max_num_points = at_least_max_num_points;
    }

    double variation_segment = accum_abs_dir_change[n-2]/(max_num_points-2);
    if (!std::isfinite(variation_segment) || variation_segment <= 0.0) 
        return;  // keep original mesh; no remesh
    double variation_segment_mpp = 0.0;
    if (refine_mode == 1) {
        variation_segment_mpp =
            accum_abs_dir_change_near_mpp[n - 2] / refinement_points;
        if (!std::isfinite(variation_segment_mpp) || variation_segment_mpp <= 0.0) 
            return;  // keep original mesh; no remesh
    }
    idx[0] = 0;
    int idx_size = 1;
    int count = 1;
    int countmpp = 1;
    for (int i = 1; i < n - 1; ++i) {
        if (accum_abs_dir_change[i] >= count * variation_segment) {
            int last_index = idx_size-1;
            if (last_index >= 0 && i > idx_V_closest_to_SC_left && idx_V_closest_to_SC_left > idx[last_index]) {
                idx[idx_size] = (idx_V_closest_to_SC_left);  // just also capture points closest to SC to keep Isc accurate
                ++idx_size;
            }
            if (last_index >= 0 && i > idx_V_closest_to_SC && idx_V_closest_to_SC > idx[last_index]) {
                idx[idx_size] = (idx_V_closest_to_SC);  // just also capture points closest to SC to keep Isc accurate
                ++idx_size;
            }
            if (last_index >= 0 && i > idx_V_closest_to_SC_right && idx_V_closest_to_SC_right > idx[last_index]) {
                idx[idx_size] = (idx_V_closest_to_SC_right);  // just also capture points closest to SC to keep Isc accurate
                ++idx_size;
            }
            idx[idx_size] = i;
            ++idx_size;
            while (accum_abs_dir_change[i] >= count * variation_segment)
                ++count;
        } else if (refine_mode==1 && variation_segment_mpp>0 && accum_abs_dir_change_near_mpp[i] >= countmpp * variation_segment_mpp) {
            idx[idx_size] = i;
            ++idx_size;
            while (accum_abs_dir_change_near_mpp[i] >= countmpp * variation_segment_mpp)
                ++countmpp;
        }
    }
    idx[idx_size] = n-1;
    ++idx_size;

    // remesh
    int n_out = idx_size;;
    for (int i = 0; i < n_out; ++i) {
        int k = idx[i];  
        out_V[i] = Vs[k];
        out_I[i] = Is[k];
    }
    *out_len = n_out;
} 

bool combine_iv_jobs_batch(int n_jobs, IVJobDesc* jobs, 
    int parallel, int refine_mode, int interp_method, int use_existing_grid, 
    double refine_V_half_width, double max_tolerable_radians_change, 
    int has_any_intrinsic_diode, int has_any_photon_coupling, int largest_abs_max_num_points) {

    int max_threads = 1;
    int num_threads = 1;
    if (parallel==1 && n_jobs>1) {
        max_threads = omp_get_max_threads();
        num_threads = max_threads;
        if (n_jobs < max_threads*2) num_threads = max_threads/2;
        if (n_jobs < max_threads) num_threads = max_threads/4;
        if (n_jobs < max_threads/4) num_threads = n_jobs;
    }

    std::vector<double> scratch_delta_n;
    std::vector<double> scratch_BGN;
    std::vector<double> scratch_pn;
    std::vector<double> scratch_extra_Is;
    std::vector<double> scratch_this_V;
    std::vector<double> scratch_added_I;
    std::vector<double> scratch_xq;
    std::vector<double> scratch_new_points;
    std::vector<double> scratch_accum_change;
    std::vector<double> scratch_accum_change_mpp;
    std::vector<int>    scratch_idx;

    int scratch_size = num_threads * largest_abs_max_num_points;
    if (has_any_intrinsic_diode) {
        scratch_delta_n.resize(scratch_size);
        scratch_BGN.resize(scratch_size);
        scratch_pn.resize(scratch_size);
    }
    if (has_any_photon_coupling) {
        scratch_extra_Is.resize(scratch_size);
        scratch_this_V.resize(scratch_size);
        scratch_added_I.resize(scratch_size);
        scratch_xq.resize(scratch_size);
        scratch_new_points.resize(scratch_size);
    }
    
    #pragma omp parallel for num_threads(num_threads) if(parallel==1 && n_jobs>1)
    for (int j = 0; j < n_jobs; ++j) {
        int tid = omp_get_thread_num();

        size_t baseP = (size_t)tid * largest_abs_max_num_points;

        double* delta_n = nullptr;
        double* BGN     = nullptr;
        double* pn      = nullptr;
        double* extra_Is   = nullptr;
        double* this_V     = nullptr;
        double* added_I    = nullptr;
        double* xq         = nullptr;
        double* new_points = nullptr;
        if (has_any_intrinsic_diode) {
            delta_n    = scratch_delta_n.data()    + baseP;
            BGN        = scratch_BGN.data()        + baseP;
            pn         = scratch_pn.data()         + baseP;
        }
        if (has_any_photon_coupling) {
            extra_Is   = scratch_extra_Is.data()   + baseP;
            this_V     = scratch_this_V.data()     + baseP;
            added_I    = scratch_added_I.data()    + baseP;
            xq         = scratch_xq.data()         + baseP;
            new_points = scratch_new_points.data() + baseP;
        }

        IVJobDesc& job = jobs[j];
        combine_iv_job(
            job.connection,
            job.circuit_component_type_number,
            job.n_children,
            job.this_IV,
            job.children_IVs,
            job.children_pc_IVs,
            job.has_photon_coupling,
            job.max_num_points,
            job.area,
            job.abs_max_num_points,
            job.circuit_element_parameters,
            job.out_V,
            job.out_I,
            job.out_len,
            job.out_extrapolation_allowed,
            job.out_extrapolation_dI_dV,
            job.out_has_I_domain_limit,
            refine_mode,
            job.all_children_are_elements,
            job.operating_point[0],
            job.operating_point[1],
            job.operating_point[2],
            job.refinement_points,
            interp_method,
            use_existing_grid,
            refine_V_half_width, 
            delta_n, BGN, pn, extra_Is, this_V, added_I, xq, new_points
        );
    }

    for (int j = 0; j < n_jobs; ++j) 
        if (*jobs[j].out_len==0)
            return false; // some mesh have no overlaps, fail! 

    if (use_existing_grid==1)
        return true; // success

    std::vector<int> remesh_indices;
    remesh_indices.reserve(n_jobs);
    for (int j = 0; j < n_jobs; ++j) {
        if (jobs[j].max_num_points > 2 && jobs[j].circuit_component_type_number>=5) 
            remesh_indices.push_back(j);
    }
    int num_jobs_need_remesh = remesh_indices.size();
    if (num_jobs_need_remesh > 0) {
        int max_threads = 1;
        int num_threads = 1;
        if (parallel==1 && num_jobs_need_remesh>1) {
            max_threads = omp_get_max_threads();
            num_threads = max_threads;
            if (num_jobs_need_remesh < max_threads*2) num_threads = max_threads/2;
            if (num_jobs_need_remesh < max_threads) num_threads = max_threads/4;
            if (num_jobs_need_remesh < max_threads/4) num_threads = num_jobs_need_remesh;
        }

        int scratch_size = num_threads * largest_abs_max_num_points;
        scratch_accum_change.resize(scratch_size);
        if (refine_mode)
            scratch_accum_change_mpp.resize(scratch_size);
        scratch_idx.resize(scratch_size);

        #pragma omp parallel for num_threads(num_threads) if(parallel==1 && num_jobs_need_remesh>1)
        for (int j = 0; j < num_jobs_need_remesh; ++j) {
            int tid = omp_get_thread_num();

            size_t baseP = (size_t)tid * largest_abs_max_num_points;

            double* accum_abs_dir_change = nullptr;
            double* accum_abs_dir_change_near_mpp     = nullptr;
            int* idx      = nullptr;
            accum_abs_dir_change = scratch_accum_change.data() + baseP;
            if (refine_mode)
                accum_abs_dir_change_near_mpp = scratch_accum_change_mpp.data() + baseP;
            idx = scratch_idx.data() + baseP;
            
            IVJobDesc& job = jobs[remesh_indices[j]];
            remesh_IV(
                job.operating_point[0],
                job.operating_point[1],
                job.operating_point[2],
                refine_mode,
                job.max_num_points,
                job.refinement_points,
                job.out_V,
                job.out_I,
                job.out_len,
                refine_V_half_width,
                max_tolerable_radians_change,
                accum_abs_dir_change, accum_abs_dir_change_near_mpp, idx
            );
        }
    }
    return true; // success!
}

}// extern "C"
