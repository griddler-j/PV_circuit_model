// ivkernel.cpp
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

extern "C" {


void build_current_source_iv(
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len
) {
    double IL = circuit_element_parameters[0];
    int n = 2;
    for (int i=0; i < n; ++i) {
        out_V[i] = -0.1 + i*0.2;
        out_I[i] = -IL;
    }
    *out_len = n;
}

void build_resistor_iv(
    const double* circuit_element_parameters,
    double* out_V,
    double* out_I,
    int* out_len
) {
    double cond = circuit_element_parameters[0];
    int n = 5;
    for (int i=0; i < n; ++i) {
        out_V[i] = -0.1 + i*0.05;
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
        // Degenerate: constant function
        for (int j = 0; j < m; ++j) {
            yq[j] = y[0];
        }
        return;
    }

    int j = 0;

    // --- Left extrapolation: xq <= x[0] ---
    while (j < m && xq[j] < x[0]) {
        double t = (xq[j] - x[0]) / (x[1] - x[0]);
        yq[j] = (additive? yq[j]:0.0) + y[0] + t * (y[1] - y[0]);
        ++j;
    }

    // --- Main interpolation region: x[0] < xq < x[n-1] ---
    int i = 0; // segment index for x
    for (; j < m && xq[j] < x[n-1]; ++j) {
        double xj = xq[j];

        // Advance i until x[i] <= xj <= x[i+1]
        while (i + 1 < n - 1 && x[i+1] < xj) {
            ++i;
        }

        double x0 = x[i];
        double x1 = x[i+1];
        double y0 = y[i];
        double y1 = y[i+1];

        double t = (xj - x0) / (x1 - x0);
        yq[j] = (additive? yq[j]:0.0) + y0 + t * (y1 - y0);
    }

    // --- Right extrapolation: xq >= x[n-1] ---
    for (; j < m; ++j) {
        double t = (xq[j] - x[n-2]) / (x[n-1] - x[n-2]);
        yq[j] = (additive? yq[j]:0.0) + y[n-2] + t * (y[n-1] - y[n-2]);
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

std::vector<double> get_V_range(const double* circuit_element_parameters,int max_num_points,bool intrinsic_Si_calc) {
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
    V.reserve(N + 5);

    // First 5 fixed points
    V.push_back(V_shift - 1.1);
    V.push_back(V_shift - 1.0);
    V.push_back(V_shift);
    V.push_back(V_shift + 0.02);
    V.push_back(V_shift + 0.08);

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
    int* out_len
) {
    // circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I
    double I0 = circuit_element_parameters[0];
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = circuit_element_parameters[3];

    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,false);

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
    int* out_len
) {
    // circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I
    double I0 = circuit_element_parameters[0];
    double n = circuit_element_parameters[1];
    double VT = circuit_element_parameters[2];
    double V_shift = circuit_element_parameters[3];

    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,false);

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
    int* out_len
) {
    std::vector<double> V = get_V_range(circuit_element_parameters, max_num_points,true);
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


/**
 * Helper to thin a sorted grid using tolerance-based quantization
 */
static void thin_grid_by_quantization(std::vector<double>& xs, double tol)
{
    std::vector<long long> quant(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        quant[i] = static_cast<long long>(std::llround(xs[i] / tol));
    }
    std::vector<double> out;
    out.reserve(xs.size());
    long long last = std::numeric_limits<long long>::min();
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i == 0 || quant[i] != last) {
            out.push_back(xs[i]);
            last = quant[i];
        }
    }
    xs.swap(out);
}

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
    int* out_len) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // std::printf("cpp: combine_iv_job: n_children = %d, connection = %d\n",
    //         n_children, connection);
    // std::fflush(stdout); 

    std::vector<double> Vs(children_Vs_size);
    std::vector<double> Is(children_Vs_size);

    double cap_current_by_area = cap_current / area;

    if (connection == -1 && circuit_component_type_number <=4) { // CircuitElement; the two conditions are actually redundant as connection == -1 iff circuit_component_type_number <=4
        switch (circuit_component_type_number) {
            case 0: // CurrentSource
                build_current_source_iv(circuit_element_parameters, out_V, out_I, out_len);
                break;
            case 1: // Resistor
                build_resistor_iv(circuit_element_parameters, out_V, out_I, out_len);
                break;
            case 2: // ForwardDiode
                build_forward_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len);
                break;
            case 3: // ReverseDiode
                build_reverse_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len);
                break;
            case 4: // Intrinsic Si Diode
                build_Si_intrinsic_diode_iv(circuit_element_parameters, max_num_points, out_V, out_I, out_len);
                break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ms;
    }
    // --- Series connection branch (connection == 0) ---
    else if (connection == 0) {
        // add voltage
        Is.assign(children_Is, children_Is + children_Vs_size); 
        std::vector<double> extra_Is;
        for (int iteration = 0; iteration < 2; ++iteration) {
            if (iteration == 1 && !extra_Is.empty()) Is.insert(Is.end(), extra_Is.begin(), extra_Is.end());       
            std::sort(Is.begin(), Is.end());
            if (max_num_points == -1) {
                auto new_end = std::unique(Is.begin(), Is.end());
                Is.erase(new_end, Is.end());
            } else {
                double tol = (Is.back()-Is.front())/(max_num_points*1000);
                thin_grid_by_quantization(Is, tol);
            }     
            Vs.assign(Is.size(), 0.0);     
            std::vector<double> this_V(Is.size());
            // do reverse order to allow for photon coupling
            for (int i = n_children-1; i >= 0; --i) {
                int offset = children_offsets[i];
                int len = children_lengths[i];
                if (len > 0) { 
                    const double* IV_table_V = children_Vs + offset;  // pointer to first V
                    const double* IV_table_I = children_Is + offset;  // pointer to first I
                    if (i<n_children-1 && children_pc_lengths[i+1]>0 && children_lengths[i+1]>0) {  // need to add the current transferred by the subcell above via pc 
                        int pc_offset = children_pc_offsets[i+1];
                        int pc_len = children_pc_lengths[i+1];
                        const double* pc_IV_table_V = children_pc_Vs + pc_offset;  // pointer to first V
                        const double* pc_IV_table_I = children_pc_Is + pc_offset;  // pointer to first I
                        std::vector<double> added_I(Is.size());
                        // the first time this is reached, i<n_children-1 (at least second iteration through the loop)
                        // children_lengths[i+1]>0 which means in the previous iteration, this_V.data() would have been filled already!
                        interp_monotonic_inc(pc_IV_table_V, pc_IV_table_I, pc_len, this_V.data(), (int)this_V.size(), added_I.data(), false); 
                        for (int j=0; j < added_I.size(); j++) added_I[j] *= -1;
                        std::vector<double> xq(Is.size());
                        std::transform(Is.begin(), Is.end(),added_I.begin(),xq.begin(),std::minus<double>());
                        interp_monotonic_inc(IV_table_I, IV_table_V, len, xq.data(), (int)xq.size(), this_V.data(),false); 
                        std::vector<double> new_points(Is.size());
                        std::transform(Is.begin(), Is.end(),added_I.begin(),new_points.begin(),std::plus<double>());
                        extra_Is.insert(extra_Is.end(), new_points.begin(), new_points.end());
                    } else {
                        interp_monotonic_inc(IV_table_I, IV_table_V, len, Is.data(), (int)Is.size(), this_V.data(), false); 
                    }
                    std::transform(Vs.begin(), Vs.end(),this_V.begin(),Vs.begin(),std::plus<double>());
                }
            }
            if (extra_Is.empty()) break;
        }
    }
    // --- parallel connection branch (connection == 1) ---
    else if (connection == 1) {
        // add current
        std::memcpy(Vs.data(), children_Vs, children_Vs_size * sizeof(double));
        std::sort(Vs.begin(), Vs.end());
        double left_limit = -100;
        double right_limit = 100;
        for (int i=0; i < n_children; ++i) {
            int offset = children_offsets[i];
            int len = children_lengths[i];
            const double* IV_table_V = children_Vs + offset;  // pointer to first V
            if (children_type_numbers[i]==2) { //  forward diode
                right_limit = std::min(right_limit, IV_table_V[len-1]);
            } else if (children_type_numbers[i]==3) {  // rev diode
                left_limit = std::max(left_limit, IV_table_V[0]);
            }
        }
        Vs.erase(std::remove_if(Vs.begin(),Vs.end(),[left_limit, right_limit](double v) {return v < left_limit || v > right_limit;}),Vs.end());
        if (max_num_points == -1) {
            auto new_end = std::unique(Vs.begin(), Vs.end());
            Vs.erase(new_end, Vs.end());
        } else {
            double tol = (Vs.back()-Vs.front())/(max_num_points*1000);
            thin_grid_by_quantization(Vs, tol);
        }
        Is.assign(Vs.size(), 0.0);
        for (int i = 0; i < n_children; ++i) {  
            int offset = children_offsets[i];
            int len = children_lengths[i];
            if (len > 0) {
                const double* IV_table_V = children_Vs + offset;  // pointer to first V
                const double* IV_table_I = children_Is + offset;  // pointer to first I
                interp_monotonic_inc(IV_table_V, IV_table_I, len, Vs.data(), (int)Vs.size(), Is.data(), true); // keeps adding 
            }
        }    
        for (int k = 0; k < (int)Is.size(); ++k) Is[k] = Is[k] + total_IL;
        if (cap_current_by_area > 0) {
            size_t write = 0;
            for (size_t read = 0; read < Is.size(); ++read) {
                if (std::abs(Is[read]) < cap_current_by_area) {
                    Vs[write] = Vs[read];
                    Is[write] = Is[read];
                    ++write;
                }
            }
            Vs.resize(write);
            Is.resize(write);
        }
    }

    // remesh
    if (max_num_points > 0) {
        double V_range = Vs.back() - Vs.front();
        double I_range = Is.back() - Is.front();
        std::vector<double> segment_lengths(Vs.size()-1);
        double total_length = 0.0;
        for (int i=0; i<(int)segment_lengths.size(); ++i) {
            double dv = (Vs[i+1]-Vs[i])/V_range;
            double di = (Is[i+1]-Is[i])/I_range;
            segment_lengths[i] = std::sqrt(dv * dv + di * di);
            total_length = total_length + segment_lengths[i];
        }
        double ideal_segment_length = total_length / max_num_points;
        std::vector<int> short_segments;
        std::vector<int> long_segments;
        std::vector<double> short_segment_lengths;
        std::vector<double> short_segment_lengths_cum;
        short_segments.reserve((int)segment_lengths.size());
        long_segments.reserve((int)segment_lengths.size());
        short_segment_lengths.reserve((int)segment_lengths.size());
        short_segment_lengths_cum.reserve((int)segment_lengths.size());
        for (int i = 0; i < (int)segment_lengths.size(); ++i) {
            if (segment_lengths[i] < ideal_segment_length) {
                short_segments.push_back(i);
                short_segment_lengths.push_back(segment_lengths[i]);
                short_segment_lengths_cum.push_back(segment_lengths[i]);
                int s = short_segment_lengths_cum.size();
                if (s > 1) short_segment_lengths_cum[s-1] = short_segment_lengths_cum[s-1] + short_segment_lengths_cum[s-2];
            } else {
                long_segments.push_back(i);
            }
        }
        if (short_segments.size()>0) {
            // how many “short” samples we want
            int n_target = max_num_points - (int)long_segments.size();
            if (n_target < 1) n_target = 1;
            // ideal_Vs: linspace(0, short_segment_lengths_cum.back(), n_target)
            std::vector<double> ideal_Vs(n_target);
            double L_total = short_segment_lengths_cum.back();
            if (n_target == 1) {
                ideal_Vs[0] = L_total;  // or 0.0, depending how close you want to match np.linspace
            } else {
                for (int k = 0; k < n_target; ++k) {
                    ideal_Vs[k] = L_total * (double)k / (double)(n_target - 1);
                }
            }
            // index = searchsorted(short_segment_lengths_cum, ideal_Vs, side='right') - 1
            std::vector<int> index(n_target);
            for (int k = 0; k < n_target; ++k) {
                double v = ideal_Vs[k];
                auto it = std::upper_bound(short_segment_lengths_cum.begin(),short_segment_lengths_cum.end(),v);  // first element > v
                int idx = static_cast<int>(it - short_segment_lengths_cum.begin()) - 1;
                if (idx < 0) idx = 0;  // clamp like numpy’s behavior for very small v
                index[k] = idx;
            }
            
            // ---- Build left-endpoint arrays for short and long segments ----
            std::vector<double> short_segment_left_V(short_segments.size());
            for (size_t j = 0; j < short_segments.size(); ++j) {
                short_segment_left_V[j] = Vs[ short_segments[j] ];
            }

            std::vector<double> long_segment_left_V(long_segments.size());
            for (size_t j = 0; j < long_segments.size(); ++j) {
                long_segment_left_V[j] = Vs[ long_segments[j] ];
            }

            // ---- new_Vs = short_segment_left_V[index] + ideal_Vs - short_segment_lengths_cum[index] ----
            std::vector<double> new_Vs;
            new_Vs.reserve(index.size() + long_segment_left_V.size() + 2); // +2 for endpoints

            for (size_t k = 0; k < index.size(); ++k) {
                int idx = index[k];  // index into short_segments / short_segment_left_V / short_segment_lengths_cum

                double left_V = short_segment_left_V[idx];
                double cum_L  = short_segment_lengths_cum[idx];
                double v      = left_V + ideal_Vs[k] - cum_L;

                new_Vs.push_back(v);
            }

            // ---- new_Vs.extend(long_segment_left_V) ----
            new_Vs.insert(new_Vs.end(), long_segment_left_V.begin(), long_segment_left_V.end());

            // ---- new_Vs = np.sort(new_Vs) ----
            std::sort(new_Vs.begin(), new_Vs.end());

            // ---- If needed, ensure endpoints cover [Vs.front(), Vs.back()] ----
            if (!new_Vs.empty() && new_Vs.front() > Vs.front()) {
                new_Vs.insert(new_Vs.begin(), Vs.front());
            }
            if (!new_Vs.empty() && new_Vs.back() < Vs.back()) {
                new_Vs.push_back(Vs.back());
            }

            // ---- new_Is = interp_(new_Vs, Vs, Is) ----
            // Assuming you have a monotonic interp like:
            //   interp_monotonic_inc(x, y, n, xq, m, yq, additive)
            std::vector<double> new_Is(new_Vs.size(), 0.0);

            interp_monotonic_inc(
                Vs.data(),          // x  (original V grid)
                Is.data(),          // y  (original I(V))
                (int)Vs.size(),
                new_Vs.data(),      // xq (new grid)
                (int)new_Vs.size(),
                new_Is.data(),      // yq
                false               // additive = false → overwrite, not accumulate
            );

            int n_out = (int)new_Vs.size();
            if (abs_max_num_points > 0 && n_out > abs_max_num_points) {
                n_out = abs_max_num_points;
            }
            std::memcpy(out_V, new_Vs.data(), n_out * sizeof(double));
            std::memcpy(out_I, new_Is.data(), n_out * sizeof(double));
            *out_len = n_out;
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            return ms;

        }
    } 

    if (area != 1) {
        for (int i=0; i<Is.size(); i++) Is[i] *= area;
    }

    int n_out = (int)Vs.size();

    if (n_out > abs_max_num_points) {
        std::printf("cpp: combine_iv_job error: n_out = %d, abs_max_num_points = %d\n",
            n_out, abs_max_num_points);
        std::fflush(stdout); 
    }
    if (abs_max_num_points > 0 && n_out > abs_max_num_points) {
        n_out = abs_max_num_points;
    }
    std::memcpy(out_V, Vs.data(), n_out * sizeof(double));
    std::memcpy(out_I, Is.data(), n_out * sizeof(double));
    *out_len = n_out;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms;

} 

}// extern "C"
