import numpy as np
from PV_Circuit_Model.utilities import *
import tqdm
import time

PACKAGE_ROOT = Path(__file__).resolve().parent
PARAM_DIR = PACKAGE_ROOT / "parameters"

REFINE_V_HALF_WIDTH = 0.005
MAX_TOLERABLE_RADIANS_CHANGE = 0.008726638 # half a degree
REMESH_POINTS_DENSITY = 500
REFINEMENT_POINTS_DENSITY = 125
REMESH_NUM_ELEMENTS_THRESHOLD = 50

try:
    ParameterSet(name="solver_env_variables",filename=PARAM_DIR / "solver_env_variables.json")
    solver_env_variables = ParameterSet.get_set("solver_env_variables")
    REFINE_V_HALF_WIDTH = solver_env_variables["REFINE_V_HALF_WIDTH"]
    MAX_TOLERABLE_RADIANS_CHANGE = solver_env_variables["MAX_TOLERABLE_RADIANS_CHANGE"]
    REMESH_POINTS_DENSITY = solver_env_variables["REMESH_POINTS_DENSITY"]
    REFINEMENT_POINTS_DENSITY = solver_env_variables["REFINEMENT_POINTS_DENSITY"]
    REMESH_NUM_ELEMENTS_THRESHOLD = solver_env_variables["REMESH_NUM_ELEMENTS_THRESHOLD"]
except Exception:
    ParameterSet(name="solver_env_variables",data={})
    solver_env_variables = ParameterSet.get_set("solver_env_variables")
    solver_env_variables.set("REFINE_V_HALF_WIDTH", REFINE_V_HALF_WIDTH)
    solver_env_variables.set("MAX_TOLERABLE_RADIANS_CHANGE", MAX_TOLERABLE_RADIANS_CHANGE)
    solver_env_variables.set("REMESH_POINTS_DENSITY", REMESH_POINTS_DENSITY)
    solver_env_variables.set("REFINEMENT_POINTS_DENSITY", REFINEMENT_POINTS_DENSITY)
    solver_env_variables.set("REMESH_NUM_ELEMENTS_THRESHOLD", REMESH_NUM_ELEMENTS_THRESHOLD)
solver_env_variables.set("_REPORT_UNCERTAINTY", False)
solver_env_variables.set("_PARALLEL_MODE", False)
solver_env_variables.set("_USE_CYTHON", False)

def get_V_range(component):
    VT = component.VT
    max_num_points = getattr(component,"max_num_points",None)
    if not max_num_points:
        max_num_points = 100
    max_I = getattr(component,"max_I",None)
    if not max_I:
        max_I = 0.2
    max_num_points_ = max_num_points*max_I/0.2
    Voc = 10
    if component._type_number == 4: # Intrinsic Si Diode
        if component.base_thickness>0:
            Voc = 0.7
            for _ in range(10):
                I = calc_intrinsic_Si_I(component,Voc)
                if I >= max_I and I <= max_I*1.1:
                    break
                Voc += VT*np.log(max_I/I)
    else:
        if component.I0>0:
            Voc = component.n*VT*np.log(max_I/component.I0) 

    V = [-1.1,-1.0,0]+list(Voc*np.log(np.arange(1,max_num_points_))/np.log(max_num_points_-1))
    V = np.array(V) + component.V_shift
    return V

def calc_intrinsic_Si_I(component, V):
    ni = component.ni
    VT = component.VT
    N_doping = component.base_doping
    pn = ni**2*np.exp(V/VT)
    delta_n = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2*np.exp(V/VT)))
    if component.base_type == "p":
        n0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        p0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
    else:
        p0 = 0.5*(-N_doping + np.sqrt(N_doping**2 + 4*ni**2))
        n0 = 0.5*(N_doping + np.sqrt(N_doping**2 + 4*ni**2))
    BGN = interp_(delta_n,component.bandgap_narrowing_RT[:,0],component.bandgap_narrowing_RT[:,1])
    ni_eff = ni*np.exp(BGN/2/VT)

    q = 1.602e-19
    geeh = 1 + 13*(1-np.tanh((n0/3.3e17)**0.66))
    gehh = 1 + 7.5*(1-np.tanh((p0/7e17)**0.63))
    Brel = 1
    Blow = 4.73e-15
    intrinsic_recomb = (pn - ni_eff**2)*(2.5e-31*geeh*n0+8.5e-32*gehh*p0+3e-29*delta_n**0.92+Brel*Blow) # in units of 1/s/cm3
    return q*intrinsic_recomb*component.base_thickness*component.area


def build_component_IV_python(component,refine_mode=False):
    if refine_mode:
        refinement_points = int(REFINEMENT_POINTS_DENSITY*np.sqrt(component.num_circuit_elements))
    circuit_component_type_number = component._type_number
    if circuit_component_type_number <=4: # CircuitElement
        if circuit_component_type_number == 0: # CurrentSource
            IL = component.IL
            component.IV_V = np.array([0])
            component.IV_I = np.array([-IL])
        elif circuit_component_type_number == 1: # Resistor
            cond = component.cond
            component.IV_V = np.array([-0.1,0.1])
            component.IV_I = component.IV_V*cond
        else:
            I0 = component.I0
            n = component.n
            VT = component.VT
            V_shift = component.V_shift
            component.IV_V = get_V_range(component)
            if circuit_component_type_number == 2: # ForwardDiode
                component.IV_I = I0*(np.exp((component.IV_V-V_shift)/(n*VT))-1)
            elif circuit_component_type_number == 3: # ReverseDiode
                component.IV_I = I0*(np.exp((component.IV_V-V_shift)/(n*VT)))
                component.IV_V *= -1
                component.IV_I *= -1
                component.IV_V = component.IV_V[::-1].copy()
                component.IV_I = component.IV_I[::-1].copy()
            else:
                component.IV_I = calc_intrinsic_Si_I(component, component.IV_V)
        return

    if refine_mode:
        bottom_up_operating_point_V = 0
        bottom_up_operating_point_I = 0
        normalized_operating_point_V = 0
        normalized_operating_point_I = 0
        all_children_are_elements = True
        for element in component.subgroups:
            if element._type_number >= 5: # circuitgroup
                all_children_are_elements = False
                bottom_up_operating_point_V += element.bottom_up_operating_point[0]
                bottom_up_operating_point_I += element.bottom_up_operating_point[1]
                normalized_operating_point_V += element.normalized_operating_point[0]
                normalized_operating_point_I += element.normalized_operating_point[1]
            else:
                bottom_up_operating_point_V += element.operating_point[0]
                bottom_up_operating_point_I += element.operating_point[1]
                normalized_operating_point_V += 1
                normalized_operating_point_I += 1

        if component.connection == "series":
            bottom_up_operating_point_I /= len(component.subgroups)
            normalized_operating_point_I /=len(component.subgroups)
        else:
            bottom_up_operating_point_V /= len(component.subgroups)
            normalized_operating_point_V /= len(component.subgroups)

        component.bottom_up_operating_point = [bottom_up_operating_point_V,bottom_up_operating_point_I]
        component.normalized_operating_point = [normalized_operating_point_V,normalized_operating_point_I]

        left_V_refine = min(component.operating_point[0],component.bottom_up_operating_point[0])
        right_V_refine = max(component.operating_point[0],component.bottom_up_operating_point[0])
        normalized_op_pt_V = component.normalized_operating_point[0]
        left_V_refine -= normalized_op_pt_V*REFINE_V_HALF_WIDTH
        right_V_refine += normalized_op_pt_V*REFINE_V_HALF_WIDTH

    if component.connection=="series":
        I_range = 0
        for element in component.subgroups:
            I_range = max(I_range, element.IV_I[-1]-element.IV_I[0])
        extra_Is = []
        for iteration in [0,1]: # goes to 1 only if there is PC diode
            # add voltage
            Is = []
            for element in component.subgroups:
                Is.extend(list(element.IV_I))
            if iteration==1:
                Is.extend(extra_Is)
            Is = np.sort(np.array(Is))
            eps = I_range/1e8

            I_diff = np.abs(Is[1:] - Is[:-1])
            indices = np.where(I_diff > eps)[0]
            indices = np.concatenate(([0], indices + 1))
            Is = Is[indices]
            # Is = np.unique(Is)
            Vs = np.zeros_like(Is)
            # do reverse order to allow for photon coupling
            pc_IVs = []
            for element in reversed(component.subgroups):
                if len(pc_IVs)>0:
                    prev_subcell_V = interp_(Is,prev_IV[1,:],prev_IV[0,:])
                    added_I = np.zeros_like(prev_subcell_V)
                    for pc_IV in pc_IVs:
                        added_I -= interp_(prev_subcell_V, pc_IV[0,:], pc_IV[1,:])
                    V = interp_(Is-added_I,element.IV_I,element.IV_V)
                    extra_Is.extend(Is+added_I)
                else:
                    V = interp_(Is,element.IV_I,element.IV_V)
                Vs += V
                pc_IVs = []
                prev_IV = []
                if hasattr(element,"photon_coupling_diodes"):
                    prev_IV = element.IV_table
                    for pc in element.photon_coupling_diodes:
                        pc_IVs.append(pc.IV_table.copy())
                        pc_IVs[-1][1,:] *= element.area
            if len(extra_Is)==0:
                break
    else:
        # add current
        Vs = []
        left_limit = None
        right_limit = None
        for element in component.subgroups:
            Vs.extend(list(element.IV_table[0,:]))
            if element._type_number == 2 or element._type_number == 4: # ForwardDiode
                if right_limit is None:
                    right_limit = element.IV_V[-1]
                else:
                    right_limit = min(element.IV_V[-1],right_limit)
            elif element._type_number == 3: # ReverseDiode
                if left_limit is None:
                    left_limit = element.IV_V[0]
                else:
                    left_limit = max(element.IV_V[0],left_limit)
        Vs = np.sort(np.array(Vs))
        if left_limit is not None:
            find_ = np.where(Vs >= left_limit)[0]
            Vs = Vs[find_]
        if right_limit is not None:
            find_ = np.where(Vs <= right_limit)[0]
            Vs = Vs[find_]

        if refine_mode:
            if all_children_are_elements: # add more points near the operating point
                step = (right_V_refine - left_V_refine)/(refinement_points-1)
                added_Vs = np.arange(left_V_refine,right_V_refine,step)
                Vs = np.concatenate((Vs, added_Vs))
            
        eps = 1e-6
        V_diff = np.abs(Vs[1:] - Vs[:-1])
        indices = np.where(V_diff > eps)[0]
        indices = np.concatenate(([0], indices + 1))
        Vs = Vs[indices]

        Is = np.zeros_like(Vs)
        for element in component.subgroups:
            if element._type_number < 5: # CircuitElement, direct evaluate
                if element._type_number == 0: # CurrentSource
                    IL = element.IL
                    Is -= IL*np.ones_like(Vs) 
                elif element._type_number == 1: # Resistor
                    cond = element.cond
                    Is += cond*Vs
                else:
                    I0 = element.I0
                    n = element.n
                    VT = element.VT
                    V_shift = element.V_shift
                    if element._type_number == 2: # ForwardDiode
                        Is += I0*(np.exp((Vs-V_shift)/(n*VT))-1)
                    elif element._type_number == 3: # ReverseDiode
                        Is += -I0*np.exp((-Vs-V_shift)/(n*VT))
                    else:
                        Is += calc_intrinsic_Si_I(element, Vs)
            else:
                Is += interp_(Vs,element.IV_V,element.IV_I)

    component.IV_V = Vs
    component.IV_I = Is

    if component._type_number == 6: # cell
        component.IV_I *= component.area

    #remesh
    max_num_points = getattr(component,"max_num_points",None)
    if max_num_points is None or max_num_points <= 2 or max_num_points >= component.IV_V.size:
        return

    Vs = component.IV_V
    Vs, idx = np.unique(Vs, return_index=True)
    Is = component.IV_I[idx]
    V_range = Vs[-1]
    left_V = 0.05 * Vs[0]
    right_V = 0.05 * Vs[-1]
    I_range = min(abs(Is[-1]), abs(Is[0]))
    idx_V_closest_to_SC = np.argmin(np.abs(Vs))
    idx_V_closest_to_SC_right = np.argmin(np.abs(Vs - right_V))
    idx_V_closest_to_SC_left  = np.argmin(np.abs(Vs - left_V))

    dx = (Vs[1:]-Vs[:-1])/V_range
    dy = (Is[1:]-Is[:-1])/I_range
    mag = np.sqrt(dx**2+dy**2)
    unit_x = dx/mag
    unit_y = dy/mag
    findbad_ = np.where((mag<1e-8) | np.isnan(unit_x) | np.isnan(unit_y) | np.isinf(unit_x) | np.isinf(unit_y))[0]
    sqrt_half = np.sqrt(0.5)
    unit_x[findbad_] = sqrt_half
    unit_y[findbad_] = sqrt_half
    findbad_ = np.concatenate((findbad_,findbad_+1))
    findbad_ = np.unique(findbad_)
    within = np.where(findbad_<unit_x.size)[0]
    findbad_ = findbad_[within]
    dux = np.diff(unit_x)  # length n-2
    duy = np.diff(unit_y)
    dux[findbad_] = 0
    dux[findbad_] = 0
    change = np.zeros_like(unit_x)
    change[1:] = np.sqrt(dux * dux + duy * duy)  # change[i] for i>=1
    change[findbad_] = 0
    accum_abs_dir_change = np.cumsum(change)

    # ---- accum_abs_dir_change_near_mpp (also vectorized) ----
    if refine_mode == 1:
        window = (Vs[1:]>=left_V_refine) & (Vs[1:]<=right_V_refine)
        increments_mpp = change * window
        accum_abs_dir_change_near_mpp = np.cumsum(increments_mpp)
        variation_segment_mpp = accum_abs_dir_change_near_mpp[-1] / refinement_points
    else:
        accum_abs_dir_change_near_mpp = None
        variation_segment_mpp = None

    # ---- variation segment for global curve ----
    # C++ used accum_abs_dir_change[n-2] / (max_num_points-2)
    total_variation = accum_abs_dir_change[-1] if Vs.size > 1 else 0.0

    if (MAX_TOLERABLE_RADIANS_CHANGE > 0):
        at_least_max_num_points = accum_abs_dir_change[-1]/MAX_TOLERABLE_RADIANS_CHANGE+2
        if (Vs.size <= at_least_max_num_points):
            return
        if (max_num_points < at_least_max_num_points):
            max_num_points = at_least_max_num_points

    variation_segment = total_variation / (max_num_points - 2)
    ideal_points = variation_segment*np.arange(1,max_num_points-1)
    idx = list(np.searchsorted(accum_abs_dir_change, ideal_points, side='right'))
    idx[-1] = min(idx[-1],Vs.size-2)
    idx.append(0)
    idx.append(Vs.size-1)
    idx.append(idx_V_closest_to_SC)
    idx.append(idx_V_closest_to_SC_left)
    idx.append(idx_V_closest_to_SC_right)
    if variation_segment_mpp is not None:
        ideal_points = variation_segment_mpp*np.arange(refinement_points)
        idx2 = list(np.searchsorted(accum_abs_dir_change_near_mpp, ideal_points, side='right'))
        idx2[-1] = min(idx2[-1],Vs.size-2)
        idx.extend(idx2)
    idx = np.array(sorted(set(idx)), dtype=int)  # ensure unique + sorted (optional)

    component.IV_V = component.IV_V[idx]
    component.IV_I = component.IV_I[idx]


# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component):
        self.components = [circuit_component]
        self.children_job_ids = [[]]
        self.job_done_index = len(self.components)
        self.timers = {"build":0.0,"IV":0.0,"refine":0.0,"bounds":0.0}
        self.build()
    def build(self):
        start_time = time.time()
        pos = 0
        while pos < len(self.components):
            circuit_component = self.components[pos] 
            subgroups = getattr(circuit_component, "subgroups", None)
            if subgroups:
                for element in subgroups:
                    self.components.append(element)
                    self.children_job_ids.append([])
                    self.children_job_ids[pos].append(len(self.components)-1)
            pos += 1
        duration = time.time() - start_time
        self.timers["build"] = duration
    def get_runnable_iv_jobs(self,forward=True,refine_mode=False):
        include_indices = []
        start_job_index = self.job_done_index
        if forward:
            for i in reversed(range(start_job_index)):
                ids = self.children_job_ids[i]
                if len(ids)>0 and min(ids)<start_job_index:
                    break
                self.job_done_index = i
                # if refine_mode, then don't bother with CircuitElements, but run even if there is IV
                if (refine_mode and len(self.children_job_ids[i])>=0) or self.components[i].IV_V is None:
                    include_indices.append(i)
        else:
            min_id = len(self.components) + 100
            for i in range(start_job_index,len(self.components)):
                if i >= min_id:
                    break
                ids = self.children_job_ids[i]
                if len(ids)>0: min_id = min(min_id, min(ids))
                self.job_done_index = i+1
                include_indices.append(i)
        return [self.components[j] for j in include_indices]
    def reset(self,forward=True):
        if forward:
            self.job_done_index = len(self.components)
        else:
            self.job_done_index = 0
    def set_operating_point(self,V=None,I=None):
        start_time = time.time()
        self.reset(forward=False)
        pbar = None
        if V is not None:
            self.components[0].operating_point = [V,None]
        else:
            self.components[0].operating_point = [None,I]
        if len(self.components) > 100000:
            pbar = tqdm(total=len(self.components), desc="Processing the circuit hierarchy: ")
        while self.job_done_index < len(self.components):
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs(forward=False)
            if len(components_) > 0:
                for component in components_:
                    V = component.operating_point[0]
                    I = component.operating_point[1]
                    if V is not None:
                        if component._type_number < 5: # CircuitElement, direct evaluate
                            if component._type_number == 0: # CurrentSource
                                IL = component.IL
                                component.operating_point[1] = IL
                            elif component._type_number == 1: # Resistor
                                cond = component.cond
                                component.operating_point[1] = cond*V
                            else:
                                I0 = component.I0
                                n = component.n
                                VT = component.VT
                                V_shift = component.V_shift
                                if component._type_number == 2: # ForwardDiode
                                    component.operating_point[1] = I0*(np.exp((V-V_shift)/(n*VT))-1)
                                elif component._type_number == 3: # ReverseDiode
                                    component.operating_point[1] = -I0*np.exp((-V-V_shift)/(n*VT))
                                else:
                                    component.operating_point[1] = calc_intrinsic_Si_I(component, V)
                        else:
                            component.operating_point[1] = interp_(V,component.IV_V,component.IV_I)
                    elif I is not None:
                        component.operating_point[0] = interp_(I,component.IV_I,component.IV_V)
                    if component._type_number>=5:
                        is_series = False
                        if component.connection=="series":
                            is_series = True
                        current_ = component.operating_point[1]
                        if component._type_number==6: # cell
                            current_ /= component.area
                        for child in component.subgroups:
                            if is_series:
                                child.operating_point = [None, current_]
                            else:
                                child.operating_point = [component.operating_point[0], None]
            if pbar is not None:
                pbar.update(self.job_done_index-job_done_index_before)
        if pbar is not None:
            pbar.close()
        duration = time.time() - start_time
        self.timers["refine"] = duration
    def run_IV(self, refine_mode=False):
        start_time = time.time()
        self.reset()
        pbar = None
        if self.job_done_index > 100000:
            pbar = tqdm(total=self.job_done_index, desc="Processing the circuit hierarchy: ")
        while self.job_done_index > 0:
            job_done_index_before = self.job_done_index
            components_ = self.get_runnable_iv_jobs(refine_mode=refine_mode)
            if len(components_) > 0:
                for component in components_:
                    build_component_IV_python(component,refine_mode=refine_mode)
            if pbar is not None:
                pbar.update(job_done_index_before-self.job_done_index)
        if pbar is not None:
            pbar.close()

        duration = time.time() - start_time
        if refine_mode:
            self.timers["refine"] += duration  # added to the operating point time
        else:
            self.timers["IV"] = duration

    def refine_IV(self):
        if self.components[0].IV_V is not None and self.components[0].operating_point is not None:
            self.run_IV(refine_mode=True)

    def calc_Kirchoff_law_errors(self):
        worst_I_error = 0
        worst_V_error = 0
        if self.components[0].refined_IV:
            for component in self.components:
                if component._type_number>=5: #CircuitGroup
                    has_started = False
                    for i, element in enumerate(component.subgroups):
                        if element._type_number >= 5:
                            if not has_started:
                                largest_V = element.bottom_up_operating_point[0]
                                smallest_V = element.bottom_up_operating_point[0]
                                largest_I = element.bottom_up_operating_point[1]
                                smallest_I = element.bottom_up_operating_point[1]
                                has_started = True
                            else:
                                largest_V = max(largest_V,element.bottom_up_operating_point[0])
                                smallest_V = min(smallest_V,element.bottom_up_operating_point[0])
                                largest_I = max(largest_I,element.bottom_up_operating_point[1])
                                smallest_I = min(smallest_I,element.bottom_up_operating_point[1])
                    if component.connection=="series": # require same I
                        worst_I_error = max(worst_I_error, largest_I-smallest_I)
                    else: # require same V
                        worst_V_error = max(worst_V_error, largest_V-smallest_V)
        return worst_V_error, worst_I_error


