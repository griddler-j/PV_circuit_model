import numpy as np
import os
import time
from PV_Circuit_Model.iv_jobs_cython import build_children_buffers
import PV_Circuit_Model.ivkernel as ivkernel  

class KernelTimer:
    def __init__(self):
        self.kernel_timer = 0.0
        self.wrapper_timer = 0.0
        self.wrapper_timer_tic = 0.0
        self.wrapper_timer_toc = 0.0
        self.build_IV_events = []
        self.is_activated = False
    def reset(self):
        self.is_activated = True
        self.kernel_timer = 0.0
        self.wrapper_timer = 0.0
        self.build_IV_events = []
    def register(self,circuit_component):
        if self.is_activated and type(circuit_component).__name__=="Cell":
            assert(1==0)
        self.build_IV_events.append({'component':type(circuit_component).__name__, 'wrapper_timer': 0.0, 'kernel_timer': 0.0})
    def tic(self):
        self.wrapper_timer_tic = time.time()
    def toc(self):
        lapse = time.time()-self.wrapper_timer_tic
        self.build_IV_events[-1]["wrapper_timer"] += lapse
        self.wrapper_timer += lapse
        self.wrapper_timer_tic = 0.0
    def inc(self,time_):
        self.build_IV_events[-1]["kernel_timer"] += time_
        self.kernel_timer += time_
    def show_log(self):
        for i, event in enumerate(self.build_IV_events):
            print(f"{i}: {event["component"]}\t{event["kernel_timer"]}\t{event["wrapper_timer"]}")
    def __str__(self):
        return f"kernel_timer = {self.kernel_timer}, wrapper_timer = {self.wrapper_timer}"
    
kernel_timer = KernelTimer()

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component,max_num_points=None, cap_current=None):
        self.job_list = []
        this_job_id = self.add(circuit_component)
        self.max_num_points = max_num_points
        self.cap_current = cap_current
        self.build(circuit_component,this_job_id)
        self.job_done_index = len(self.job_list)
        kernel_timer.register(circuit_component)
    def add(self,circuit_component,parent_id=None):
        this_job_id = len(self.job_list)
        self.job_list.append({"circuit_component": circuit_component, "children_job_ids": [], "done": False})
        if parent_id is not None:
            self.job_list[parent_id]["children_job_ids"].append(this_job_id)
        return this_job_id
    def build(self,circuit_component,this_job_id=None):
        if hasattr(circuit_component,"subgroups"):
            new_job_ids = []
            for _, element in enumerate(circuit_component.subgroups):
                if element.IV_table is None:
                    new_job_id = self.add(element, parent_id=this_job_id)
                    new_job_ids.append(new_job_id)
                else:
                    new_job_ids.append(-1)
            for i, element in enumerate(circuit_component.subgroups):
                if element.IV_table is None:
                    self.build(element, this_job_id=new_job_ids[i])
    def prep_job(self,job):
        circuit_component = job["circuit_component"]
        type_name = type(circuit_component).__name__
        circuit_component_type_number = 5 # CircuitGroup
        if type_name in type_numbers: # is CircuitElement
            circuit_component_type_number = type_numbers[type_name]

        job["connection"] = None
        job["dark_IV_table"] = None
        job["area"] = 1
        if hasattr(circuit_component,"connection"):
            job["connection"] = circuit_component.connection
        if hasattr(circuit_component,"dark_IV_table") and circuit_component.dark_IV_table is not None:
            job["dark_IV_table"] = circuit_component.dark_IV_table
        if hasattr(circuit_component,"shape") and hasattr(circuit_component,"area") and circuit_component.area is not None:
            job["area"] = circuit_component.area
        job["total_IL"] = 0.0
        job["children_IVs"] = []
        job["children_pc_IVs"] = []
        job["children_types"] = []
        job["max_num_points"] = self.max_num_points
        job["cap_current"] = self.cap_current
        all_children_are_CircuitElement = False
        if hasattr(circuit_component,"subgroups"):
            all_children_are_CircuitElement = True
            for element in circuit_component.subgroups:
                type_name = type(element).__name__
                if type_name not in type_numbers: # not CircuitElement
                    all_children_are_CircuitElement = False

                job["children_types"].append(type(element))
                if circuit_component_type_number==0: # current source
                    element.set_IL(element.IL)
                    job["total_IL"] -= element.IL
                    job["children_IVs"].append(np.zeros((2, 0)))
                else:
                    job["children_IVs"].append(element.IV_table.copy())

                job["children_pc_IVs"].append(np.zeros((2, 0)))
                if hasattr(element,"photon_coupling_diodes") and len(element.photon_coupling_diodes)>0:
                    pc_IV = element.photon_coupling_diodes[0].IV_table.copy()
                    pc_IV[1,:] *= element.area
                    job["children_pc_IVs"][-1] = pc_IV
        job["all_children_are_CircuitElement"] = all_children_are_CircuitElement
    def get_runnable_jobs(self):
        for i in range(self.job_done_index-1,-1,-1):
            job = self.job_list[i]
            if len(job["children_job_ids"])>0 and min(job["children_job_ids"])<self.job_done_index:
                return self.job_list[i+1:self.job_done_index]
            self.prep_job(job)
        return self.job_list[:self.job_done_index]
    def run_jobs(self):
        t1 = time.time()
        total_ms_ = 0
        total_ms_2 = 0
        get_job_time = 0
        while self.job_done_index > 0:
            t1b = time.time()
            jobs = self.get_runnable_jobs()
            get_job_time += time.time()-t1b

            if len(jobs) >= 2:
                job_descs = []
                kernel_timer.tic()
                for i, job in enumerate(jobs):
                    job_descs.append(make_job_desc(job))
                job_descs_ = [jd for jd in job_descs if jd is not None]
                IVs, total_ms, total_ms2 = run_jobs_in_cpp(job_descs_)
                total_ms_ += total_ms
                total_ms_2 += total_ms2
                counter = 0
                for i, job in enumerate(jobs):
                    circuit_component = job["circuit_component"]
                    if job_descs[i] is not None:
                        circuit_component.IV_table = IVs[counter]
                        counter += 1
                    if job["all_children_are_CircuitElement"]:
                        circuit_component.dark_IV_table = circuit_component.IV_table.copy()
                        circuit_component.dark_IV_table[1,:] -= job["total_IL"]*job["area"]
                    job["done"] = True
                    self.job_done_index -= 1
                kernel_timer.toc()

            else:

        # duration = time.time()-t1
        # print(f"Dude, total time used was {duration}s and cpp took up {total_ms_/1000}s, {total_ms2/1000}s including marshalling")

                for i, job in enumerate(jobs):
                    kernel_timer.tic()
                    # job_descs = [make_job_desc(job) for _ in range(10)]

                    result = make_job_desc(job,run_immediately=True)
                    circuit_component = job["circuit_component"]
                    if result is not None:
                        # IVs, total_ms, total_ms2 = run_jobs_in_cpp(job_descs)
                        # IV = IVs[0]
                        IV, total_ms, total_ms2 = result[0], result[1], result[2]
                        total_ms_ += total_ms
                        total_ms_2 += total_ms2
                        circuit_component.IV_table = IV
                    kernel_timer.toc()
                    if job["all_children_are_CircuitElement"]:
                        circuit_component.dark_IV_table = circuit_component.IV_table.copy()
                        circuit_component.dark_IV_table[1,:] -= job["total_IL"]*job["area"]
                    job["done"] = True
                    self.job_done_index -= 1

        duration = time.time()-t1
        # print(f"Dude, total time used was {duration}s and cpp took up {total_ms_/1000}s, {total_ms_2/1000}s including marshalling, get job time was {get_job_time}s")
    def __str__(self):
        return str(self.job_list)


type_numbers = {"CurrentSource": 0, 
                "Resistor": 1, 
                "Diode": 2,
                "ForwardDiode": 2,
                "PhotonCouplingDiode": 2,
                "ReverseDiode": 3,
                "Intrinsic_Si_diode": 4,
                }

def make_job_desc(job, run_immediately=False):
    global kernel_timer
    circuit_component = job["circuit_component"]
    type_name = type(circuit_component).__name__
    circuit_component_type_number = 5 # CircuitGroup
    if type_name in type_numbers: # is CircuitElement
        circuit_component_type_number = type_numbers[type_name]

    max_I = None
    if circuit_component_type_number==0: # current source
        circuit_element_parameters = np.array([circuit_component.IL])
    elif circuit_component_type_number==1: # Resistor
        circuit_element_parameters = np.array([circuit_component.cond])
    else:
        max_I = 0.2
        if hasattr(circuit_component,"max_I"):
            max_I = circuit_component.max_I
        if circuit_component_type_number==2 or circuit_component_type_number==3: # Forward or Reversed Diodes
            circuit_element_parameters = np.array([circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I])
        elif circuit_component_type_number==4: # Intrinsic_Si_diode
            base_type_number = 0.0 # p
            if circuit_component.base_type=="n":
                base_type_number = 1.0
            circuit_element_parameters = np.array([circuit_component.base_doping, circuit_component.n, circuit_component.VT, circuit_component.base_thickness, max_I, circuit_component.ni, base_type_number])
        else:
            circuit_element_parameters = np.array([0,0,0,0]) # just a dummy for the CircuitGroup
            
    circuit_element_parameters = np.ascontiguousarray(circuit_element_parameters.astype(np.float64))

    total_IL = float(job["total_IL"])
    area = float(job["area"])

    if job["dark_IV_table"] is not None:
        if job["connection"] == "parallel":
            # skip the cpp  call, just return shifted IV
            IV_table = job["dark_IV_table"].copy()
            IV_table[1,:] += total_IL*area
            circuit_component.IV_table = IV_table
            return

    children_IVs = job["children_IVs"]
    children_types = job["children_types"]
    abs_max_num_points = 0
    type_numbers_ = []
    for children_type in children_types:
        type_name_ = children_type.__name__
        type_number = 5 # CircuitGroup
        if type_name_ in type_numbers: # is CircuitElement
            type_number = type_numbers[type_name_]
        type_numbers_.append(type_number)
    children_type_numbers = np.array(type_numbers_, dtype=np.int32)
    children_type_numbers = np.ascontiguousarray(children_type_numbers)  

    children_Vs, children_Is, children_offsets, children_lengths = build_children_buffers(children_IVs)

    children_Vs = np.ascontiguousarray(children_Vs)
    children_Is = np.ascontiguousarray(children_Is)
    children_offsets = np.ascontiguousarray(children_offsets)
    children_lengths = np.ascontiguousarray(children_lengths)
    abs_max_num_points += children_Vs.size

    children_pc_IVs = job["children_pc_IVs"]
    children_pc_Vs, children_pc_Is, children_pc_offsets, children_pc_lengths = build_children_buffers(children_pc_IVs)
    children_pc_Vs = np.ascontiguousarray(children_pc_Vs)
    children_pc_Is = np.ascontiguousarray(children_pc_Is)
    children_pc_offsets = np.ascontiguousarray(children_pc_offsets)
    children_pc_lengths = np.ascontiguousarray(children_pc_lengths)
    abs_max_num_points += children_Vs.size

    max_num_points = job["max_num_points"] if job["max_num_points"] is not None else -1
    max_num_points = int(max_num_points)
    cap_current = float(job["cap_current"]) if job["cap_current"] is not None else -1.0  # <=0 => no cap

    connection = int(-1)
    if job["connection"] == "series":
        connection = 0
    elif job["connection"] == "parallel":
        connection = 1
    else:
        connection = -1
    

    # --- output buffers ---
    max_num_points_ = 105
    if max_I is not None:
        max_num_points_ = np.ceil(100/0.2*max_I) + 5
    abs_max_num_points = max(abs_max_num_points, max_num_points_)
    abs_max_num_points = max(abs_max_num_points, max_num_points)
    abs_max_num_points = int(abs_max_num_points)

    if run_immediately:
        t1 = time.time()
        V, I, kernel_ms = ivkernel.run_single_job(
            int(connection),
            int(circuit_component_type_number),
            children_type_numbers.astype(np.int32, copy=False),
            children_Vs.astype(np.float64, copy=False),
            children_Is.astype(np.float64, copy=False),
            children_offsets.astype(np.int32, copy=False),
            children_lengths.astype(np.int32, copy=False),
            children_pc_Vs.astype(np.float64, copy=False),
            children_pc_Is.astype(np.float64, copy=False),
            children_pc_offsets.astype(np.int32, copy=False),
            children_pc_lengths.astype(np.int32, copy=False),
            float(total_IL),
            float(cap_current) if cap_current is not None else -1.0,
            int(max_num_points),
            float(area),
            int(abs_max_num_points),
            circuit_element_parameters.astype(np.float64, copy=False),
            int(abs_max_num_points)  # abs_max_num_points_out
        )
        kernel_timer.inc(kernel_ms/1000)
        t2 = time.time() - t1
        return np.vstack([V, I]), kernel_ms, t2*1000
    
    return (
            int(connection),
            int(circuit_component_type_number),
            children_type_numbers.astype(np.int32, copy=False),
            children_Vs.astype(np.float64, copy=False),
            children_Is.astype(np.float64, copy=False),
            children_offsets.astype(np.int32, copy=False),
            children_lengths.astype(np.int32, copy=False),
            children_pc_Vs.astype(np.float64, copy=False),
            children_pc_Is.astype(np.float64, copy=False),
            children_pc_offsets.astype(np.int32, copy=False),
            children_pc_lengths.astype(np.int32, copy=False),
            float(total_IL),
            float(cap_current) if cap_current is not None else -1.0,
            int(max_num_points),
            float(area),
            int(abs_max_num_points),  # passed into C++ for remeshing cap
            circuit_element_parameters.astype(np.float64, copy=False),
            int(abs_max_num_points),  # abs_max_num_points_out
        )

def run_jobs_in_cpp(job_descs):
    global kernel_timer
    """
    job_descs: list of argument tuples for ivkernel.run_single_job.
    Returns:
        IVs: list of 2×N_i IV tables
        total_kernel_ms: sum of kernel times from C++
        total_wall_ms: wall-clock time (ms), including marshalling
    """
    if not job_descs:
        return [], 0.0, 0.0

    t0 = time.time()
    IVs_, kernel_ms = ivkernel.run_multiple_jobs_in_parallel(job_descs)
    wall_ms = (time.time() - t0) * 1000.0
    IVs = [np.vstack(IV_) for IV_ in IVs_]

    kernel_timer.inc(kernel_ms/1000)

    return IVs, kernel_ms, wall_ms






