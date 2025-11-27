import numpy as np
import os
import time
import PV_Circuit_Model.ivkernel as ivkernel  
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

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
        self.build_IV_events.append({'component':type(circuit_component).__name__, 'wrapper_timer': 0.0, 'kernel_timer': 0.0})
    def tic(self):
        self.wrapper_timer_tic = time.time()
    def toc(self):
        assert(self.wrapper_timer_tic >= 0)
        lapse = time.time()-self.wrapper_timer_tic
        self.build_IV_events[-1]["wrapper_timer"] += lapse
        self.wrapper_timer += lapse
        self.wrapper_timer_tic = -1.0
    def inc(self,time_):
        self.build_IV_events[-1]["kernel_timer"] += time_
        self.kernel_timer += time_
    def show_log(self):
        for i, event in enumerate(self.build_IV_events):
            print(f'{i}: {event["component"]}\t{event["kernel_timer"]}\t{event["wrapper_timer"]}')
    def __str__(self):
        return f"kernel_timer = {self.kernel_timer}, wrapper_timer = {self.wrapper_timer}"
    
kernel_timer = KernelTimer()

def get_runnable_iv_jobs(job_list, job_done_index, refine_mode=False):
    include_indices = []
    for i in range(job_done_index-1,-1,-1):
        job = job_list[i]
        if len(job["children_job_ids"])>0 and min(job["children_job_ids"])<job_done_index:
            if "circuit_component_type_number" in job:
                return include_indices
            else:
                return [job_list[j] for j in include_indices], i+1
        if refine_mode:
            if "circuit_component_type_number" in job:
                if job["circuit_component_type_number"] > 1:
                    include_indices.append(i)
            else:
                circuit_component = job["circuit_component"]
                type_name = type(circuit_component).__name__
                if type_name not in ["CurrentSource", "Resistor"]:
                    include_indices.append(i)
        else:
            include_indices.append(i)
    if "circuit_component_type_number" in job:
        return include_indices
    else:
        return [job_list[j] for j in include_indices], 0

def run_iv_jobs(job_list, refine_mode=False):
    job_done_index = len(job_list)
    while job_done_index > 0:
        jobs, min_include_index = get_runnable_iv_jobs(job_list, job_done_index, refine_mode=refine_mode)
        ivkernel.run_multiple_jobs(jobs,refine_mode=refine_mode)
        job_done_index = min_include_index

def run_iv_jobs_simplified(job_list, aux_IV_list, refine_mode=False):
    job_done_index = len(job_list)
    while job_done_index > 0:
        include_indices = get_runnable_iv_jobs(job_list, job_done_index, refine_mode=refine_mode)
        ivkernel.run_multiple_jobs_simplified(job_list,aux_IV_list,include_indices,refine_mode=refine_mode)
        job_done_index = min(include_indices)
    return [{"IV_table": job["IV"],"dark_IV_table": job["dark_IV"]} for job in job_list] 

def make_simplified_job_list(job_list):
    type_numbers = {
        "CurrentSource": 0,
        "Resistor": 1,
        "Diode": 2,
        "ForwardDiode": 2,
        "PhotonCouplingDiode": 2,
        "ReverseDiode": 3,
        "Intrinsic_Si_diode": 4,
        "CircuitGroup": 5,
    }
    job_list_ = []
    aux_IV_list_ = []
    for job in job_list:
        circuit_component = job["circuit_component"]
        dark_IV = None
        if hasattr(circuit_component,"dark_IV_table"):
            dark_IV = circuit_component.dark_IV_table
        max_I = 0.2
        if hasattr(circuit_component, "max_I"):
            max_I = circuit_component.max_I
        type_name = type(circuit_component).__name__
        circuit_component_type_number = 5  # default CircuitGroup
        if type_name in type_numbers:
            circuit_component_type_number = type_numbers[type_name]
        if circuit_component_type_number == 0:      # CurrentSource
            params = np.array([circuit_component.IL], dtype=np.float64)
        elif circuit_component_type_number == 1:    # Resistor
            params = np.array([circuit_component.cond], dtype=np.float64)
        elif circuit_component_type_number in (2, 3):  # diodes
            params = np.array([
                circuit_component.I0,
                circuit_component.n,
                circuit_component.VT,
                circuit_component.V_shift,
                max_I,
            ], dtype=np.float64)
        elif circuit_component_type_number == 4:    # Intrinsic_Si_diode
            base_type_number = 0.0  # p
            try:
                if circuit_component.base_type == "n":
                    base_type_number = 1.0
            except AttributeError:
                pass
            params = np.array([
                circuit_component.base_doping,1,circuit_component.VT,
                circuit_component.base_thickness,
                max_I,
                circuit_component.ni,
                base_type_number,
            ], dtype=np.float64)
        else:
            # CircuitGroup or unknown
            params = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        total_IL = 0.0
        area = 1.0
        if hasattr(circuit_component,"shape") and hasattr(circuit_component,"area") and circuit_component.area is not None:
            area = circuit_component.area
        cap_current = -1.0
        if hasattr(circuit_component,"cap_current"):
            cap_current = circuit_component.cap_current if circuit_component.cap_current is not None else -1.0
        max_num_points = -1
        if hasattr(circuit_component,"max_num_points"):
            max_num_points = circuit_component.max_num_points if circuit_component.max_num_points is not None else -1

        connection = -1
        if hasattr(circuit_component, "connection"):
            connection = 0  # series default
            if circuit_component.connection == "parallel":
                connection = 1

        n_children = 0
        all_children_are_CircuitElement = 0
        if hasattr(circuit_component,"subgroups"):
            n_children = len(circuit_component.subgroups)
            all_children_are_CircuitElement = 1

        operating_V = None
        if hasattr(circuit_component,"operating_point") and circuit_component.operating_point is not None:
            operating_V = circuit_component.operating_point[0]

        for j in range(n_children):
            type_number = 5
            cname = type(circuit_component.subgroups[j]).__name__
            if cname in type_numbers:
                type_number = type_numbers[cname]
            else: # not CircuitElement
                all_children_are_CircuitElement = 0

            if type_number==0: # current source
                total_IL -= circuit_component.subgroups[j].IL

        job_list_.append({"circuit_component_type_number": circuit_component_type_number, 
                            "total_IL": total_IL,
                            "operating_V": operating_V,
                            "all_children_are_CircuitElement": all_children_are_CircuitElement,
                            "max_I": max_I,"params":params,"area":area,"connection":connection,
                            "IV": circuit_component.IV_table, "dark_IV": dark_IV,"cap_current":cap_current,"max_num_points":max_num_points,
                            "children_job_ids": job["children_job_ids"], "children_job_ids_ordered": job["children_job_ids_ordered"]})
        if hasattr(circuit_component,"photon_coupling_diodes") and len(circuit_component.photon_coupling_diodes)>0:
            job_list_[-1]["pc_IV_table"] = circuit_component.photon_coupling_diodes[0].IV_table
            job_list_[-1]["pc_IV_table_scale"] = circuit_component.area
        for i, id in enumerate(job["children_job_ids_ordered"]):
            if id == -1:
                job_list_[-1]["children_job_ids_ordered"][i] = -1*len(aux_IV_list_)
                element = circuit_component.subgroups[i]
                type_number = 5
                type_name = type(element).__name__
                if type_name in type_numbers:
                    type_number = type_numbers[type_name]
                aux_IV_list_.append({"circuit_component_type_number": type_number, "IV": element.IV_table})
                if hasattr(element,"photon_coupling_diodes") and len(element.photon_coupling_diodes)>0:
                    aux_IV_list_[-1]["pc_IV_table"] = element.photon_coupling_diodes[0].IV_table
                    aux_IV_list_[-1]["pc_IV_table_scale"] = element.area
    return job_list_, aux_IV_list_

def run_iv_job_heaps_parallel(job_list_list,refine_mode=False):
    # n_workers = min(len(job_list_list), multiprocessing.cpu_count())
    n_workers = 26
    # simplify the list structure first
    job_list_list_ = []
    for job_list in job_list_list:
        job_list_, aux_IV_list_ = make_simplified_job_list(job_list)
        job_list_list_.append({"job_list_": job_list_, "aux_IV_list_": aux_IV_list_})
    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(run_iv_jobs_simplified)(item["job_list_"], item["aux_IV_list_"], refine_mode)
        for item in job_list_list_
    )
    for i, job_list in enumerate(job_list_list):
        result = results[i]
        for j, job in enumerate(job_list):
            circuit_component = job["circuit_component"]
            circuit_component.IV_table = result[j]["IV_table"]
            if result[j]["dark_IV_table"] is not None:
                circuit_component.dark_IV_table = result[j]["dark_IV_table"]

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component,max_num_points=None, cap_current=None):
        self.job_list = []
        self.add(circuit_component) # now job_list has one
        self.max_num_points = max_num_points
        self.cap_current = cap_current
        self.build()
        # kernel_timer.register(circuit_component)
    def add(self,circuit_component):
        self.job_list.append({"circuit_component": circuit_component, "children_job_ids": [], "children_job_ids_ordered": []})
        return len(self.job_list)-1
    def build(self):
        pos = 0
        while pos < len(self.job_list):
            circuit_component = self.job_list[pos]["circuit_component"] 
            if hasattr(circuit_component,"subgroups"):
                new_job_ids = []
                for _, element in enumerate(circuit_component.subgroups):
                    if element.IV_table is None:
                        new_job_id = self.add(element)
                        self.job_list[pos]["children_job_ids"].append(new_job_id)
                        new_job_ids.append(new_job_id)
                    else:
                        new_job_ids.append(-1)
                self.job_list[pos]["children_job_ids_ordered"] = new_job_ids
            pos += 1
    def run_IV(self):
        run_iv_jobs(self.job_list)
    def refine_IV(self):
        # circuit_component = self.job_list[0]["circuit_component"]
        # if type(circuit_component.subgroups[0]).__name__=="Module":
        #     job_list_list = []
        #     for module in circuit_component.subgroups:
        #         job_list_list.append(module.job_heap.job_list)
        #     run_iv_job_heaps_parallel(job_list_list,refine_mode=True)
        run_iv_jobs(self.job_list, refine_mode=True)
    def __str__(self):
        return str(self.job_list)






