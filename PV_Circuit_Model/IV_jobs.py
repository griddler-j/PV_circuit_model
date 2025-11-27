import numpy as np
import os
import time
import PV_Circuit_Model.ivkernel as ivkernel  
from matplotlib import pyplot as plt

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
            print(f"{i}: {event["component"]}\t{event["kernel_timer"]}\t{event["wrapper_timer"]}")
    def __str__(self):
        return f"kernel_timer = {self.kernel_timer}, wrapper_timer = {self.wrapper_timer}"
    
kernel_timer = KernelTimer()

def get_runnable_iv_jobs(job_list, job_done_index, refine_mode=False):
    include_indices = []
    for i in range(job_done_index-1,-1,-1):
        job = job_list[i]
        if len(job["children_job_ids"])>0 and min(job["children_job_ids"])<job_done_index:
            return [job_list[j] for j in include_indices], i+1
        if refine_mode:
            circuit_component = job["circuit_component"]
            type_name = type(circuit_component).__name__
            if type_name not in ["CurrentSource", "Resistor"]:
                include_indices.append(i)
        else:
            include_indices.append(i)
    return [job_list[j] for j in include_indices], min(include_indices)

def run_iv_jobs(job_list, refine_mode=False):
    job_done_index = len(job_list)
    while job_done_index > 0:
        jobs, min_include_index = get_runnable_iv_jobs(job_list, job_done_index, refine_mode=refine_mode)
        ivkernel.run_multiple_jobs(jobs,refine_mode=refine_mode)
        job_done_index = min_include_index

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component,max_num_points=None, cap_current=None):
        self.job_list = []
        this_job_id = self.add(circuit_component)
        self.max_num_points = max_num_points
        self.cap_current = cap_current
        self.build(circuit_component,this_job_id)
        # kernel_timer.register(circuit_component)
    def add(self,circuit_component,parent_id=None):
        this_job_id = len(self.job_list)
        self.job_list.append({"circuit_component": circuit_component, "children_job_ids": [], "done": False})
        return this_job_id
    def build(self,circuit_component,this_job_id=None):
        if hasattr(circuit_component,"subgroups"):
            new_job_ids = []
            for _, element in enumerate(circuit_component.subgroups):
                if element.IV_table is None:
                    new_job_id = self.add(element, parent_id=this_job_id)
                    self.job_list[this_job_id]["children_job_ids"].append(new_job_id)
                    new_job_ids.append(new_job_id)
                else:
                    new_job_ids.append(-1)
            for i, element in enumerate(circuit_component.subgroups):
                if element.IV_table is None:
                    self.build(element, this_job_id=new_job_ids[i])
    def get_runnable_jobs(self,refine_mode=False):
        include_indices = []
        for i in range(self.job_done_index-1,-1,-1):
            job = self.job_list[i]
            if len(job["children_job_ids"])>0 and min(job["children_job_ids"])<self.job_done_index:
                return [self.job_list[j] for j in include_indices], i+1
            if refine_mode:
                circuit_component = job["circuit_component"]
                type_name = type(circuit_component).__name__
                if type_name not in ["CurrentSource", "Resistor"]:
                    include_indices.append(i)
            else:
                include_indices.append(i)
        return [self.job_list[j] for j in include_indices], min(include_indices)
    def run_IV(self):
        run_iv_jobs(self.job_list)
    def refine_IV(self):
        run_iv_jobs(self.job_list, refine_mode=True)
    def __str__(self):
        return str(self.job_list)






