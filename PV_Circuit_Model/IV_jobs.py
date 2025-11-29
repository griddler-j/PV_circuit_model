import time
import PV_Circuit_Model.ivkernel as ivkernel  
from tqdm import tqdm

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component):
        t1 = time.time()
        self.components = []
        self.children_job_ids = []
        self.add(circuit_component) # now job_list has one
        self.build()
    def add(self,circuit_component):
        self.components.append(circuit_component)
        self.children_job_ids.append([])
        return len(self.components)-1
    def build(self):
        pos = 0
        while pos < len(self.components):
            circuit_component = self.components[pos] 
            subgroups = getattr(circuit_component, "subgroups", None)
            if subgroups:
                for element in subgroups:
                    if element.IV_V is None:
                        new_job_id = self.add(element)
                        self.children_job_ids[pos].append(new_job_id)
            pos += 1
    def run_IV(self):
        run_iv_jobs(self.components,self.children_job_ids)
    def refine_IV(self):
        run_iv_jobs(self.components,self.children_job_ids, refine_mode=True)

def get_runnable_iv_jobs(components, children_job_ids, job_done_index):
    include_indices = []
    for i in range(job_done_index-1,-1,-1):
        ids = children_job_ids[i]
        if len(ids)>0 and min(ids)<job_done_index:
            return [components[j] for j in include_indices], i+1
        if components[i].IV_V is None:
            include_indices.append(i)
    return [components[j] for j in include_indices], 0

def run_iv_jobs(components, children_job_ids, refine_mode=False):
    ivkernel.pin_to_p_cores_only_()
    job_done_index = len(components)
    pbar = None
    t1s = 0
    t2s = 0
    t3s = 0
    if job_done_index > 100000:
        pbar = tqdm(total=job_done_index, desc="Processing the circuit hierarchy: ")
    while job_done_index > 0:
        components_, min_index = get_runnable_iv_jobs(components, children_job_ids, job_done_index)
        if len(components_) > 0:
            t1, t2, t3 = ivkernel.run_multiple_jobs(components_,refine_mode=refine_mode,parallel=False)

            # print(f"{len(components_)}: {t2}, {t1/1000}, {t3}")
            t1s += t1
            t2s += t2
            t3s += t3
        if pbar is not None:
            pbar.update(job_done_index-min_index)
        job_done_index = min_index
    if pbar is not None:
        pbar.close()
    print(f"all done: {t2s}, {t1s/1000}, {t3s}")
    
    # print(f"Dang, took {t2}s to get runnable iv jobs, {t2b}s to run them")


