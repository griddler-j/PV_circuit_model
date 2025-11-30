from tqdm import tqdm

try:
    from PV_Circuit_Model import ivkernel  # the compiled Cython extension
    _HAVE_IVKERNEL = True
except Exception as e:
    _HAVE_IVKERNEL = False
    ivkernel = None
    import warnings
    warnings.warn(
        "The optimized C++/Cython extension 'ivkernel' could not be imported.\n"
        "Falling back to pure-Python implementation (much slower).\n",
        RuntimeWarning
    )
    from PV_Circuit_Model.ivkernal_python import build_component_IV_python

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component):
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
    if job_done_index > 100000:
        pbar = tqdm(total=job_done_index, desc="Processing the circuit hierarchy: ")
    while job_done_index > 0:
        components_, min_index = get_runnable_iv_jobs(components, children_job_ids, job_done_index)
        if len(components_) > 0:
            if _HAVE_IVKERNEL:
                ivkernel.run_multiple_jobs(components_,refine_mode=refine_mode,parallel=False)
            else:
                for component in components_:
                    build_component_IV_python(component,refine_mode=refine_mode)
        if pbar is not None:
            pbar.update(job_done_index-min_index)
        job_done_index = min_index
    if pbar is not None:
        pbar.close()


