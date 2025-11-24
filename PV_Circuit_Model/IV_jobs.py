import numpy as np
import ctypes
import os
import time

# Load the shared library (example for Linux/macOS; adjust extension/path for Windows)
if os.name == "nt":
    lib = ctypes.CDLL("PV_Circuit_Model/ivkernel.dll")
else:
    lib = ctypes.CDLL("./libivkernel.so")

# A heap structure to store I-V jobs
class IV_Job_Heap:
    def __init__(self,circuit_component,max_num_points=None, cap_current=None):
        self.job_list = []
        this_job_id = self.add(circuit_component)
        self.max_num_points = max_num_points
        self.cap_current = cap_current
        self.build(circuit_component,this_job_id)
        self.job_done_index = len(self.job_list)
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
        while self.job_done_index > 0:
            jobs = self.get_runnable_jobs()

            if len(jobs) >= 1000:
                job_descs = []
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

            else:

        # duration = time.time()-t1
        # print(f"Dude, total time used was {duration}s and cpp took up {total_ms_/1000}s, {total_ms2/1000}s including marshalling")

                for i, job in enumerate(jobs):
                    result = make_job_desc(job,run_immediately=True)
                    circuit_component = job["circuit_component"]
                    if result is not None:
                        IV, total_ms, total_ms2 = result[0], result[1], result[2]
                        total_ms_ += total_ms
                        total_ms_2 += total_ms2
                        circuit_component.IV_table = IV
                    if job["all_children_are_CircuitElement"]:
                        circuit_component.dark_IV_table = circuit_component.IV_table.copy()
                        circuit_component.dark_IV_table[1,:] -= job["total_IL"]*job["area"]
                    job["done"] = True
                    self.job_done_index -= 1

        # duration = time.time()-t1
        # print(f"Dude, total time used was {duration}s and cpp took up {total_ms_/1000}s, {total_ms2/1000}s including marshalling")
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

class JobDesc(ctypes.Structure):
    _fields_ = [
        ("connection", ctypes.c_int),
        ("circuit_component_type_number", ctypes.c_int),
        ("n_children", ctypes.c_int),
        ("children_type_numbers", ctypes.POINTER(ctypes.c_int)),
        ("children_Vs", ctypes.POINTER(ctypes.c_double)),
        ("children_Is", ctypes.POINTER(ctypes.c_double)),
        ("children_offsets", ctypes.POINTER(ctypes.c_int)),
        ("children_lengths", ctypes.POINTER(ctypes.c_int)),
        ("children_Vs_size", ctypes.c_int),
        ("children_pc_Vs", ctypes.POINTER(ctypes.c_double)),
        ("children_pc_Is", ctypes.POINTER(ctypes.c_double)),
        ("children_pc_offsets", ctypes.POINTER(ctypes.c_int)),
        ("children_pc_lengths", ctypes.POINTER(ctypes.c_int)),
        ("children_pc_Vs_size", ctypes.c_int),
        ("total_IL", ctypes.c_double),
        ("cap_current", ctypes.c_double),
        ("max_num_points", ctypes.c_int),
        ("area", ctypes.c_double),
        ("abs_max_num_points", ctypes.c_int),
        ("circuit_element_parameters", ctypes.POINTER(ctypes.c_double)),
        ("out_V", ctypes.POINTER(ctypes.c_double)),
        ("out_I", ctypes.POINTER(ctypes.c_double)),
        ("out_len", ctypes.POINTER(ctypes.c_int)),
    ]

lib.combine_iv_job.argtypes = [ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int)]
lib.combine_iv_job.restype = ctypes.c_double 

lib.combine_iv_job_batch.argtypes = [
    ctypes.POINTER(JobDesc),  # jobs array
    ctypes.c_int              # n_jobs
]
lib.combine_iv_job_batch.restype = ctypes.c_double 

def make_job_desc(job, run_immediately=False):
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
    circuit_element_parameters_ptr = circuit_element_parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

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
    children_Vs = None
    children_Vs_size = 0
    children_Is = None
    children_Is_ptr = None
    children_Vs_ptr = None
    children_offsets = None
    children_lengths = None
    children_offsets_ptr = None
    children_lengths_ptr = None
    children_type_numbers = None
    children_type_numbers_ptr = None
    n_children = len(children_IVs)
    abs_max_num_points = 0
    if n_children>0:
        type_numbers_ = []
        for children_type in children_types:
            type_name_ = children_type.__name__
            type_number = 5 # CircuitGroup
            if type_name_ in type_numbers: # is CircuitElement
                type_number = type_numbers[type_name_]
            type_numbers_.append(type_number)
        children_type_numbers = np.array(type_numbers_, dtype=np.int32)
        children_type_numbers = np.ascontiguousarray(children_type_numbers)  
        children_type_numbers_ptr = children_type_numbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        children_lengths = np.array([c.shape[1] for c in children_IVs], dtype=np.int32)
        children_offsets = np.empty(len(children_lengths), dtype=np.int32)
        children_offsets[0] = 0
        children_offsets[1:] = np.cumsum(children_lengths[:-1])
        children_Vs = np.concatenate([c[0] for c in children_IVs]).astype(np.float64)
        children_Is = np.concatenate([c[1] for c in children_IVs]).astype(np.float64)
        children_Vs = np.ascontiguousarray(children_Vs)
        children_Is = np.ascontiguousarray(children_Is)
        children_offsets = np.ascontiguousarray(children_offsets)
        children_lengths = np.ascontiguousarray(children_lengths)
        children_Vs_ptr = children_Vs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        children_Is_ptr = children_Is.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        children_offsets_ptr = children_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        children_lengths_ptr = children_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        abs_max_num_points += children_Vs.size
        children_Vs_size = children_Vs.size


    children_pc_IVs = job["children_pc_IVs"]
    children_pc_Vs = None
    children_pc_Is = None
    children_pc_Is_ptr = None
    children_pc_Vs_ptr = None
    children_pc_offsets = None
    children_pc_lengths = None
    children_pc_offsets_ptr = None
    children_pc_lengths_ptr = None
    children_pc_Vs_size = 0
    if n_children>0:
        children_pc_lengths = np.array([c.shape[1] for c in children_pc_IVs], dtype=np.int32)
        children_pc_offsets = np.empty(len(children_pc_lengths), dtype=np.int32)
        children_pc_offsets[0] = 0
        children_pc_offsets[1:] = np.cumsum(children_pc_lengths[:-1])
        children_pc_Vs = np.concatenate([c[0] for c in children_pc_IVs]).astype(np.float64)
        children_pc_Is = np.concatenate([c[1] for c in children_pc_IVs]).astype(np.float64)
        children_pc_Vs = np.ascontiguousarray(children_pc_Vs)
        children_pc_Is = np.ascontiguousarray(children_pc_Is)
        children_pc_offsets = np.ascontiguousarray(children_pc_offsets)
        children_pc_lengths = np.ascontiguousarray(children_pc_lengths)
        children_pc_Vs_ptr = children_pc_Vs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        children_pc_Is_ptr = children_pc_Is.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        children_pc_offsets_ptr = children_pc_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        children_pc_lengths_ptr = children_pc_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        abs_max_num_points += children_Vs.size
        children_pc_Vs_size = children_pc_Vs.size


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
    out_V = np.empty(abs_max_num_points, dtype=np.float64)
    out_I = np.empty(abs_max_num_points, dtype=np.float64)
    out_len = ctypes.c_int(0)
    out_V_ptr = out_V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_I_ptr = out_I.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_len_ptr = ctypes.pointer(out_len)

    if run_immediately:
        t1 = time.time()
        total_ms = lib.combine_iv_job(ctypes.c_int(connection),
                        ctypes.c_int(circuit_component_type_number),
                        ctypes.c_int(n_children),
                        children_type_numbers_ptr,
                        children_Vs_ptr,
                        children_Is_ptr,
                        children_offsets_ptr,
                        children_lengths_ptr,
                        ctypes.c_int(children_Vs_size),
                        children_pc_Vs_ptr,
                        children_pc_Is_ptr,
                        children_pc_offsets_ptr,
                        children_pc_lengths_ptr,
                        ctypes.c_int(children_pc_Vs_size),
                        ctypes.c_double(total_IL),
                        ctypes.c_double(cap_current),
                        ctypes.c_int(max_num_points),
                        ctypes.c_double(area),
                        ctypes.c_int(abs_max_num_points),
                        circuit_element_parameters_ptr,
                        out_V_ptr,
                        out_I_ptr,
                        out_len_ptr)
        
        used = out_len.value
        assert(used <= abs_max_num_points)
        V = out_V[:used]
        I = out_I[:used]
        t2 = time.time() - t1
        return np.vstack([V, I]), total_ms, t2*1000

    job_desc = JobDesc()
    job_desc.connection = ctypes.c_int(connection)
    job_desc.circuit_component_type_number = ctypes.c_int(circuit_component_type_number)
    job_desc.n_children = ctypes.c_int(n_children)
    job_desc.children_type_numbers = children_type_numbers_ptr
    job_desc.children_Vs = children_Vs_ptr
    job_desc.children_Is = children_Is_ptr
    job_desc.children_offsets = children_offsets_ptr
    job_desc.children_lengths = children_lengths_ptr

    job_desc._children_type_numbers = children_type_numbers
    job_desc._children_Vs = children_Vs
    job_desc._children_Is = children_Is
    job_desc._children_offsets = children_offsets
    job_desc._children_lengths = children_lengths

    job_desc.children_Vs_size = ctypes.c_int(children_Vs_size)
    job_desc.children_pc_Vs = children_pc_Vs_ptr
    job_desc.children_pc_Is = children_pc_Is_ptr
    job_desc.children_pc_offsets = children_pc_offsets_ptr
    job_desc.children_pc_lengths = children_pc_lengths_ptr

    job_desc._children_pc_Vs = children_pc_Vs
    job_desc._children_pc_Is = children_pc_Is
    job_desc._children_pc_offsets = children_pc_offsets
    job_desc._children_pc_lengths = children_pc_lengths

    job_desc.children_pc_Vs_size = ctypes.c_int(children_pc_Vs_size)
    job_desc.total_IL = ctypes.c_double(total_IL)
    job_desc.cap_current = ctypes.c_double(cap_current)
    job_desc.max_num_points = ctypes.c_int(max_num_points)
    job_desc.area = ctypes.c_double(area)
    job_desc.abs_max_num_points = ctypes.c_int(abs_max_num_points)
    job_desc.circuit_element_parameters = circuit_element_parameters_ptr
    job_desc._circuit_element_parameters = circuit_element_parameters
    job_desc.out_V = out_V_ptr
    job_desc.out_I = out_I_ptr
    job_desc.out_len = out_len_ptr
    job_desc._out_V = out_V
    job_desc._out_I = out_I
    job_desc._out_len = out_len

    return job_desc

# def run_job_in_cpp(job_desc):
#     t1 = time.time()
#     total_ms = lib.combine_iv_job(job_desc)
#     used = job_desc.out_len.contents.value
#     assert(used <= job_desc.abs_max_num_points)
#     V = job_desc._out_V[:used].copy()
#     I = job_desc._out_I[:used].copy()
#     t2 = time.time() - t1
#     return np.vstack([V, I]), total_ms, t2*1000

def run_jobs_in_cpp(job_desc_list):
    t1 = time.time()
    JobDescArray = JobDesc * len(job_desc_list)
    job_desc_array = JobDescArray(*job_desc_list)
    total_ms = lib.combine_iv_job_batch(job_desc_array, len(job_desc_array))
    IVs = []
    for jd in job_desc_list:
        used = jd._out_len.value
        assert used <= jd.abs_max_num_points
        V = jd._out_V[:used].copy()
        I = jd._out_I[:used].copy()
        IVs.append(np.vstack([V, I]))
    t2 = time.time() - t1
    # print(f"{len(job_desc_list)} jobs took {total_ms} ms")
    return IVs, total_ms, t2*1000






