import numpy as np
import ctypes

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
    def get_runnable_jobs(self):
        for i in range(self.job_done_index-1,-1,-1):
            job = self.job_list[i]
            if len(job["children_job_ids"])>0 and min(job["children_job_ids"])<self.job_done_index:
                return self.job_list[i+1:self.job_done_index]
            # prep
            circuit_component = job["circuit_component"]
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
            if hasattr(circuit_component,"subgroups"):
                for element in circuit_component.subgroups:
                    job["children_types"].append(type(element))
                    if hasattr(element,"IL"):
                        element.set_IL(element.IL)
                        job["total_IL"] -= element.IL
                        job["children_IVs"].append(np.zeros((2, 0)))
                    else:
                        job["children_IVs"].append(element.IV_table.copy())

                    job["children_pc_IVs"].append(np.zeros((2, 0)))
                    if hasattr(element,"photon_coupling_diodes"):
                        pc_IV = element.photon_coupling_diodes[0].IV_table.copy()
                        pc_IV[1,:] *= element.area
                        job["children_pc_IVs"][-1] = pc_IV
        return []

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

# Load the shared library (example for Linux/macOS; adjust extension/path for Windows)
lib = ctypes.CDLL("./libivkernel.so")

combine_iv_job = lib.combine_iv_job
combine_iv_job.argtypes = [
    ctypes.c_int,                         # connection
    ctypes.c_int,                         # circuit_component_type_number
    ctypes.c_int,                         # n_children
    ctypes.POINTER(ctypes.c_int),         #children_type_numbers
    ctypes.POINTER(ctypes.c_double),      # children_Vs
    ctypes.POINTER(ctypes.c_double),      # children_Is
    ctypes.POINTER(ctypes.c_int),      # children_offsets
    ctypes.POINTER(ctypes.c_int),      # children_lengths
    ctypes.c_int,                       #(children_Vs_size)
    ctypes.POINTER(ctypes.c_double),      # children_pc_Vs
    ctypes.POINTER(ctypes.c_double),      # children_pc_Is
    ctypes.POINTER(ctypes.c_int),      # children_pc_offsets
    ctypes.POINTER(ctypes.c_int),      # children_pc_lengths
    ctypes.c_int,                         #(children_Vs_size)
    ctypes.c_double,                      # total_IL
    ctypes.c_double,                      # cap_current
    ctypes.c_int,                         # max_num_points
    ctypes.c_double,                      # area
    ctypes.c_int,                         # abs_max_num_points
    ctypes.POINTER(ctypes.c_double),      # circuit_element_parameters
    ctypes.POINTER(ctypes.c_double),         # out_V
    ctypes.POINTER(ctypes.c_double),         # out_I
    ctypes.POINTER(ctypes.c_int)            # out_len
]

combine_iv_job.restype = None

def run_job_in_cpp(job):
    circuit_component = job["circuit_component"]
    type_name = type(circuit_component).__name__
    circuit_component_type_number = 5 # CircuitGroup
    if type_name in type_numbers: # is CircuitElement
        circuit_component_type_number = type_numbers[type_name]

    if circuit_component_type_number==0: # current source
        circuit_element_parameters = np.array([circuit_component.IL])
    elif circuit_component_type_number==1: # Resistor
        circuit_element_parameters = np.array([circuit_component.cond])
    else:
        max_I = 0.2
        if hasattr(circuit_component,"max_I"):
            max_I = circuit_component.max_I
        elif circuit_component_type_number==2 or circuit_component_type_number==3: # Forward or Reversed Diodes
            circuit_element_parameters = np.array([circuit_component.I0, circuit_component.n, circuit_component.VT, circuit_component.V_shift, max_I])
        elif circuit_component_type_number==4: # Intrinsic_Si_diode
            base_type_number = 0.0 # p
            if circuit_component.base_type=="n":
                base_type_number = 1.0
            circuit_element_parameters = np.array([circuit_component.base_doping, circuit_component.n, circuit_component.VT, circuit_component.base_thickness, max_I, circuit_component.ni, base_type_number])

    circuit_element_parameters = np.ascontiguousarray(circuit_element_parameters.astype(np.float64))
    circuit_element_parameters_ptr = circuit_element_parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    total_IL = float(job["total_IL"])
    area = float(job["area"])

    if job["dark_IV_table"] is not None:
        if job["connection"] == "parallel":
            # skip the cpp  call, just return shifted IV
            IV_table = job["dark_IV_table"].copy()
            IV_table[1,:] += total_IL*area
            return IV_table

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
            type_name = children_type.__name__
            type_number = 4 # CircuitGroup
            if type_name in type_numbers: # is CircuitElement
                type_number = type_numbers[type_name]
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
        abs_max_num_points += children_pc_Vs.size
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
    abs_max_num_points = max(abs_max_num_points, 500)
    abs_max_num_points = max(abs_max_num_points, max_num_points)
    abs_max_num_points = int(abs_max_num_points)
    out_V = np.empty(abs_max_num_points, dtype=np.float64)
    out_I = np.empty(abs_max_num_points, dtype=np.float64)
    out_len = ctypes.c_int(0)
    out_V_ptr = out_V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_I_ptr = out_I.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_len_ptr = ctypes.byref(out_len)

    combine_iv_job(
        ctypes.c_int(connection),
        ctypes.c_int(circuit_component_type_number),
        ctypes.c_int(n_children),
        children_type_numbers_ptr,
        children_Vs_ptr, children_Is_ptr, children_offsets_ptr, children_lengths_ptr,ctypes.c_int(children_Vs_size),
        children_pc_Vs_ptr, children_pc_Is_ptr, children_pc_offsets_ptr, children_pc_lengths_ptr,ctypes.c_int(children_pc_Vs_size),
        ctypes.c_double(total_IL),
        ctypes.c_double(cap_current),
        ctypes.c_int(max_num_points),
        ctypes.c_double(area),
        ctypes.c_int(abs_max_num_points),
        circuit_element_parameters_ptr,
        out_V_ptr,
        out_I_ptr,
        out_len_ptr
    )

    used = out_len.value
    V = out_V[:used].copy()
    I = out_I[:used].copy()

    return np.vstack([V, I])

