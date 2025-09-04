import numpy as np
from matplotlib import pyplot as plt
from PV_Circuit_Model.cell_analysis import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
from PV_Circuit_Model.utilities import *
import numbers
import os
import json

class Measurement():
    keys = []
    data_rows = []
    # measurement can be on its own, or belonging to a device
    def __init__(self,measurement_condition=None,measurement_data=None,json_filepath=None,device=None,key_parameters=None):
        # either must input the measurement_condition + measurement_data, or
        # read these from a json file
        assert((measurement_condition is not None and (measurement_data is not None or key_parameters is not None)) or json_filepath is not None)
        self.aux = {}
        if json_filepath is not None:
            self.aux["filepath"] = json_filepath
            self.read_file(json_filepath)
        else:
            self.measurement_condition = measurement_condition
            self.measurement_data = measurement_data
            self.simulated_data = None
            self.tag = None
            self.fit_weight = {}
        self.key_parameters = {}
        self.simulated_key_parameters = {}
        self.simulated_key_parameters_baseline = {}
        self.unit_errors = {}
        self.parent_device=device
        if key_parameters is not None:
            self.key_parameters = key_parameters
        else:
            self.derive_key_parameters(self.measurement_data,self.key_parameters,self.measurement_condition)
        self.set_unit_errors()
    def set_unit_error(self,key,value=None):
        if value is not None:
            self.unit_errors[key] = value
        else: # just make as percentage, say, one in a thousand
            if isinstance(self.key_parameters[key],numbers.Number):
                avg_ = abs(self.key_parameters[key])
                self.unit_errors[key] = avg_/1000
            else: 
                avg_ = np.mean(np.abs(self.key_parameters[key])) 
                self.unit_errors[key] = avg_*np.sqrt(self.key_parameters[key].size)/1000
    def set_unit_errors(self):
        for key in self.keys:
            self.set_unit_error(key)
    @staticmethod
    def derive_key_parameters(data, key_parameters, conditions):
        pass
    def simulate(self,device=None):
        pass
    def plot(self):
        self.plot_func(self.measurement_data,color="blue")
        if self.simulated_data is not None:
            self.plot_func(self.simulated_data,color="red")
        plt.show()
    @staticmethod
    def plot_func(x,y,xlabel=None,ylabel=None,title=None,color="black",ax=None,kwargs={}):
        if kwargs is None:
            kwargs = {}
        if "color" not in kwargs:
            kwargs["color"] = color
        if ax is None:
            plt.plot(x,y,**kwargs)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            if title is not None:
                plt.title(title)
        else:
            ax.plot(x,y,linewidth=0.5,**kwargs)
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=6)
            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=6)
            if title is not None:
                ax.set_title(title, fontsize=6)

    def get_diff_vector(self,key_parameters1,key_parameters2=None):
        diff = []
        for key in self.keys:
            fit_weight = 1.0
            if key in self.fit_weight:
                fit_weight = self.fit_weight[key]
            
            # in parallel mode, the baseline doesn't exist, so just return the simulated case without subtracting the baseline
            if key not in key_parameters1:    
                parameter1 = key_parameters2[key]*0.0
            else:
                parameter1 = key_parameters1[key]
            if key_parameters2 is None or key not in key_parameters2:
                parameter2 = key_parameters1[key]*0.0
            else:
                parameter2 = key_parameters2[key]
            if isinstance(parameter1,numbers.Number):
                parameter1 = [parameter1]
                parameter2 = [parameter2]
            if isinstance(parameter1, list):
                parameter1 = np.array(parameter1)
                parameter2 = np.array(parameter2)
            has_nan = np.isnan(parameter1).any()
            if has_nan:
                print(type(self))
                print(self.key_parameters)
                print(self.simulated_data)
                assert(1==0)
            has_nan = np.isnan(parameter2).any()
            if has_nan:
                print(type(self))
                print(self.key_parameters)
                print(self.simulated_data)
                assert(1==0)
            diff_vector = np.array(parameter1-parameter2)
            diff.extend(list(diff_vector/self.unit_errors[key]*fit_weight))
        return np.array(diff)
    def set_simulation_baseline(self):
        self.simulated_key_parameters_baseline = self.simulated_key_parameters.copy()
    def get_baseline_vector(self):
        return self.get_diff_vector(self.simulated_key_parameters_baseline,None)
    def get_error_vector(self):
        return self.get_diff_vector(self.key_parameters,self.simulated_key_parameters)
    def get_differential_vector(self):
        return self.get_diff_vector(self.simulated_key_parameters,self.simulated_key_parameters_baseline)
    def __str__(self):
        return str(self.key_parameters)
    def write_file(self,filename,exp_or_sim="exp"):
        output = {"measurement_type": self.__class__.__name__, 
                  "measurement_condition": self.measurement_condition.copy(),
                  "fit_weight": self.fit_weight.copy(),
                  "tag": self.tag}
        if exp_or_sim=="exp" or exp_or_sim=="both":
            output["experimental_data"] = []
            for i, row in enumerate(self.data_rows):
                output["experimental_data"].append({"row_name":row, "data":self.measurement_data[i,:].copy()})
        if exp_or_sim=="sim" or exp_or_sim=="both":
            output["simulated_data"] = []
            for i, row in enumerate(self.data_rows):
                output["simulated_data"].append({"row_name":row, "data":self.simulated_data[i,:].copy()})
        output = convert_ndarrays_to_lists(output)
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
    def read_file(self,filename):      
        with open(filename, "r") as f:
            json_ = json.load(f)
        if "measurement_condition" in json_:
            self.measurement_condition = json_["measurement_condition"]
        if "experimental_data" in json_:
            self.measurement_data = []
            for row in json_["experimental_data"]:
                self.measurement_data.append(row["data"])
            self.measurement_data = np.array(self.measurement_data)
        if "simulated_data" in json_:
            self.simulated_data = []
            for row in json_["simulated_data"]:
                self.simulated_data.append(row["data"])
            self.simulated_data = np.array(self.simulated_data)
        if "fit_weight" in json_:
            self.fit_weight = json_["fit_weight"]
        if "tag" in json_:
            self.tag = json_["tag"]

def assign_measurements(self:CircuitGroup, measurements):
    for measurement in measurements:
        measurement.parent_device = self
    self.measurements = measurements
CircuitGroup.assign_measurements = assign_measurements

def set_simulation_baseline(measurements):
    for measurement in measurements:
        measurement.set_simulation_baseline()

def get_measurements_baseline_vector(measurements,measurement_class=None,include_tags=None,exclude_tags=["do_not_fit"]):
    vector = []
    for measurement in measurements:
        if measurement_class is None or isinstance(measurement,measurement_class):
            if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                vector.extend(measurement.get_baseline_vector())
    return vector
        
def get_measurements_error_vector(measurements,measurement_class=None,include_tags=None,exclude_tags=["do_not_fit"]):
    vector = []
    for measurement in measurements:
        if measurement_class is None or isinstance(measurement,measurement_class):
            if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                vector.extend(measurement.get_error_vector())
    return vector

def get_measurements_differential_vector(measurements,measurement_class=None,include_tags=None,exclude_tags=["do_not_fit"]):
    vector = []
    for measurement in measurements:
        if measurement_class is None or isinstance(measurement,measurement_class):
            if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                vector.extend(measurement.get_differential_vector())
    return vector

def get_measurements_groups(measurements,measurement_class=None,
                            include_tags=None,exclude_tags=None,categories=[],
                            optional_x_axis=None,plot_offset=False):
    exp_groups = {}
    sim_groups = {}
    x_axis_groups = {}
    for measurement in measurements:
        if measurement_class is None or isinstance(measurement,measurement_class):
            if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                conditions = []
                for sought_category in categories:
                    for category, condition in measurement.measurement_condition.items():
                        if category==sought_category:
                            conditions.append(condition)
                for key in measurement.keys:
                    tuple_ = key
                    if len(conditions)>0:
                        tuple_ = tuple([key]+conditions)
                    if tuple_ not in exp_groups:
                        exp_groups[tuple_] = []
                        sim_groups[tuple_] = []
                        x_axis_groups[tuple_] = []
                    offset = 0
                    if plot_offset and "plot_offset" in measurement.aux:
                        offset = measurement.aux["plot_offset"]
                    exp_ = measurement.key_parameters[key] + offset
                    sim_ = []
                    if key in measurement.simulated_key_parameters:
                        sim_ = measurement.simulated_key_parameters[key] + offset
                    if optional_x_axis is not None:
                        condition = measurement.measurement_condition[optional_x_axis]
                    if isinstance(exp_,numbers.Number):
                        exp_ = [exp_]
                    elif isinstance(exp_, np.ndarray):
                        exp_ = exp_.tolist()
                    if isinstance(sim_,numbers.Number):
                        sim_ = [sim_]
                    elif isinstance(sim_, np.ndarray):
                        sim_ = sim_.tolist()
                    exp_groups[tuple_].extend(exp_)
                    sim_groups[tuple_].extend(sim_)
                    if optional_x_axis is not None:
                        x_axis_groups[tuple_].extend([condition]*len(exp_))
    if optional_x_axis is not None:
        return exp_groups, sim_groups, x_axis_groups
    return exp_groups,sim_groups

def collate_device_measurements(devices,measurement_class=None,include_tags=None,exclude_tags=None):
    measurement_list = []
    if not isinstance(devices,list):
        devices = [devices]
    for device in devices:
        measurements = device.measurements
        for measurement in measurements:
            if measurement_class is None or isinstance(measurement,measurement_class):
                if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                    measurement_list.append(measurement)
    return measurement_list

def simulate_device_measurements(devices,measurement_class=None,include_tags=None,exclude_tags=None):
    for device in devices:
        measurements = device.measurements
        for measurement in measurements:
            if measurement_class is None or isinstance(measurement,measurement_class):
                if (include_tags==None or (measurement.tag is not None and measurement.tag in include_tags)) and (exclude_tags==None or (measurement.tag is None or measurement.tag not in exclude_tags)):
                    measurement.simulate(device)

# row 0 = voltage, row 1 = current
class IV_measurement(Measurement):
    keys = ["Voc", "Isc", "Pmax"]
    data_rows = ["Voltage(V)","Current(A)"]
    def __init__(self,Suns=None,IV_curve=None,temperature=25,measurement_cond_kwargs={},IL=None,JL=None,json_filepath=None,key_parameters=None,**kwargs):
        if json_filepath is not None:
            super().__init__(json_filepath=json_filepath)
        else:
            if IV_curve is not None:
                if isinstance(IV_curve, np.ndarray) and IV_curve.shape[0]>0 and IV_curve.shape[1]==2:
                    IV_curve = IV_curve.T
                # upside down
                if (IV_curve[0,0]-IV_curve[0,-1])*(IV_curve[1,0]-IV_curve[1,-1]) < 0:
                    IV_curve[1,:] *= -1
            if not hasattr(self,"measurement_condition"):
                self.measurement_condition = {}
            self.measurement_condition = {**self.measurement_condition, 
                                    **{'Suns':Suns,'IL':IL,'JL':JL,'temperature':temperature},
                                    **measurement_cond_kwargs}
            super().__init__(measurement_condition=self.measurement_condition,
                            measurement_data=IV_curve,key_parameters=key_parameters,**kwargs)
    @staticmethod
    def derive_key_parameters(data,key_parameters,conditions):
        key_parameters["Voc"] = get_Voc(data)
        key_parameters["Isc"] = get_Isc(data)
        key_parameters["Pmax"], _, _ = get_Pmax(data)
    def simulate(self,device=None):
        temperature = self.measurement_condition["temperature"]
        Suns = self.measurement_condition["Suns"]
        IL = self.measurement_condition["IL"]
        JL = self.measurement_condition["JL"]
        if device is None:
            device = self.parent_device
        device.set_temperature(temperature,rebuild_IV=False)
        if JL is not None:
            device.set_JL(JL,rebuild_IV=False)
            device.set_Suns(1.0)
        elif IL is not None:
            device.set_IL(IL,rebuild_IV=False)
            device.set_Suns(1.0)
        else:
            device.set_Suns(Suns)
        self.simulated_data = device.IV_table
        self.derive_key_parameters(self.simulated_data, self.simulated_key_parameters, self.measurement_condition)
    def plot_func(self,data,color="black",ax=None,title=None,kwargs={}):
        Measurement.plot_func(data[0,:],data[1,:],color=color,
                              xlabel="Voltage (V)",ylabel="Current (A)",title=title,
                              ax=ax,kwargs=kwargs)
        # if ax is None:
        #     _, Vmp, Imp = get_Pmax(data)
        #     plt.scatter(Vmp,Imp)

class Light_IV_measurement(IV_measurement):
    keys = ["Voc", "Isc", "Pmax"]
    data_rows = ["Voltage(V)","Current(A)"]

class Dark_IV_measurement(IV_measurement):
    keys = ["log_shunt_cond","I_bias"]
    data_rows = ["Voltage(V)","Current(A)"]
    def __init__(self,Suns=None,IV_curve=None,temperature=25,measurement_cond_kwargs={},IL=None,JL=None,json_filepath=None):
        if json_filepath is not None:
            super().__init__(json_filepath=json_filepath)
        else:
            if isinstance(IV_curve, np.ndarray) and IV_curve.shape[0]>0 and IV_curve.shape[1]==2:
                IV_curve = IV_curve.T
            # upside down
            if (IV_curve[0,0]-IV_curve[0,-1])*(IV_curve[1,0]-IV_curve[1,-1]) < 0:
                IV_curve[1,:] *= -1
            self.measurement_condition = {"base_point":np.min(IV_curve[0,:])}
            super().__init__(Suns=Suns,IV_curve=IV_curve,temperature=temperature,
                            measurement_cond_kwargs=measurement_cond_kwargs,IL=IL,JL=JL)
    @staticmethod
    def derive_key_parameters(data,key_parameters,conditions):
        Rshunt = Rshunt_extraction(data,base_point=conditions["base_point"])
        key_parameters["I_bias"] = interp_(conditions["base_point"],data[0,:],data[1,:])
        key_parameters["log_shunt_cond"] = np.log10(1/Rshunt)
    def plot_func(self,data,color="black",ax=None,title=None,kwargs={}):
        base_point = self.measurement_condition["base_point"]
        indices = np.where((data[0,:]>=base_point) & (data[0,:]<=base_point+0.2))[0]
        if len(indices) >= 2 and np.max(data[0,indices])-np.min(data[0,indices])>0.05:
            data_ = data[:,indices]
        else:
            x = [base_point,base_point+0.2]
            y = interp_(x,data[0,:],data[1,:])
            data_ = np.array([x,y])
        super().plot_func(data=data_,color=color,ax=ax,title=title,kwargs=kwargs)

# row 0 = voltage, row 1 onwards Suns or current
class Suns_Voc_measurement(Measurement):
    keys = ["Voc"]
    def __init__(self,Suns_Isc_Voc_curve=None,temperature=25,measurement_cond_kwargs={},json_filepath=None,**kwargs):
        if json_filepath is not None:
            super().__init__(json_filepath=json_filepath)
        else:
            super().__init__(measurement_condition={'temperature':temperature,
                                                    **measurement_cond_kwargs},
                            measurement_data=Suns_Isc_Voc_curve,**kwargs)
        self.data_rows = ["Voc(V)"]
        num_row = self.measurement_data.shape[0]
        num_subcells = int((num_row-1)/2)
        for i in range(num_subcells):
            self.data_rows.append(f"subcell {i} Suns")
        for i in range(num_subcells):
            self.data_rows.append(f"subcell {i} IL(A)")
    @staticmethod
    def derive_key_parameters(data,key_parameters,conditions):
        key_parameters["Voc"] = data[0,:]
    def simulate(self,device=None):
        num_row = self.measurement_data.shape[0]
        num_subcells = int((num_row-1)/2)
        Suns = self.measurement_data[1:num_subcells+1,:]
        Iscs = self.measurement_data[num_subcells+1:,:]
        if np.isnan(Suns[0,0]):
            Suns = None
        if np.isnan(Iscs[0,0]):
            Iscs = None
        if device is None:
            device = self.parent_device
        self.simulated_data, _ = simulate_Suns_Voc(device, Suns=Suns, Iscs=Iscs)
        self.derive_key_parameters(self.simulated_data, self.simulated_key_parameters, None)
    def plot_func(self,data,color="black",ax=None,title=None,kwargs=None):
        num_row = data.shape[0]
        num_subcells = int((num_row-1)/2)
        y_label = "log10(Suns)"
        ys = np.max(data[1:num_subcells+1,:],axis=0)
        if np.isnan(ys[0]):
            y_label = "log10(Current(A))"
            ys = np.max(data[num_subcells+1:,:],axis=0)
        ys = np.log10(ys)
        Measurement.plot_func(data[0,:],ys,color=color,
                              xlabel="Voc (V)",ylabel=y_label,title=title,
                              ax=ax,kwargs=kwargs)

def simulate_Suns_Voc(cell, Suns=None, Iscs=None):
    subcells_num = 1
    if isinstance(cell,MultiJunctionCell):
        subcells_num = len(cell.cells)
    if Suns is None and Iscs is None:
        Suns = 10.0**(np.arange(-3,1,0.1))
        Suns = Suns[:,None]
    if Iscs is not None:
        if isinstance(Iscs,numbers.Number):
            Iscs = Iscs*np.ones((subcells_num,1))
        if Iscs.ndim == 1:
            Iscs = Iscs[None,:]
        assert(Iscs.shape[0]==subcells_num)
        Suns = np.ones_like(Iscs)*np.nan
    else:
        if isinstance(Suns,numbers.Number):
            Suns = Suns*np.ones((subcells_num,1))
        if Suns.ndim == 1:
            Suns = Suns[None,:]
        assert(Suns.shape[0]==subcells_num)
        Iscs = np.ones_like(Suns)*np.nan
    Vocs = []
    cell.set_Suns(1.0, rebuild_IV=False)
    for i, _ in enumerate(Suns[0,:]):
        if not np.isnan(Suns[0,i]):
            if isinstance(cell,MultiJunctionCell):
                for j, cell_ in enumerate(cell.cells):
                    cell_.set_Suns(Suns[j,i])
                cell.build_IV()
            else:
                cell.set_Suns(Suns[0,i])
        else:
            if isinstance(cell,MultiJunctionCell):
                for j, cell_ in enumerate(cell.cells):
                    cell_.set_IL(Iscs[j,i], temperature=cell.temperature)
                cell.build_IV()
            else:
                cell.set_IL(Iscs[0,i], temperature=cell.temperature)
        Vocs.append(cell.get_Voc())
    Suns_Isc_Voc_curve = np.vstack([np.array(Vocs)[None,:],Suns,Iscs])
    if len(Vocs)==1:
        Vocs = Vocs[0]
    return Suns_Isc_Voc_curve, Vocs

def get_measurements(measurements_directory):
    # Read in all the measurements and collate them
    measurements = []
    json_files = [f for f in os.listdir(measurements_directory) if f.endswith('.json')]
    for filename in json_files:
        try:
            fullpath = os.path.join(measurements_directory, filename)
            with open(fullpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                measurement_type = data["measurement_type"]
                match measurement_type:
                    case "Suns_Voc_measurement":
                        measurements.append(Suns_Voc_measurement(json_filepath=fullpath))
                    case "Light_IV_measurement":
                        measurements.append(Light_IV_measurement(json_filepath=fullpath))
                    case "Dark_IV_measurement":
                        measurements.append(Dark_IV_measurement(json_filepath=fullpath))
        except Exception as e:
            # Log and keep going (or re-raise if you want to fail fast)
            print(f"ERROR reading {filename}: {e}")
    return measurements