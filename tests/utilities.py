import os
from datetime import datetime
import re
import pickle
import json
import numpy as np
import time
from PV_Circuit_Model.utilities import Artifact, ParameterSet
from PV_Circuit_Model.circuit_model import CircuitComponent
from PV_Circuit_Model.cell_analysis import get_Pmax

def get_mode():
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, "config.json"), "r") as f:
        config = json.load(f)
        return config.get("mode")
    
def get_fields(device, prefix=None):
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, "config.json"), "r") as f:
        config = json.load(f)
        dict = {}
        test_attributes = config.get("test_attributes")
        keys = ["common", prefix]
        device.null_all_IV()
        t1 = time.perf_counter()
        device.build_IV()
        for key in keys:
            if key is not None and key in test_attributes:
                for attribute in test_attributes[key]:
                    attribute_name = attribute["name"]
                    if hasattr(device,attribute_name):
                        attr = getattr(device,attribute_name)
                        if "atol" not in attribute:
                            attribute["atol"] = 1e-5
                        if "rtol" not in attribute:
                            attribute["rtol"] = 1e-5
                        dict[attribute_name] = {"atol": attribute["atol"], "rtol": attribute["rtol"]}
                        if callable(attr):
                            dict[attribute_name]["value"] = attr()
                        else:
                            dict[attribute_name]["value"] = attr
        print(f"Finished in {time.perf_counter()-t1} seconds")
    return dict

def make_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def make_file_path_with_timestamp(prefix, extension):
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, prefix + "_" + make_timestamp() + "." + extension)

def find_latest_file(prefix,extension="pkl"):
    directory = os.path.dirname(os.path.abspath(__file__))

    # Regex to match files like prefix_YYYY-MM-DD_HHMMSS.pkl
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{6}})\.{extension}")

    latest_file = None
    latest_time = None

    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match:
            timestamp_str = match.group(1)
            try:
                file_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
                    latest_file = filename
            except ValueError:
                continue  # Skip files with invalid timestamp format

    return os.path.join(directory, latest_file)

def compare_nested_dicts(dict1, dict2, path="", pytest_mode=False):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    all_keys = keys1.union(keys2)
    all_pass = True

    for key in all_keys:
        full_path = f"{path}.{key}" if path else key

        if key not in dict1:
            print(f"{full_path} only in dict2")
            continue
        if key not in dict2:
            print(f"{full_path} only in dict1")
            continue

        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict) and "value" in val1 and "value" in val2:
            atol = 1e-5
            rtol = 1e-5
            if "atol" in val2:
                atol = val2["atol"]
            if "rtol" in val2:
                rtol = val2["rtol"]
            val1 = val1["value"]
            val2 = val2["value"]
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.allclose(val1, val2, rtol=rtol, atol=atol):
                    print(f"Difference at {full_path}: arrays not equal")
                    all_pass = False
                if pytest_mode:
                    assert(np.allclose(val1, val2, rtol=rtol, atol=atol))
            elif isinstance(val1, (float, int)) and isinstance(val2, (float, int)):
                if isinstance(val1, float) or isinstance(val2, float):
                    if not np.isclose(val1, val2, rtol=rtol, atol=atol):
                        print(f"Difference at {full_path}: {val1} (old) != {val2} (new)")
                        all_pass = False
                    if pytest_mode:
                        assert(np.isclose(val1, val2, rtol=rtol, atol=atol))
                else:
                    if val1 != val2:
                        print(f"Difference at {full_path}: {val1} (old) != {val2} (new)")
                        all_pass = False
                    if pytest_mode:
                        assert(val1==val2)
        else:
            compare_nested_dicts(val1, val2, path=full_path)
    
    return all_pass

def run_record_or_test(device, this_file_prefix=None, pytest_mode=False):
    mode = get_mode()
    this_time_dict = get_fields(device, prefix=this_file_prefix)
    this_time_solver_env_variables = ParameterSet.get_set("solver_env_variables")()
    total_set = {"solver_env_variables": this_time_solver_env_variables, "results": this_time_dict}
    match mode:
        case "record":
            with open(make_file_path_with_timestamp(this_file_prefix+"_result", "pkl"), "wb") as f:
                pickle.dump(total_set, f)
        case "test":
            filepath = find_latest_file(this_file_prefix+"_result")
            with open(filepath, "rb") as f:
                last_time_set = pickle.load(f)
                last_time_dict = last_time_set["results"]
                last_time_solver_env_variables = last_time_set["solver_env_variables"]
                if this_time_solver_env_variables != last_time_solver_env_variables: 
                    for key, value in this_time_solver_env_variables.items():
                        if key in last_time_solver_env_variables:
                            value2 = last_time_solver_env_variables[key]
                            if value != value2:
                                print(f"WARNING: The solver env variable {key} is different between last time ({value2}) and this time ({value})")
                all_pass = compare_nested_dicts(last_time_dict, this_time_dict, pytest_mode=pytest_mode)
                if all_pass:
                    print(this_file_prefix + " all pass!")

def compare_devices(device1,device2,lineage=[],pytest_mode=False):
    if pytest_mode:
        assert(device1==device2)
    if device1!=device2:
        lineage_word = ""
        for num in lineage:
            lineage_word += f".{num}"
        print("-------------------------------")
        print(f"device1{lineage_word} != device2{lineage_word} in the following:")
        for field in device1._critical_fields:
            if pytest_mode:
                assert(hasattr(device1,field))
                assert(hasattr(device2,field))
            if not hasattr(device1,field):
                print(f"device1{lineage_word} is missing field {field}")
            if not hasattr(device2,field):
                print(f"device2{lineage_word} is missing field {field}")
            if hasattr(device1,field) and hasattr(device2,field):
                device1_field = getattr(device1,field)
                device2_field = getattr(device2,field)
                if pytest_mode:
                    assert(device1_field == device2_field)
                if device1_field != device2_field:
                    if isinstance(device1_field,list):
                        if len(device1_field) != len(device2_field):
                            print(f"{field} has different lengths in device1{lineage_word} ({len(device1_field)}) vs device2{lineage_word} ({len(device2_field)})")
                        else:
                            for i in range(len(device1_field)):
                                item1 = device1_field[i]
                                item2 = device2_field[i]
                                if pytest_mode:
                                    assert(item1 == item2)
                                if item1 != item2:
                                    if isinstance(item1,CircuitComponent):
                                        lineage_ = lineage + [i]
                                        compare_devices(item1,item2,lineage=lineage_)
                                    else:
                                        print(f"{field}[{i}] is different in device1{lineage_word} ({item1}) vs device2{lineage_word} ({item2})")
                    else:
                        print(f"{field} is different in device1{lineage_word} ({device1_field}) vs device2{lineage_word} ({device2_field})")

def record_or_compare_artifact(device, this_file_prefix=None,pytest_mode=False):
    mode = get_mode()
    match mode:
        case "record":
            device.dump(make_file_path_with_timestamp(this_file_prefix+"_result", "bson"))
            return device
        case "test":
            filepath = find_latest_file(this_file_prefix+"_result","bson")
            device2 = Artifact.load(filepath)
            if device==device2:
                print(this_file_prefix + " artifact matches!")
            else:
                compare_devices(device,device2,pytest_mode=pytest_mode)
                print("\nLoaded the saved device for subsequent testing")
            return device2
        
def get_LT_spice_IV(folder, device_name):
    LT_spice_IV = np.empty((0,2))
    for i in range(1,5):
        filename = f"{folder}/{device_name}_scan{i}.txt"
        if os.path.exists(filename):
            LT_spice_IV = np.concatenate((LT_spice_IV,np.loadtxt(filename,skiprows=1)))
    LT_spice_IV[:,1] *= -1
    LT_spice_IV = LT_spice_IV.T
    return LT_spice_IV

def compare_artifact_against_LT_spice(device, LT_spice_IV, pytest_mode=False):
    mode = get_mode()
    if mode != "test":
        return None, None, None
    Pmax1 = get_Pmax(device.IV_table)
    Pmax2 = get_Pmax(LT_spice_IV)
    all_pass = True
    rtol = 1e-5
    atol = 1e-5
    if not np.isclose(Pmax1, Pmax2, rtol=rtol, atol=atol):
        print(f"Difference at Pmax: {Pmax2} (LT spice) != {Pmax1} (PV Circuit Model)")
        all_pass = False
        if pytest_mode:
            assert(np.isclose(Pmax1, Pmax2, rtol=rtol, atol=atol))

    rtol = 1e-4
    atol = 1e-4
    for i in range(LT_spice_IV.shape[1]):
        V = LT_spice_IV[0,i]
        I = LT_spice_IV[1,i]
        I_interp = np.interp(V,device.IV_V,device.IV_I)
        V_interp = np.interp(I,device.IV_I,device.IV_V)
        if not np.isclose(V_interp, V, rtol=rtol, atol=atol) and not np.isclose(I_interp, I, rtol=rtol, atol=atol):
            print(f"Difference in IV curves at V = {V}:  I={I} (LT spice) != {I_interp} (PV Circuit Model)")
            all_pass = False
            if pytest_mode:
                assert(np.isclose(V_interp, V, rtol=rtol, atol=atol) or np.isclose(I_interp, I, rtol=rtol, atol=atol))
    
    return all_pass, Pmax1, Pmax2



                                