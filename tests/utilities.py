import os
from datetime import datetime
import re
import pickle
import json
import numpy as np
import time

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
        t1 = time.time()
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
        print(f"Finished in {time.time()-t1} seconds")
    return dict

def make_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def make_file_path_with_timestamp(prefix, extension):
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, prefix + "_" + make_timestamp() + extension)

def find_latest_pickle(prefix):
    directory = os.path.dirname(os.path.abspath(__file__))

    # Regex to match files like prefix_YYYY-MM-DD_HHMMSS.pkl
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{6}})\.pkl")

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
    match mode:
        case "record":
            with open(make_file_path_with_timestamp(this_file_prefix+"_result", ".pkl"), "wb") as f:
                pickle.dump(this_time_dict, f)
        case "test":
            filepath = find_latest_pickle(this_file_prefix+"_result")
            with open(filepath, "rb") as f:
                last_time_dict = pickle.load(f)
                all_pass = compare_nested_dicts(last_time_dict, this_time_dict, pytest_mode=pytest_mode)
                if all_pass:
                    print(this_file_prefix + " all pass!")