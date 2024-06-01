#import GPUtil
import torch
import os
import json
import pickle
import sys
from datetime import datetime
import time
import re
from pathlib import Path

def find_nested_field(json_dict, field_key):
    """
    Recursively search for a field in a dictionary and its nested dictionaries.
    """
    if field_key in json_dict:
        return json_dict[field_key]
    for key, value in json_dict.items():
        if isinstance(value, dict):
            result = find_nested_field(value, field_key)
            if result:
                return result
    return None  

def load_experiment_objects(experiment_jsons, file_path_key):
    """
    Load objects from pickle files specified in experiment JSONs, supporting nested file path keys.
    
    :param experiment_jsons: List of experiment JSON objects.
    :param file_path_key: The key in the JSON object that contains the path to the pickle file, potentially nested.
    :return: A list of loaded objects.
    """
    loaded_objects = []
    total_size_bytes = 0

    for experiment_json in experiment_jsons:
        file_path = find_nested_field(experiment_json, file_path_key)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
                loaded_objects.append(obj)
                obj_size = sys.getsizeof(obj)
                total_size_bytes += obj_size
                print(f"Loaded object from {file_path} ({obj_size} bytes)")
        else:
            print(f"File path not found or file does not exist for key '{file_path_key}'.")

    print(f"Total objects loaded: {len(loaded_objects)}. Total memory usage: {total_size_bytes} bytes ({total_size_bytes / (1024**2):.2f} MB).")
    
    return loaded_objects

def load_experiment_jsons(root_folder_path, dataset_name="", experiment_details_subfolder="experiment_details"):
    """
    Load JSONs from specified experiment details subfolder for a given dataset.
    
    :param root_folder_path: The base path where the datasets and experiment details are stored.
    :param dataset_name: The name of the dataset for which to load the experiment details.
    :param experiment_details_subfolder: The subfolder within the dataset folder where experiment details JSONs are stored.
    :return: A list of dictionaries, each representing the contents of an experiment JSON file.
    """
    if dataset_name == "":
        experiment_details_folder_path = os.path.join(root_folder_path, experiment_details_subfolder)
    else:
        experiment_details_folder_path = os.path.join(root_folder_path, experiment_details_subfolder, dataset_name)
    
    if Path(experiment_details_folder_path).exists():
        json_file_paths = [os.path.join(experiment_details_folder_path, file) for file in os.listdir(experiment_details_folder_path) if file.endswith('.json')]
    
        experiment_jsons = []
        for json_path in json_file_paths:
            with open(json_path, 'r') as file:
                experiment_details = json.load(file)
                experiment_jsons.append(experiment_details)

        if not experiment_jsons:
            print("No experiment details files found.")

        return experiment_jsons
    else:
        print(f"Folder: {experiment_details_folder_path} does not exist, returning empty list")
        return []
    
def get_newest_json(json_dicts, timestamp_key="timestamp"):
    """
    Given a list of JSON dictionaries and a timestamp key, return the newest JSON based on the timestamp.
    This function now searches for the timestamp key even in nested dictionaries.
    """
    valid_jsons = []
    for json_dict in json_dicts:
        timestamp = find_nested_field(json_dict, timestamp_key)
        if timestamp:
            valid_jsons.append((json_dict, parse_timestamp(timestamp)))
    
    if valid_jsons:
        # Sort valid_jsons by their parsed timestamp, which is the second item in each tuple
        valid_jsons.sort(key=lambda x: x[1], reverse=True)
        # Return the JSON part of the first tuple (newest JSON)
        return valid_jsons[0][0]
    
    print("No valid timestamps found in JSONs.")
    return None    
    
def get_unique_dictionaries(dict_list, return_strings = False):
    """
    Takes a list of dictionaries and returns a new list with only unique dictionaries,
    where uniqueness is determined by the dictionaries having the same fields and values.
    
    :param dict_list: A list of dictionaries or JSON strings.
    :return: A list containing only unique dictionaries.
    """
    unique_dict_strings = set()  # Use a set to store unique dictionary strings
    unique_dicts = []  # List to store the unique dictionaries
    
    for item in dict_list:
        # Convert item to a dictionary if it's a JSON string
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                print("Invalid JSON string:", item)
                continue  # Skip invalid JSON strings
        
        # Use json.dumps to get a consistent string representation of the dictionary
        # sorted_keys ensures the order of keys doesn't affect uniqueness
        dict_string = json.dumps(item, sort_keys=True)
        
        if dict_string not in unique_dict_strings:
            unique_dict_strings.add(dict_string)
            unique_dicts.append(item)
            
    if return_strings:
        return list(unique_dict_strings)
    else:
        return unique_dicts

def get_current_time_and_unix_timestamp(format_str = "%Y%m%d_%H%M%S_UTC"):
    """
    Returns the current time as both a Unix timestamp and a human-readable string safe for filenames.
    
    :return: A tuple containing the Unix timestamp and the formatted, filename-safe string.
    """
    unix_timestamp = int(time.time())
    # Use a filename-safe format for the readable timestamp
    readable_timestamp = datetime.utcfromtimestamp(unix_timestamp).strftime(format_str)
    
    return unix_timestamp, readable_timestamp

def parse_timestamp(timestamp_str, format_str = "%Y%m%d_%H%M%S_UTC"):
    # Convert the timestamp string to a datetime object
    return datetime.strptime(timestamp_str, format_str)

def select_gpu_with_most_free_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device
#    GPUs = GPUtil.getGPUs()
#    if not GPUs:
#        print("No GPU found. Using CPU.")
#        return torch.device('cpu')
    
    # Find the GPU with the maximum free memory
#    max_memory_gpu = max(GPUs, key=lambda gpu: gpu.memoryFree)
#    print(f"Selecting GPU {max_memory_gpu.id} with {max_memory_gpu.memoryFree}MB free memory")
#    return torch.device(f'cuda:{max_memory_gpu.id}')

def save_combination_list(combination_list, root_folder_path, filename="processed_combinations.json"):
    """
    Saves a list of JSON objects to a file.
    
    :param combination_list: A list of dictionaries (JSON objects) to be saved.
    :param root_folder_path: The root folder path where the file will be saved.
    :param filename: The name of the file to save the JSON objects in. Defaults to 'processed_combinations.json'.
    """
    # Ensure the root folder exists
    if not os.path.exists(root_folder_path):
        print(f"Folder path {root_folder_path} doesn't exist")
        return
    
    file_path = os.path.join(root_folder_path, filename)
    
    with open(file_path, 'w') as json_file:
        json.dump(combination_list, json_file, indent=4)
    
    print(f"Combinations list saved to {file_path}")