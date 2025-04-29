""" Data Manager to load configuration, sensordata and vehicle parameters """
import os
import csv
from dataclasses import fields, is_dataclass

from tqdm import tqdm
import toml
import pandas as pd
import jax.numpy as jnp


from data_types.config import Config
from data_types.sensordata import FilteredData, CorData, GenData, ImuData
from data_types.vehicleparameters import VhlParams


def get_repository_path():
    ''' Get the absolute path to the repository root '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return repo_path


def load_config(config_path: str = None):
    ''' Load the configuration from the TOML file at ../tireparameval/setup/config.toml '''
    try:
        if config_path is None:
            repo_path = get_repository_path()
            config_path = os.path.join(repo_path, 'setup', 'config.toml')
        with open(config_path, 'r') as fh:
            configfile = toml.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError('Config file does not exist!') from exc
    conf = Config(**configfile)
    return conf


def load_filtered_data(conf: Config, data: FilteredData = FilteredData()):
    ''' Load filtered data from a single file '''
    repo_path = get_repository_path()

    for run_name in tqdm(conf.run_names, desc="Loading filtered data"):
        if conf.file_path != '':
            data_path = os.path.join(conf.file_path, run_name + '.csv')
        else:
            data_path = os.path.join(repo_path, 'inputs', 'filtered', run_name + '.csv')
        signalnames = {}
        for field in fields(FilteredData):
            nested_fields = fields(field.type)
            for nested_field in nested_fields:
                signalnames[nested_field.name] = f"{field.name}/{nested_field.name}"
        if hasattr(data, 'gen_data'):
            data.cor_data = read_in_data(data_path, signalnames, data.cor_data)
            data.gen_data = read_in_data(data_path, signalnames, data.gen_data)
            data.imu_data = read_in_data(data_path, signalnames, data.imu_data)
        else:
            data.cor_data = read_in_data(data_path, signalnames, CorData)
            data.gen_data = read_in_data(data_path, signalnames, GenData)
            data.imu_data = read_in_data(data_path, signalnames, ImuData)
    return data


def read_in_data(data_file_path: str, signalnames: dict, data):
    ''' Read in data from a CSV file and set the attributes of the dataclass '''
    df = pd.read_csv(data_file_path)
    for key in data.__dataclass_fields__:
        try:
            # Attempt to set the attribute from the DataFrame using the signal names
            dat = jnp.array(getattr(data, key)) if hasattr(data, key) else jnp.array([])
            dat = jnp.concatenate((dat, jnp.array(df[signalnames[key]])))
            setattr(data, key, dat)
        except KeyError:
            # For other missing keys, set the value to zeros
            print(f"KeyError: '{key}' not found in signal names or DataFrame.")
            setattr(data, key, [0] * len(df))
    return data


def load_raw_data(conf: Config, run_name: str):
    ''' Load raw data from cor, gen and imu files '''
    repo_path = get_repository_path()
    signalnames_path = os.path.join(repo_path, 'setup', conf.signal_names + '.toml')
    try:
        with open(signalnames_path, 'r') as fh:
            signalnames = toml.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Signalname file does not exist or is of invalid type (only .toml supported)!'
        ) from None
    data_cor = CorData()
    data_gen = GenData()
    data_imu = ImuData()
    if conf.file_path != '':
        data_path = os.path.join(conf.file_path, run_name)
    else:
        data_path = os.path.join(repo_path, 'inputs', 'sensor_data', run_name)
    cor_file_path = os.path.join(data_path, run_name + '_cor.csv')
    gen_file_path = os.path.join(data_path, run_name + '_gen.csv')
    imu_file_path = os.path.join(data_path, run_name + '_imu.csv')
    data_cor = read_in_data(cor_file_path, signalnames, data_cor)
    data_gen = read_in_data(gen_file_path, signalnames, data_gen)
    data_imu = read_in_data(imu_file_path, signalnames, data_imu)
    return data_cor, data_gen, data_imu


def load_params(vehicle_parameter_names: str):
    ''' Load vehicle parameters from the TOML file at /setup/vehicle_parameter_names.toml '''
    try:
        if vehicle_parameter_names.endswith('.toml'):
            vehicle_parameter_names_path = vehicle_parameter_names
        else:
            repo_path = get_repository_path()
            vehicle_parameter_names_path = os.path.join(
                repo_path, 'setup', vehicle_parameter_names + '.toml')
        with open(vehicle_parameter_names_path, 'r') as fh:
            vhlparamsfile = toml.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError('Vehicle Parameter file does not exist!') from None
    vhlparams = VhlParams(**vhlparamsfile)
    return vhlparams


def flatten_dataclass(data):
    """Flatten a nested dataclass into a dictionary with dot-separated keys."""
    def _flatten(obj, parent_key='', sep='/'):
        items = []
        for field in fields(obj):
            value = getattr(obj, field.name)
            new_key = f"{parent_key}{sep}{field.name}" if parent_key else field.name
            if is_dataclass(value):
                items.extend(_flatten(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
    return _flatten(data)


def save_dataclass_to_csv(data, folderpath, foldername, filename):
    ''' Save a dataclass to a CSV file '''
    # Flatten the dataclass
    flat_data = flatten_dataclass(data)
    # Get the field names
    fieldnames = flat_data.keys()
    # Transpose the data to write each element in separate rows
    transposed_data = {key: list(value) if isinstance(value, jnp.ndarray) else [
        value] for key, value in flat_data.items()}
    max_length = max(len(v) for v in transposed_data.values())
    # Ensure all lists are of the same length
    for key in transposed_data:
        if len(transposed_data[key]) < max_length:
            transposed_data[key].extend([0.0] * (max_length - len(transposed_data[key])))
            print(f"Warning: {key} is shorter than the maximum length. Padding with zeros.")
    if folderpath == '':
        folderpath = get_repository_path()
    if foldername == '':
        file_path = os.path.join(folderpath, 'outputs', 'default', filename)
    else:
        file_path = os.path.join(folderpath, 'outputs', foldername, filename)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write the header
        writer.writeheader()
        # Write the data
        for i in range(max_length):
            row = {key: transposed_data[key][i] for key in fieldnames}
            writer.writerow(row)


def unflatten_dict(flat_dict):
    """Unflatten a dictionary with dot-separated keys into a nested dictionary."""
    unflattened = {}
    for key, value in flat_dict.items():
        parts = key.split('/')
        d = unflattened
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return unflattened


def dict_to_dataclass(data_dict, cls):
    """Convert a dictionary to a dataclass instance."""
    if not isinstance(data_dict, dict):
        raise ValueError(f"Expected a dictionary, got {type(data_dict)}")
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    field_values = {}
    for field in fields(cls):
        if field.name in data_dict:
            value = data_dict[field.name]
            field_type = field.type
            if is_dataclass(field_type):
                value = dict_to_dataclass(value, field_type)
            elif isinstance(value, dict):
                value = dict_to_dataclass(value, field_type)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                # Handle list of dataclasses
                sub_cls = field_type.__args__[0]
                if is_dataclass(sub_cls):
                    value = [dict_to_dataclass(item, sub_cls) for item in value]
            field_values[field.name] = value
    return cls(**field_values)


def load_dataclass_from_csv(cls, foldername, filename):
    ''' Load a dataclass from a CSV file '''
    repo_path = get_repository_path()
    file_path = os.path.join(repo_path, foldername, filename)
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # Initialize a dictionary to collect the data
    flat_data = {field: [] for field in reader.fieldnames}
    # Collect data from each row
    for row in rows:
        for key, value in row.items():
            if value == 'None':
                flat_data[key].append(0.0)
                print(f"Warning: {key} is None. Setting to 0.0.")
            else:
                try:
                    # Try to convert the value to a float
                    flat_data[key].append(float(value))
                except ValueError:
                    # If it fails, assume it's a list and convert to numpy array
                    flat_data[key].append(jnp.array(eval(value)))
    # Convert lists back to numpy arrays if needed
    for key, value in flat_data.items():
        if isinstance(value[0], (int, float, list)):
            flat_data[key] = jnp.array(value)
    unflattened_data = unflatten_dict(flat_data)
    return dict_to_dataclass(unflattened_data, cls)
