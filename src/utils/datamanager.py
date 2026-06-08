"""Data loading and dataclass CSV utilities.

The library does not own default input, output, or parameter folders. Callers
must pass explicit paths from the integrating repository or application.
"""

from __future__ import annotations

import csv
import os
from dataclasses import fields, is_dataclass
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import toml
from tqdm import tqdm

from data_types.config import Config
from data_types.sensordata import CorData, FilteredData, GenData, ImuData
from data_types.vehicleparameters import VhlParams


def get_repository_path():
    """Return the absolute path to the library root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def _require_path(path: str | os.PathLike, label: str) -> Path:
    if path is None or str(path) == "":
        raise ValueError(f"{label} must be provided explicitly")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_config(config_path: str | os.PathLike):
    """Load an explicit TOML configuration file."""
    with _require_path(config_path, "Config file").open("r") as fh:
        configfile = toml.load(fh)
    return Config(**configfile)


def load_filtered_data(conf: Config, data: FilteredData = FilteredData()):
    """Load filtered data CSV files from conf.file_path."""
    input_root = _require_path(conf.file_path, "Filtered data directory")

    for run_name in tqdm(conf.run_names, desc="Loading filtered data"):
        data_path = input_root / f"{run_name}.csv"
        signalnames = {}
        for field in fields(FilteredData):
            nested_fields = fields(field.type)
            for nested_field in nested_fields:
                signalnames[nested_field.name] = f"{field.name}/{nested_field.name}"
        if hasattr(data, "gen_data"):
            data.cor_data = read_in_data(data_path, signalnames, data.cor_data)
            data.gen_data = read_in_data(data_path, signalnames, data.gen_data)
            data.imu_data = read_in_data(data_path, signalnames, data.imu_data)
        else:
            data.cor_data = read_in_data(data_path, signalnames, CorData)
            data.gen_data = read_in_data(data_path, signalnames, GenData)
            data.imu_data = read_in_data(data_path, signalnames, ImuData)
    return data


def read_in_data(data_file_path: str | os.PathLike, signalnames: dict, data):
    """Read a CSV file and set matching dataclass attributes."""
    df = pd.read_csv(data_file_path)
    for key in data.__dataclass_fields__:
        try:
            dat = jnp.array(getattr(data, key)) if hasattr(data, key) else jnp.array([])
            dat = jnp.concatenate((dat, jnp.array(df[signalnames[key]])))
            setattr(data, key, dat)
        except KeyError:
            print(f"KeyError: '{key}' not found in signal names or DataFrame.")
            setattr(data, key, [0] * len(df))
    return data


def load_raw_data(conf: Config, run_name: str):
    """Load raw cor, gen, and imu CSV files from conf.file_path."""
    signalnames_path = _require_path(conf.signal_names, "Signal names file")
    with signalnames_path.open("r") as fh:
        signalnames = toml.load(fh)

    data_root = _require_path(conf.file_path, "Raw data directory") / run_name
    cor_file_path = data_root / f"{run_name}_cor.csv"
    gen_file_path = data_root / f"{run_name}_gen.csv"
    imu_file_path = data_root / f"{run_name}_imu.csv"

    data_cor = read_in_data(cor_file_path, signalnames, CorData())
    data_gen = read_in_data(gen_file_path, signalnames, GenData())
    data_imu = read_in_data(imu_file_path, signalnames, ImuData())
    return data_cor, data_gen, data_imu


def load_params(vehicle_parameter_names: str | os.PathLike):
    """Load an explicit vehicle-parameter TOML file."""
    with _require_path(vehicle_parameter_names, "Vehicle parameter file").open("r") as fh:
        vhlparamsfile = toml.load(fh)
    return VhlParams(**vhlparamsfile)


def flatten_dataclass(data):
    """Flatten a nested dataclass into a dictionary with slash-separated keys."""

    def _flatten(obj, parent_key="", sep="/"):
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
    """Save a dataclass to folderpath/foldername/filename."""
    output_root = Path(folderpath).expanduser().resolve()
    if str(folderpath) == "":
        raise ValueError("Output folder path must be provided explicitly")

    flat_data = flatten_dataclass(data)
    fieldnames = flat_data.keys()
    transposed_data = {
        key: list(value) if isinstance(value, jnp.ndarray) else [value]
        for key, value in flat_data.items()
    }
    max_length = max(len(v) for v in transposed_data.values())
    for key in transposed_data:
        if len(transposed_data[key]) < max_length:
            transposed_data[key].extend([0.0] * (max_length - len(transposed_data[key])))
            print(f"Warning: {key} is shorter than the maximum length. Padding with zeros.")

    directory = output_root / foldername if foldername else output_root
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename

    with file_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(max_length):
            row = {key: transposed_data[key][i] for key in fieldnames}
            writer.writerow(row)


def unflatten_dict(flat_dict):
    """Unflatten a dictionary with slash-separated keys into a nested dictionary."""
    unflattened = {}
    for key, value in flat_dict.items():
        parts = key.split("/")
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
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                sub_cls = field_type.__args__[0]
                if is_dataclass(sub_cls):
                    value = [dict_to_dataclass(item, sub_cls) for item in value]
            field_values[field.name] = value
    return cls(**field_values)


def load_dataclass_from_csv(cls, foldername, filename):
    """Load a dataclass from foldername/filename."""
    file_path = _require_path(Path(foldername) / filename, "Dataclass CSV file")
    with file_path.open("r") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    flat_data = {field: [] for field in reader.fieldnames}
    for row in rows:
        for key, value in row.items():
            if value == "None":
                flat_data[key].append(0.0)
                print(f"Warning: {key} is None. Setting to 0.0.")
            else:
                try:
                    flat_data[key].append(float(value))
                except ValueError:
                    flat_data[key].append(jnp.array(eval(value)))
    for key, value in flat_data.items():
        if isinstance(value[0], (int, float, list)):
            flat_data[key] = jnp.array(value)
    unflattened_data = unflatten_dict(flat_data)
    return dict_to_dataclass(unflattened_data, cls)
