''' Helper functions for handling folders in the GUI '''
import os
from tkinter import filedialog

from src.utils.datamanager import load_filtered_data, load_config, load_params, save_dataclass_to_csv

from src.gui_helpers.preprocessing_handling import open_preprocessing


def open_raw_data_folder(config_local, data):
    ''' Open a file dialog to select a data folder for raw data '''
    data_folder = filedialog.askdirectory()
    config_local.filetype = 'raw'
    if data_folder:
        config_local.file_path = data_folder
        run_names = set()

        for root, _, files in os.walk(data_folder):
            cor_files = [f for f in files if f.endswith('_cor.csv')]
            imu_files = [f for f in files if f.endswith('_imu.csv')]
            gen_files = [f for f in files if f.endswith('_gen.csv')]

            for cor_file in cor_files:
                run_name = cor_file.replace('_cor.csv', '')
                if (f'{run_name}_imu.csv' in imu_files and
                        f'{run_name}_gen.csv' in gen_files):
                    run_names.add(os.path.relpath(root, data_folder))
                    break
        config_local.run_names = run_names
        print(f"Data folder selected: {data_folder}")
        print(f"Runs found: {run_names}")

        open_preprocessing(config_local, data)


def open_filtered_data_folder(config_local, data):
    ''' Open a file dialog to select a data folder for filtered data '''
    data_folder = filedialog.askdirectory()
    config_local.filetype = 'filtered'
    if data_folder:
        config_local.file_path = data_folder
        run_names = set()

        for _, _, files in os.walk(data_folder):
            csv_files = [f for f in files if f.endswith('.csv')]

            for csv_file in csv_files:
                run_names.add(csv_file.replace('.csv', ''))

        config_local.run_names = run_names
        print(f"Data folder selected: {data_folder}")
        print(f"Runs found: {run_names}")

        data = load_filtered_data(config_local, data)


def open_config(config):
    ''' Open a file dialog to select a config file '''
    config_file_path = filedialog.askopenfilename(filetypes=[("Config files", "*.toml")])
    if config_file_path:
        print(f"Config file selected: {config_file_path}")
    config = load_config(config_file_path)


def open_vhl_params(config, vhl_params):
    ''' Open a file dialog to select a vehicle parameter file '''
    vhl_params_file_path = filedialog.askopenfilename(
        filetypes=[("Vehicle Parameter files", "*.toml")])
    if vhl_params_file_path:
        print(f"Vehicle parameter file selected: {vhl_params_file_path}")
    else:
        print("Load default parameters")
        vhl_params_file_path = config.vehicle_parameter_names
    vhl_params = load_params(vhl_params_file_path)


def open_storeage_folder(config):
    ''' Open a file dialog to select a config file '''
    selected_folder = filedialog.askdirectory()
    if selected_folder:
        config.output_folder_path = os.path.dirname(selected_folder)
        config.output_folder = os.path.basename(selected_folder)
        print("Selected Folder for storing data: ", config.output_folder_path + "/" + config.output_folder)


def save_parameters(tire_params_set_nelder, tire_params_set_svi, std_params_set_svi, output_folder_path, output_folder):
    save_dataclass_to_csv(tire_params_set_nelder, output_folder_path,
                          output_folder, 'tire_params_nelder.csv')
    save_dataclass_to_csv(tire_params_set_svi, output_folder_path, output_folder, 'tire_params_svi.csv')
    save_dataclass_to_csv(std_params_set_svi, output_folder_path, output_folder, 'std_params_svi.csv')
