''' Graphical User Interface for Tire Parameter Fitting '''
import os
import sys
import tkinter as tk
from tkinter import scrolledtext
from matplotlib import pyplot as plt

from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from data_types.vehiclestates import STMStates, STMForces
from data_types.sensordata import FilteredData

from src.utils.evaluation_helpers import plot_bell_curves, plot_tire_curves, plot_excitation_histograms
from src.utils.filter_data import filter_vhl_data
from src.utils.datamanager import load_config
from src.utils.datamanager import save_dataclass_to_csv, load_params, get_repository_path

from src.gui_helpers.frame_handling import ConsoleRedirect, create_image_canvas, update_grid, configure_root_grid
from src.gui_helpers.folder_handling import open_raw_data_folder, open_filtered_data_folder, open_config, open_vhl_params, open_storeage_folder, save_parameters
from src.gui_helpers.preprocessing_handling import open_preprocessing
from src.gui_helpers.equn_handling import select_vhl_model, param_fitting


def main():
    ''' Main function for the GUI '''
    root = tk.Tk()
    root.grid()
    root.title("Tire Parameter Fitting")
    root.config(bg='black')
    # configure base window
    width_screen = root.winfo_screenwidth()
    height_screen = root.winfo_screenheight()
    taskbar_height = 80
    window_height = height_screen - taskbar_height
    root.geometry(f"{width_screen}x{window_height}+0+0")
    # font configuration
    root.option_add("*Font", "Arial 11")  # set default font
    font_headline = ("Arial", 16, "bold")
    bald_font = ("Arial", 11, "bold")
    # create frames
    frames = {
        "frame_data_input": tk.Frame(root),
        "frame_param_fitting": tk.Frame(root),
        "frame_postprocess": tk.Frame(root),
        "frame_image": tk.Frame(root),
        "frame_console": tk.Frame(root, padx=10)
    }
    frames["frame_data_input"].grid(column=0, row=0, padx=(10, 10), pady=5, sticky="nsew")
    frames["frame_param_fitting"].grid(column=0, row=1, padx=(10, 10), pady=5, sticky="nsew")
    frames["frame_postprocess"].grid(column=0, row=2, padx=(10, 10), pady=5, sticky="nsew")
    frames["frame_image"].grid(column=1, row=0, rowspan=2, padx=(10, 10), pady=5, sticky="nsew")
    frames["frame_console"].grid(column=1, row=2, sticky="n")
    for frame in frames.values():
        frame.grid_propagate(True)
    configure_root_grid(root)
    # Bind dynamic Frame size
    root.bind("<Configure>", lambda event: update_grid(event, frames.values()))
    # Create default config and data objects
    config = load_config()
    data = FilteredData()
    vhl_params = load_params(config.vehicle_parameter_names)
    vhl_states = STMStates()
    vhl_forces = STMForces()
    tire_params_set_svi, std_params_set_svi, tire_params_set_nelder = STMTireParams(), STMTireParams(), STMTireParams()
    try:
        repo_path = get_repository_path()
        icon_path = repo_path + "/setup/icon.png"
        icon = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, icon)
    except Exception as e:
        print(f"Error loading icon: {e}")
    try:
        image_path = os.path.join(repo_path, "setup", "image.png")
        create_image_canvas(frames["frame_image"], image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
    # ----------------------------------------------------------------
    # Configure Console Output
    # ----------------------------------------------------------------
    console = scrolledtext.ScrolledText(
        frames["frame_console"], wrap='word', state='disable', width=180)
    console.grid(row=0, column=0, sticky="nsew", padx=10)
    sys.stdout = ConsoleRedirect(console)
    sys.stderr = ConsoleRedirect(console)
    # ----------------------------------------------------------------
    # Configure Data Input
    # ----------------------------------------------------------------
    tk.Label(frames["frame_data_input"], text="Configurate Data",
             font=font_headline).grid(row=0, column=0, sticky="w")
    tk.Label(frames["frame_data_input"], text="Load Configuration",
             font=bald_font).grid(row=1, column=0, sticky="w")
    tk.Button(frames["frame_data_input"], text='Load Configuration',
              command=lambda: open_config(config)).grid(row=1, column=1, sticky='w')

    tk.Label(frames["frame_data_input"], text="Load Data",
             font=bald_font).grid(row=2, column=0, sticky="w")
    tk.Button(frames["frame_data_input"], text="Choose real Data",
              command=lambda: open_raw_data_folder(config, data)).grid(row=2, column=1, sticky="w")
    tk.Button(frames["frame_data_input"], text='Choose Filtered Data',
              command=lambda: open_filtered_data_folder(config, data)).grid(row=2, column=2, sticky="w")

    tk.Label(frames["frame_data_input"], text="Preprocess Data",
             font=bald_font).grid(row=3, column=0, sticky="w")
    tk.Button(frames["frame_data_input"], text='Preprocess Data',
              command=lambda: open_preprocessing(config, data)).grid(row=3, column=1, sticky='w')

    tk.Label(frames["frame_data_input"], text="Load Vehicle Parameters",
             font=bald_font).grid(row=4, column=0, sticky="w")
    tk.Button(frames["frame_data_input"], text='Load Vehicle Parameters',
              command=lambda: open_vhl_params(config, vhl_params)).grid(row=4, column=1, sticky='w')

    tk.Label(frames["frame_data_input"], text="Print Configuration",
             font=bald_font).grid(row=6, column=0, sticky="w")
    tk.Button(frames["frame_data_input"], text='Print Config',
              command=lambda: print(config)).grid(row=6, column=1, sticky='w')
    tk.Button(frames["frame_data_input"], text='Print Run Names',
              command=lambda: print(config.run_names)).grid(row=7, column=1, sticky='w')
    tk.Button(frames["frame_data_input"], text='Print Vehicle Parameters',
              command=lambda: print(vhl_params)).grid(row=8, column=1, sticky='w')
    # ----------------------------------------------------------------
    # Configure Parameter Fitting
    # ----------------------------------------------------------------
    tk.Label(frames["frame_param_fitting"], text="Parameter Fitting",
             font=font_headline).grid(row=0, column=0, sticky="w")

    tk.Label(frames["frame_param_fitting"], text="Model Type",
             font=bald_font).grid(row=1, column=0, sticky="w")
    tk.Button(frames["frame_param_fitting"], text='STM Model',
              command=lambda: select_vhl_model("STM", data, vhl_params, vhl_states, vhl_forces)).grid(row=1, column=1, sticky='w')
    tk.Button(frames["frame_param_fitting"], text='LSD-STM',
              command=lambda: select_vhl_model("LSD-STM", data, vhl_params, vhl_states, vhl_forces)).grid(row=1, column=2, sticky='w')

    tk.Label(frames["frame_param_fitting"], text="Outlier Filtering",
             font=bald_font).grid(row=2, column=0, sticky="w")
    tk.Button(frames["frame_param_fitting"], text='Filter Outliers',
              command=lambda: filter_vhl_data(vhl_states, vhl_forces)).grid(row=2, column=1, sticky='w')

    def set_optimization_config(config_local):
        ''' Set the optimization configuration '''
        def save_values():
            ''' Save the values from the optimization configuration window '''
            config_local.nelder_options['maxiter'] = int(entry_nelder_maxiter.get())
            config_local.nelder_options['sample_points'] = int(entry_nelder_sample_points.get())
            config_local.svi_options['iter_svi_MFS'] = int(entry_svi_iter_svi_MFS.get())
            config_local.svi_options['step_MFS'] = float(entry_svi_step_MFS.get())
            config_local.svi_options['sample_points'] = int(entry_svi_sample_points.get())
            config_local.fit_flags_front_axle_x = {
                'B': var_fit_flags_front_axle_x_B.get(),
                'C': var_fit_flags_front_axle_x_C.get(),
                'D': var_fit_flags_front_axle_x_D.get(),
                'E': var_fit_flags_front_axle_x_E.get()
            }
            config_local.fit_flags_front_axle_y = {
                'B': var_fit_flags_front_axle_y_B.get(),
                'C': var_fit_flags_front_axle_y_C.get(),
                'D': var_fit_flags_front_axle_y_D.get(),
                'E': var_fit_flags_front_axle_y_E.get()
            }
            config_local.fit_flags_rear_axle_x = {
                'B': var_fit_flags_rear_axle_x_B.get(),
                'C': var_fit_flags_rear_axle_x_C.get(),
                'D': var_fit_flags_rear_axle_x_D.get(),
                'E': var_fit_flags_rear_axle_x_E.get()
            }
            config_local.fit_flags_rear_axle_y = {
                'B': var_fit_flags_rear_axle_y_B.get(),
                'C': var_fit_flags_rear_axle_y_C.get(),
                'D': var_fit_flags_rear_axle_y_D.get(),
                'E': var_fit_flags_rear_axle_y_E.get()
            }
            config_window.destroy()

        config_window = tk.Toplevel()
        config_window.title("Set Optimization Configuration")

        tk.Label(config_window, text="Nelder Settings",
                 font=bald_font).grid(row=0, column=0, sticky="w")
        tk.Label(config_window, text="nelder_options maxiter").grid(row=0, column=1)
        entry_nelder_maxiter = tk.Entry(config_window)
        entry_nelder_maxiter.grid(row=0, column=2)
        entry_nelder_maxiter.insert(0, config.nelder_options['maxiter'])

        tk.Label(config_window, text="nelder_options sample_points").grid(row=0, column=3)
        entry_nelder_sample_points = tk.Entry(config_window)
        entry_nelder_sample_points.grid(row=0, column=4)
        entry_nelder_sample_points.insert(0, config.nelder_options['sample_points'])

        tk.Label(config_window, text="SVI Settings",
                 font=bald_font).grid(row=1, column=0, sticky="w")

        tk.Label(config_window, text="svi_options iter_svi_MFS").grid(row=1, column=1)
        entry_svi_iter_svi_MFS = tk.Entry(config_window)
        entry_svi_iter_svi_MFS.grid(row=1, column=2)
        entry_svi_iter_svi_MFS.insert(0, config.svi_options['iter_svi_MFS'])

        tk.Label(config_window, text="svi_options step_MFS").grid(row=1, column=3)
        entry_svi_step_MFS = tk.Entry(config_window)
        entry_svi_step_MFS.grid(row=1, column=4)
        entry_svi_step_MFS.insert(0, config.svi_options['step_MFS'])

        tk.Label(config_window, text="svi_options sample_points").grid(row=1, column=5)
        entry_svi_sample_points = tk.Entry(config_window)
        entry_svi_sample_points.grid(row=1, column=6)
        entry_svi_sample_points.insert(0, config.svi_options['sample_points'])

        tk.Label(config_window, text="Fit Flags Front Axle Longitudinal",
                 font=bald_font).grid(row=3, column=0)
        var_fit_flags_front_axle_x_B = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_x_C = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_x_D = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_x_E = tk.BooleanVar(value=True)
        tk.Checkbutton(config_window, text="B",
                       variable=var_fit_flags_front_axle_x_B).grid(row=4, column=0)
        tk.Checkbutton(config_window, text="C",
                       variable=var_fit_flags_front_axle_x_C).grid(row=4, column=1)
        tk.Checkbutton(config_window, text="D",
                       variable=var_fit_flags_front_axle_x_D).grid(row=4, column=2)
        tk.Checkbutton(config_window, text="E",
                       variable=var_fit_flags_front_axle_x_E).grid(row=4, column=3)

        tk.Label(config_window, text="Fit Flags Front Axle Lateral",
                 font=bald_font).grid(row=5, column=0)
        var_fit_flags_front_axle_y_B = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_y_C = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_y_D = tk.BooleanVar(value=True)
        var_fit_flags_front_axle_y_E = tk.BooleanVar(value=True)
        tk.Checkbutton(config_window, text="B",
                       variable=var_fit_flags_front_axle_y_B).grid(row=6, column=0)
        tk.Checkbutton(config_window, text="C",
                       variable=var_fit_flags_front_axle_y_C).grid(row=6, column=1)
        tk.Checkbutton(config_window, text="D",
                       variable=var_fit_flags_front_axle_y_D).grid(row=6, column=2)
        tk.Checkbutton(config_window, text="E",
                       variable=var_fit_flags_front_axle_y_E).grid(row=6, column=3)

        tk.Label(config_window, text="Fit Flags Rear Axle Longitudinal",
                 font=bald_font).grid(row=7, column=0)
        var_fit_flags_rear_axle_x_B = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_x_C = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_x_D = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_x_E = tk.BooleanVar(value=True)
        tk.Checkbutton(config_window, text="B",
                       variable=var_fit_flags_rear_axle_x_B).grid(row=8, column=0)
        tk.Checkbutton(config_window, text="C",
                       variable=var_fit_flags_rear_axle_x_C).grid(row=8, column=1)
        tk.Checkbutton(config_window, text="D",
                       variable=var_fit_flags_rear_axle_x_D).grid(row=8, column=2)
        tk.Checkbutton(config_window, text="E",
                       variable=var_fit_flags_rear_axle_x_E).grid(row=8, column=3)

        tk.Label(config_window, text="Fit Flags Rear Axle Lateral",
                 font=bald_font).grid(row=9, column=0)
        var_fit_flags_rear_axle_y_B = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_y_C = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_y_D = tk.BooleanVar(value=True)
        var_fit_flags_rear_axle_y_E = tk.BooleanVar(value=True)
        tk.Checkbutton(config_window, text="B",
                       variable=var_fit_flags_rear_axle_y_B).grid(row=10, column=0)
        tk.Checkbutton(config_window, text="C",
                       variable=var_fit_flags_rear_axle_y_C).grid(row=10, column=1)
        tk.Checkbutton(config_window, text="D",
                       variable=var_fit_flags_rear_axle_y_D).grid(row=10, column=2)
        tk.Checkbutton(config_window, text="E",
                       variable=var_fit_flags_rear_axle_y_E).grid(row=10, column=3)

        tk.Label(config_window, text="Save Configuration",
                 font=bald_font).grid(row=11, column=0, sticky="w")
        tk.Button(config_window, text="Save", command=save_values).grid(row=11, column=1)

    tk.Label(frames["frame_param_fitting"], text="Configure Optimization",
             font=bald_font).grid(row=3, column=0, sticky="w")
    tk.Button(frames["frame_param_fitting"], text='Configure Optimization',
              command=lambda: set_optimization_config(config)).grid(row=3, column=1, sticky='w')

    def set_bounds_config(config_local):
        ''' Set the parameter bounds '''
        def save_values():
            config_local.params_min = {
                'B': float(entry_params_min_B.get()),
                'C': float(entry_params_min_C.get()),
                'D': float(entry_params_min_D.get()),
                'E': float(entry_params_min_E.get())
            }
            config_local.params_max = {
                'B': float(entry_params_max_B.get()),
                'C': float(entry_params_max_C.get()),
                'D': float(entry_params_max_D.get()),
                'E': float(entry_params_max_E.get())
            }
            config_local.params_init = {
                'B': float(entry_params_init_B.get()),
                'C': float(entry_params_init_C.get()),
                'D': float(entry_params_init_D.get()),
                'E': float(entry_params_init_E.get())
            }
            config_window.destroy()

        config_window = tk.Toplevel()
        config_window.title("Set Bounds Configuration")

        headline_min_values = tk.Label(
            config_window, text="Minimum Parameter Bounds", font=bald_font)
        headline_min_values.grid(row=0, column=0, sticky="w")

        tk.Label(config_window, text="params_min B").grid(row=1, column=0)
        entry_params_min_B = tk.Entry(config_window)
        entry_params_min_B.grid(row=2, column=0)
        entry_params_min_B.insert(0, "5.0")

        tk.Label(config_window, text="params_min C").grid(row=1, column=1)
        entry_params_min_C = tk.Entry(config_window)
        entry_params_min_C.grid(row=2, column=1)
        entry_params_min_C.insert(0, "1.0")

        tk.Label(config_window, text="params_min D").grid(row=1, column=2)
        entry_params_min_D = tk.Entry(config_window)
        entry_params_min_D.grid(row=2, column=2)
        entry_params_min_D.insert(0, "0.0")

        tk.Label(config_window, text="params_min E").grid(row=1, column=3)
        entry_params_min_E = tk.Entry(config_window)
        entry_params_min_E.grid(row=2, column=3)
        entry_params_min_E.insert(0, "-1.0")

        headline_max_values = tk.Label(
            config_window, text="Maximum Parameter Bounds", font=bald_font)
        headline_max_values.grid(row=3, column=0, sticky="w")

        tk.Label(config_window, text="params_max B").grid(row=4, column=0)
        entry_params_max_B = tk.Entry(config_window)
        entry_params_max_B.grid(row=5, column=0)
        entry_params_max_B.insert(0, "40.0")

        tk.Label(config_window, text="params_max C").grid(row=4, column=1)
        entry_params_max_C = tk.Entry(config_window)
        entry_params_max_C.grid(row=5, column=1)
        entry_params_max_C.insert(0, "3.0")

        tk.Label(config_window, text="params_max D").grid(row=4, column=2)
        entry_params_max_D = tk.Entry(config_window)
        entry_params_max_D.grid(row=5, column=2)
        entry_params_max_D.insert(0, "2.0")

        tk.Label(config_window, text="params_max E").grid(row=4, column=3)
        entry_params_max_E = tk.Entry(config_window)
        entry_params_max_E.grid(row=5, column=3)
        entry_params_max_E.insert(0, "1.0")

        headline_init_values = tk.Label(
            config_window, text="Initial Parameter Values", font=bald_font)
        headline_init_values.grid(row=6, column=0, sticky="w")

        tk.Label(config_window, text="params_init B").grid(row=7, column=0)
        entry_params_init_B = tk.Entry(config_window)
        entry_params_init_B.grid(row=8, column=0)
        entry_params_init_B.insert(0, "10.0")

        tk.Label(config_window, text="params_init C").grid(row=7, column=1)
        entry_params_init_C = tk.Entry(config_window)
        entry_params_init_C.grid(row=8, column=1)
        entry_params_init_C.insert(0, "1.65")

        tk.Label(config_window, text="params_init D").grid(row=7, column=2)
        entry_params_init_D = tk.Entry(config_window)
        entry_params_init_D.grid(row=8, column=2)
        entry_params_init_D.insert(0, "1.2")

        tk.Label(config_window, text="params_init E").grid(row=7, column=3)
        entry_params_init_E = tk.Entry(config_window)
        entry_params_init_E.grid(row=8, column=3)
        entry_params_init_E.insert(0, "1.0")

        tk.Button(config_window, text="Save", command=save_values).grid(row=9, column=0)

    headline_conf_bounds = tk.Label(
        frames["frame_param_fitting"], text="Configure Parameter Bounds", font=bald_font)
    headline_conf_bounds.grid(row=4, column=0, sticky="w")
    config_bounds = tk.Button(frames["frame_param_fitting"], text='Configure Bounds',
                              command=lambda: set_bounds_config(config))
    config_bounds.grid(row=4, column=1, sticky='w')

    headline_fit_model = tk.Label(frames["frame_param_fitting"],
                                  text="Fit Tire Model", font=bald_font)
    headline_fit_model.grid(row=6, column=0, sticky="w")
    fit_model = tk.Button(frames["frame_param_fitting"], text='Fit Model',
                          command=lambda: param_fitting(config, vhl_states, vhl_forces, tire_params_set_svi,
                                                        std_params_set_svi, tire_params_set_nelder))
    fit_model.grid(row=6, column=1, sticky='w')

    # ----------------------------------------------------------------
    # Postprocessing
    # ----------------------------------------------------------------
    headline_postprocess = tk.Label(
        frames["frame_postprocess"], text="Postprocess Data", font=font_headline)
    headline_postprocess.grid(row=0, column=0, sticky="w")
    headline_plot_results = tk.Label(
        frames["frame_postprocess"], text="Plot Results", font=bald_font)
    headline_plot_results.grid(row=1, column=0, sticky="w")

    def show_plot_tire_curves(vhl_states, vhl_forces, tire_params_set_svi, tire_params_set_nelder):
        plot_tire_curves(vhl_states, vhl_forces, tire_params_set_svi, tire_params_set_nelder)
        plt.show()

    def show_plot_bell_curves(tire_params_set_svi, std_params_set_svi, params_min, params_max):
        plot_bell_curves(tire_params_set_svi, std_params_set_svi, params_min, params_max)
        plt.show()

    def show_plot_excitation_histograms(vhl_states_local):
        plot_excitation_histograms(vhl_states_local)
        plt.show()

    plot_tire_curves_button = tk.Button(frames["frame_postprocess"], text='Plot Tire Curves',
                                        command=lambda: show_plot_tire_curves(vhl_states, vhl_forces,
                                                                              tire_params_set_svi, tire_params_set_nelder))
    plot_tire_curves_button.grid(row=1, column=1, sticky='w')
    plot_bell_curves_button = tk.Button(frames["frame_postprocess"], text='Plot Bell Curves',
                                        command=lambda: show_plot_bell_curves(tire_params_set_svi, std_params_set_svi,
                                                                              MFSimpleParams(
                                                                                  **config.params_min),
                                                                              MFSimpleParams(**config.params_max)))
    plot_bell_curves_button.grid(row=1, column=2, sticky='w')
    plot_excitation_histograms_button = tk.Button(frames["frame_postprocess"], text='Plot Excitation Histograms',
                                                  command=lambda: show_plot_excitation_histograms(vhl_states))
    plot_excitation_histograms_button.grid(row=1, column=3, sticky='w')

    headline_save_results = tk.Label(
        frames["frame_postprocess"], text="Save Results", font=bald_font)
    headline_save_results.grid(row=2, column=0, sticky="w")

    out_folder_button = tk.Button(frames["frame_postprocess"], text='Choose Output Folder',
                                  command=lambda: open_storeage_folder(config))
    out_folder_button.grid(row=2, column=1, sticky='w')

    tk.Button(frames["frame_postprocess"], text='Save Configuration',
              command=lambda: save_dataclass_to_csv(config, config.output_folder_path, config.output_folder, 'config.csv')).grid(row=2, column=2, sticky='w')

    tk.Button(frames["frame_postprocess"], text='Save Preprocessed Data',
              command=lambda: save_dataclass_to_csv(data, config.output_folder_path, config.output_folder, 'filtered_data.csv')).grid(row=3, column=1, sticky='w')

    tk.Button(frames["frame_postprocess"], text='Save Tire Parameters',
              command=lambda: save_parameters(tire_params_set_nelder, tire_params_set_svi,
                                              std_params_set_svi, config.output_folder_path, config.output_folder)).grid(row=3, column=2, sticky='w')

    tk.Label(frames["frame_postprocess"], text="Print Tire Parameters",
             font=bald_font).grid(row=4, column=0, sticky="w")
    tk.Button(frames["frame_postprocess"], text='Print SVI Tire Parameters',
              command=lambda: print(tire_params_set_svi)).grid(row=4, column=1, sticky='w')
    tk.Button(frames["frame_postprocess"], text='Print Std Parameters',
              command=lambda: print(std_params_set_svi)).grid(row=4, column=2, sticky='w')
    tk.Button(frames["frame_postprocess"], text='Print Nelder Tire Parameters',
              command=lambda: print(tire_params_set_nelder)).grid(row=5, column=1, sticky='w')
    root.mainloop()


if __name__ == '__main__':
    main()
