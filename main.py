""" Skript based tire parameter fitting """
# Configure the fitting process in setup/config.toml
import jax.numpy as jnp
import matplotlib.pyplot as plt

from data_types.vehicleparameters import STMTireParams, MFSimpleParams

from src.param_fitting.param_fitting import tire_param_fitting, calc_force_shift
from src.utils.calc_vhl_states import calc_vhl_forces, calc_vhl_states
from src.utils.datamanager import load_config, load_filtered_data, load_params, save_dataclass_to_csv, load_dataclass_from_csv
from src.utils.evaluation_helpers import plot_bell_curves, plot_tire_curves, eval_force_errors
from src.utils.filter_data import preprocess_data, filter_vhl_data

FIT_TARGETS = [
    ('wheel_fl', 'x'),
    ('wheel_fr', 'x'),
    ('wheel_rl', 'x'),
    ('wheel_rr', 'x'),
    ('front_axle', 'y'),
    ('rear_axle', 'y'),
]


def main():
    """
    Main function to for tire parameter fitting
    Configuration is loaded from setup/config.toml
    """
    # Load config
    conf = load_config()
    params_min = MFSimpleParams(**conf.params_min)
    params_max = MFSimpleParams(**conf.params_max)
    params_init = MFSimpleParams(**conf.params_init)
    # Load and preprocess data
    if (conf.filetype == 'filtered'):
        sensordata = load_filtered_data(conf)
    else:
        sensordata = preprocess_data(conf)
        if conf.save_filtered_data:
            save_dataclass_to_csv(sensordata, conf.output_folder_path,
                                  conf.output_folder, 'filtered_data.csv')
    # Load vehicle parameters
    vhl_params = load_params(conf.vehicle_parameter_names)
    # Calculate tire states and forces
    vhl_states = calc_vhl_states(sensordata, vhl_params)
    vhl_forces = calc_vhl_forces(conf.model, sensordata, vhl_states, vhl_params)
    # Filter states and forces
    if conf.vhl_data_filter:
        vhl_states, vhl_forces = filter_vhl_data(vhl_states, vhl_forces)
    # Fit tire parameters
    tire_params_set_svi = STMTireParams()
    std_params_set_svi = STMTireParams()
    tire_params_set_nelder = STMTireParams()
    if conf.mode == 'fitting':
        for state_key, direction in FIT_TARGETS:
            sigma = jnp.array(getattr(getattr(vhl_states, state_key), 'sigma_' + direction))
            force_n = jnp.array(getattr(getattr(vhl_forces, state_key), 'force_' + direction + '_n'))
            load_n = jnp.array(getattr(vhl_forces, state_key).force_z_n)
            fit_flags = getattr(conf, f'fit_flags_{state_key}_{direction}')
            params_init.S_V, params_init.S_H = calc_force_shift(sigma, force_n)
            print(f'Fitting - {state_key} {direction}')
            params_svi, std_svi, _ = tire_param_fitting(
                'SVI', 'MFSimple', sigma, force_n, load_n, params_init, params_min, params_max, conf.svi_options, fit_flags)
            params_nelder, _, _ = tire_param_fitting(
                'Nelder', 'MFSimple', sigma, force_n, load_n,
                params_init, params_min, params_max, conf.nelder_options, fit_flags)
            print('SVI:')
            print(params_svi)
            print(std_svi)
            print('Nelder:')
            print(params_nelder)
            setattr(tire_params_set_svi, f'{state_key}_{direction}', params_svi)
            setattr(std_params_set_svi, f'{state_key}_{direction}', std_svi)
            setattr(tire_params_set_nelder,
                    f'{state_key}_{direction}', params_nelder)
        if conf.enable_logging:
            save_dataclass_to_csv(tire_params_set_svi, conf.output_folder_path,
                                  conf.output_folder, 'tire_params_svi.csv')
            save_dataclass_to_csv(std_params_set_svi, conf.output_folder_path,
                                  conf.output_folder, 'std_params_svi.csv')
            save_dataclass_to_csv(tire_params_set_nelder, conf.output_folder_path,
                                  conf.output_folder, 'tire_params_nelder.csv')
    else:
        tire_params_set_svi = load_dataclass_from_csv(
            STMTireParams, 'outputs/' + conf.output_folder, 'tire_params_svi.csv')
        std_params_set_svi = load_dataclass_from_csv(
            STMTireParams, 'outputs/' + conf.output_folder, 'std_params_svi.csv')
        tire_params_set_nelder = load_dataclass_from_csv(
            STMTireParams, 'outputs/' + conf.output_folder, 'tire_params_nelder.csv')
    print('-' * 80)
    print('-' * 80)
    print('SVI Force Errors')
    eval_force_errors(vhl_states, vhl_forces, tire_params_set_svi)
    print('-' * 80)
    print('Nelder Force Errors')
    eval_force_errors(vhl_states, vhl_forces, tire_params_set_nelder)
    print('-' * 80)
    print('-' * 80)
    if conf.enable_plotting:
        plot_bell_curves(tire_params_set_svi,
                         std_params_set_svi, params_min, params_max)
        plot_tire_curves(vhl_states, vhl_forces,
                         tire_params_set_svi, tire_params_set_nelder)
        plt.show()


if __name__ == "__main__":
    main()
