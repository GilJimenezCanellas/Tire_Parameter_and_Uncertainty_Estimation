''' Helper functions for handling calculations in the GUI '''
import jax.numpy as jnp

from data_types.vehicleparameters import MFSimpleParams

from src.param_fitting.param_fitting import calc_force_shift, tire_param_fitting
from src.utils.calc_vhl_states import calc_vhl_states, calc_vhl_forces


def select_vhl_model(model, data, vhl_params, vhl_states, vhl_forces):
    ''' Calculate vehicle states and forces for the selected vehicle model '''
    print(f"Selected vehicle model: {model}")
    vhl_states = calc_vhl_states(data, vhl_params, vhl_states)
    vhl_forces = calc_vhl_forces(model, data, vhl_states, vhl_params, vhl_forces)
    print('Vehicle states and force calculation finished')


def param_fitting(conf, vhl_states, vhl_forces, tire_params_set_svi, std_params_set_svi, tire_params_set_nelder):
    ''' Fit tire parameters for the selected vehicle model '''
    axle_list = ['front_axle', 'rear_axle']
    direction_list = ['x', 'y']
    params_min = MFSimpleParams(**conf.params_min)
    params_max = MFSimpleParams(**conf.params_max)
    params_init = MFSimpleParams(**conf.params_init)
    for axle in axle_list:
        for direction in direction_list:
            sigma = jnp.array(getattr(getattr(vhl_states, axle), 'sigma_' + direction))
            force_n = jnp.array(getattr(getattr(vhl_forces, axle), 'force_' + direction + '_n'))
            load_n = jnp.array(getattr(vhl_forces, axle).force_z_n)
            fit_flags = getattr(conf, f'fit_flags_{axle}_{direction}')
            params_init.S_V, params_init.S_H = calc_force_shift(sigma, force_n)
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
            setattr(tire_params_set_nelder, f'{axle}_{direction}', params_nelder)
            setattr(tire_params_set_svi, f'{axle}_{direction}', params_svi)
            setattr(std_params_set_svi, f'{axle}_{direction}', std_svi)
