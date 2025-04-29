''' Helper functions for evaluation and plotting '''
import jax.numpy as jnp
from scipy.stats import norm
import matplotlib.pyplot as plt

from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from data_types.vehiclestates import STMForces, STMStates

from src.utils.tiremodels import tire_model

def plot_bell_curves(params: STMTireParams, std_params: STMTireParams, params_min: MFSimpleParams, params_max: MFSimpleParams):
    ''' Plot the parameter distributions as bell curves '''
    if not params:
        print("No parameters to plot.")
        return
    param_names = ['B', 'C', 'D', 'E']
    column_width = 3.5
    tum_color = {
        'front_axle_y': '#0065BD',
        'rear_axle_y': '#64A0C8',
        'front_axle_x': '#A2AD00',
        'rear_axle_x': '#E37222'
    }
    fig, axes = plt.subplots(2, 2, figsize=(column_width, column_width), sharex=False)
    axes = axes.flatten()
    used_labels = []
    for i, param_name in enumerate(param_names):
        for key in params.__dict__.keys():
            mean = getattr(params, key).__dict__[param_name]
            std_dev = getattr(std_params, key).__dict__[param_name]
            x = jnp.linspace(params_min.__dict__[param_name],
                             params_max.__dict__[param_name], 100)
            y = norm.pdf(x, mean, std_dev)
            y = y / jnp.sum(y)
            if key not in used_labels:
                used_labels.append(key)
            axes[i].plot(x, y, color=tum_color[key])
        axes[i].set_title('Parameter {param_name} ')
        axes[i].set_ylabel('Probability Density', fontsize=10)
        axes[i].set_xlim(params_min.__dict__[param_name],
                         params_max.__dict__[param_name])
        fig.legend(labels=used_labels, loc='upper center', bbox_to_anchor=(
            0.5, 0.05), ncol=2, frameon=False, fontsize=10)
    plt.tight_layout()

def plot_tire_curves(vehicle_states: STMStates, vehicle_forces: STMForces,
                     tire_params_set_svi: STMTireParams, tire_params_set_nelder: STMTireParams,):
    ''' Plot the resulting tire curves '''
    column_width = 10
    fig, ax = plt.subplots(2, 2, figsize=(column_width, column_width * 0.618))
    ax = ax.flatten()
    plot_pos = 0
    slip_plot = jnp.linspace(-0.2, 0.2, 100)
    load = 3000.0
    # Calculate tire forces for SVI parameters
    tire_forces_svi = STMForces()
    tire_forces_svi.front_axle.force_x_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_svi.front_axle_x) / load
    tire_forces_svi.rear_axle.force_x_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_svi.rear_axle_x) / load
    tire_forces_svi.front_axle.force_y_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_svi.front_axle_y) / load
    tire_forces_svi.rear_axle.force_y_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_svi.rear_axle_y) / load
    # Calculate tire forces for Nelder-Mead parameters
    tire_forces_nelder = STMForces()
    tire_forces_nelder.front_axle.force_x_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_nelder.front_axle_x) / load
    tire_forces_nelder.rear_axle.force_x_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_nelder.rear_axle_x) / load
    tire_forces_nelder.front_axle.force_y_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_nelder.front_axle_y) / load
    tire_forces_nelder.rear_axle.force_y_n = tire_model(
        'MFSimple', slip_plot, load, tire_params_set_nelder.rear_axle_y) / load

    direction = ['long', 'lat']
    
    for axle in ['front_axle', 'rear_axle']:
        for direction in ['x', 'y']:
            force_key = f'force_{direction}_n'
            sigma_key = f'sigma_{direction}'
            ax[plot_pos].plot(slip_plot, getattr(
                getattr(tire_forces_svi, axle), force_key), color='#0065BD')
            ax[plot_pos].plot(slip_plot, getattr(
                getattr(tire_forces_nelder, axle), force_key), color='#E37222')
            ax[plot_pos].scatter(getattr(getattr(vehicle_states, axle), sigma_key),
                                 getattr(getattr(vehicle_forces, axle), force_key) /
                                 getattr(getattr(vehicle_forces, axle), 'force_z_n'),
                                 alpha=1.0, s=2, color='#DAD7CB')
            ax[plot_pos].set_ylabel('Tire Force / Tire Load')
            if direction == 'x':
                ax[plot_pos].set_xlabel('Slip Ratio')
            else:
                ax[plot_pos].set_xlabel('Slip Angle in rad')
            plot_pos += 1
        ax[0].set_title('Front Longitudinal')
        ax[1].set_title('Front Lateral')
        ax[2].set_title('Rear Longitudinal')
        ax[3].set_title('Rear Lateral')
    fig.legend(['SVI', 'Nelder-Mead'], loc='upper center',
               bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=10, frameon=False)
    plt.tight_layout()

def eval_force_errors(vehicle_states: STMStates, vehicle_forces: STMForces,
                      tire_params_set: STMTireParams):
    ''' Evaluate the force errors of the tire model '''
    tire_forces_model = STMForces()
    tire_forces_model.front_axle.force_x_n = tire_model('MFSimple', vehicle_states.front_axle.sigma_x,
                                                        vehicle_forces.front_axle.force_z_n, tire_params_set.front_axle_x)
    tire_forces_model.rear_axle.force_x_n = tire_model('MFSimple', vehicle_states.rear_axle.sigma_x,
                                                       vehicle_forces.rear_axle.force_z_n, tire_params_set.rear_axle_x)
    tire_forces_model.front_axle.force_y_n = tire_model('MFSimple', vehicle_states.front_axle.sigma_y,
                                                        vehicle_forces.front_axle.force_z_n, tire_params_set.front_axle_y)
    tire_forces_model.rear_axle.force_y_n = tire_model('MFSimple', vehicle_states.rear_axle.sigma_y,
                                                       vehicle_forces.rear_axle.force_z_n, tire_params_set.rear_axle_y)
    # Calculate Errors and RMSE
    force_errors = STMForces()
    force_errors.front_axle.force_x_n = jnp.abs(
        tire_forces_model.front_axle.force_x_n - vehicle_forces.front_axle.force_x_n)
    force_errors.rear_axle.force_x_n = jnp.abs(
        tire_forces_model.rear_axle.force_x_n - vehicle_forces.rear_axle.force_x_n)
    force_errors.front_axle.force_y_n = jnp.abs(
        tire_forces_model.front_axle.force_y_n - vehicle_forces.front_axle.force_y_n)
    force_errors.rear_axle.force_y_n = jnp.abs(
        tire_forces_model.rear_axle.force_y_n - vehicle_forces.rear_axle.force_y_n)
    print(f"{'Axle':<20}{'Direction':<20}{'Mean':<20}{'Max':<20}")
    print('-' * 80)
    print(f"{'Front Axle':<20}{'Longitudinal':<20}{jnp.mean(force_errors.front_axle.force_x_n):<20.2f}{jnp.max(force_errors.front_axle.force_x_n):<20.2f}")
    print(f"{'Rear Axle':<20}{'Longitudinal':<20}{jnp.mean(force_errors.rear_axle.force_x_n):<20.2f}{jnp.max(force_errors.rear_axle.force_x_n):<20.2f}")
    print(f"{'Front Axle':<20}{'Lateral':<20}{jnp.mean(force_errors.front_axle.force_y_n):<20.2f}{jnp.max(force_errors.front_axle.force_y_n):<20.2f}")
    print(f"{'Rear Axle':<20}{'Lateral':<20}{jnp.mean(force_errors.rear_axle.force_y_n):<20.2f}{jnp.max(force_errors.rear_axle.force_y_n):<20.2f}")
