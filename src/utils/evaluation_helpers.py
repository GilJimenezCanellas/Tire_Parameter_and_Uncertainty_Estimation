''' Helper functions for evaluation and plotting '''
import jax.numpy as jnp
from scipy.stats import norm
import matplotlib.pyplot as plt

from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from data_types.vehiclestates import STMForces, STMStates

from src.utils.tiremodels import tire_model

LONGITUDINAL_TARGETS = [
    ('wheel_fl', 'wheel_fl_x', 'Front Left Longitudinal', '#A2AD00'),
    ('wheel_fr', 'wheel_fr_x', 'Front Right Longitudinal', '#7F8C00'),
    ('wheel_rl', 'wheel_rl_x', 'Rear Left Longitudinal', '#E37222'),
    ('wheel_rr', 'wheel_rr_x', 'Rear Right Longitudinal', '#B85400'),
]

LATERAL_TARGETS = [
    ('front_axle', 'front_axle_y', 'Front Lateral', '#0065BD'),
    ('rear_axle', 'rear_axle_y', 'Rear Lateral', '#64A0C8'),
]


def plot_bell_curves(params: STMTireParams, std_params: STMTireParams, params_min: MFSimpleParams, params_max: MFSimpleParams):
    ''' Plot the parameter distributions as bell curves '''
    if not params:
        print("No parameters to plot.")
        return
    param_names = ['B', 'C', 'D', 'E']
    column_width = 3.5
    tum_color = {param_key: color for _, param_key, _, color in LONGITUDINAL_TARGETS + LATERAL_TARGETS}
    fig, axes = plt.subplots(2, 2, figsize=(column_width, column_width), sharex=False)
    axes = axes.flatten()
    used_labels = []
    for i, param_name in enumerate(param_names):
        for _, key, _, _ in LONGITUDINAL_TARGETS + LATERAL_TARGETS:
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
            0.5, 0.05), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()

def plot_tire_curves(vehicle_states: STMStates, vehicle_forces: STMForces,
                     tire_params_set_svi: STMTireParams, tire_params_set_nelder: STMTireParams,):
    ''' Plot the resulting tire curves '''
    column_width = 10
    fig, ax = plt.subplots(3, 2, figsize=(column_width, column_width))
    ax = ax.flatten()
    plot_targets = [(state_key, param_key, title, 'x') for state_key, param_key, title, _ in LONGITUDINAL_TARGETS]
    plot_targets += [(state_key, param_key, title, 'y') for state_key, param_key, title, _ in LATERAL_TARGETS]
    for plot_pos, (state_key, param_key, title, direction) in enumerate(plot_targets):
        sigma_key = f'sigma_{direction}'
        force_key = f'force_{direction}_n'
        sigma = getattr(getattr(vehicle_states, state_key), sigma_key)
        load_n = getattr(getattr(vehicle_forces, state_key), 'force_z_n')
        force_n = getattr(getattr(vehicle_forces, state_key), force_key)
        slip_plot = jnp.linspace(jnp.min(sigma)-0.1, jnp.max(sigma)+0.1, 200)
        load_ref = jnp.median(load_n)
        ax[plot_pos].plot(
            slip_plot,
            tire_model('MFSimple', slip_plot, load_ref, getattr(tire_params_set_svi, param_key)) / load_ref,
            color='#0065BD',
        )
        ax[plot_pos].plot(
            slip_plot,
            tire_model('MFSimple', slip_plot, load_ref, getattr(tire_params_set_nelder, param_key)) / load_ref,
            color='#E37222',
        )
        ax[plot_pos].scatter(sigma, force_n / load_n, alpha=1.0, s=2, color='#DAD7CB')
        ax[plot_pos].set_title(title)
        ax[plot_pos].set_ylabel('Tire Force / Tire Load')
        ax[plot_pos].set_xlabel('Slip Ratio' if direction == 'x' else 'Slip Angle in rad')
    fig.legend(['SVI', 'Nelder-Mead'], loc='upper center',
               bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=10, frameon=False)
    plt.tight_layout()

def eval_force_errors(vehicle_states: STMStates, vehicle_forces: STMForces,
                      tire_params_set: STMTireParams):
    ''' Evaluate the force errors of the tire model '''
    print(f"{'Target':<24}{'Direction':<20}{'Mean':<20}{'Max':<20}")
    print('-' * 80)
    for state_key, param_key, title, _ in LONGITUDINAL_TARGETS:
        force_model = tire_model('MFSimple', getattr(getattr(vehicle_states, state_key), 'sigma_x'),
                                 getattr(getattr(vehicle_forces, state_key), 'force_z_n'),
                                 getattr(tire_params_set, param_key))
        force_errors = jnp.abs(force_model - getattr(getattr(vehicle_forces, state_key), 'force_x_n'))
        print(f"{title:<24}{'Longitudinal':<20}{jnp.mean(force_errors):<20.2f}{jnp.max(force_errors):<20.2f}")
    for state_key, param_key, title, _ in LATERAL_TARGETS:
        force_model = tire_model('MFSimple', getattr(getattr(vehicle_states, state_key), 'sigma_y'),
                                 getattr(getattr(vehicle_forces, state_key), 'force_z_n'),
                                 getattr(tire_params_set, param_key))
        force_errors = jnp.abs(force_model - getattr(getattr(vehicle_forces, state_key), 'force_y_n'))
        print(f"{title:<24}{'Lateral':<20}{jnp.mean(force_errors):<20.2f}{jnp.max(force_errors):<20.2f}")
