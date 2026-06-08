''' Helper functions for evaluation and plotting '''
import jax.numpy as jnp
import numpy as np
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


def _to_numpy(values):
    '''Convert arrays to flat NumPy vectors for plotting utilities.'''
    return np.asarray(jnp.asarray(values), dtype=float).ravel()


def _extract_sigma_values(sample_entry):
    '''Extract slip values from either raw arrays or stored fit-sample dictionaries.'''
    if isinstance(sample_entry, dict):
        return _to_numpy(sample_entry.get('sigma', []))
    return _to_numpy(sample_entry)


def _extract_region_edges(sample_entry, sigma_values):
    '''Return finite plotting edges derived from stored balancing edges when available.'''
    if isinstance(sample_entry, dict) and 'region_edges' in sample_entry:
        plot_edges = np.asarray(sample_entry['region_edges'], dtype=float).copy()
        sigma_values = _to_numpy(sigma_values)
        if sigma_values.size == 0:
            return None
        sigma_min = float(np.min(sigma_values))
        sigma_max = float(np.max(sigma_values))
        span = max(sigma_max - sigma_min, 1.0e-3)
        edge_margin = 0.01 * span
        if not np.isfinite(plot_edges[0]):
            plot_edges[0] = min(sigma_min - edge_margin, plot_edges[1] - edge_margin)
        if not np.isfinite(plot_edges[-1]):
            plot_edges[-1] = max(sigma_max + edge_margin, plot_edges[-2] + edge_margin)
        for edge_idx in range(1, len(plot_edges)):
            if plot_edges[edge_idx] <= plot_edges[edge_idx - 1]:
                plot_edges[edge_idx] = np.nextafter(plot_edges[edge_idx - 1], np.inf)
        return plot_edges
    return None


def _build_symmetric_bins(value_sets, num_bins: int, min_half_range: float = 1.0e-3):
    '''Create symmetric histogram bins around zero for easier balance inspection.'''
    max_abs = max((np.max(np.abs(values)) for values in value_sets if values.size), default=min_half_range)
    max_abs = max(max_abs, min_half_range)
    return np.linspace(-max_abs, max_abs, num_bins + 1)


def _summarize_excitation(values):
    '''Compute compact balance statistics for histogram annotations.'''
    total = int(values.size)
    if total == 0:
        return None
    neg_count = int(np.sum(values < 0.0))
    pos_count = int(np.sum(values > 0.0))
    zero_count = total - neg_count - pos_count
    signed_count = max(neg_count + pos_count, 1)
    return {
        'count': total,
        'neg_share': neg_count / total,
        'pos_share': pos_count / total,
        'zero_share': zero_count / total,
        'imbalance': abs(pos_count - neg_count) / signed_count,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }


def build_fit_excitation_samples(vehicle_states: STMStates):
    '''Collect the slip signals associated with each fitting target.'''
    fit_excitation_samples = {}
    for state_key, param_key, _, _ in LONGITUDINAL_TARGETS:
        fit_excitation_samples[param_key] = _to_numpy(getattr(getattr(vehicle_states, state_key), 'sigma_x'))
    for state_key, param_key, _, _ in LATERAL_TARGETS:
        fit_excitation_samples[param_key] = _to_numpy(getattr(getattr(vehicle_states, state_key), 'sigma_y'))
    return fit_excitation_samples


def plot_excitation_histograms(excitation_source, num_bins: int = 30):
    '''Plot slip-ratio and slip-angle excitation histograms used in tire fitting.'''
    column_width = 10
    fig, axes = plt.subplots(3, 2, figsize=(column_width, column_width))
    axes = axes.flatten()

    if isinstance(excitation_source, STMStates):
        fit_excitation_samples = build_fit_excitation_samples(excitation_source)
    else:
        fit_excitation_samples = excitation_source

    longitudinal_values = [
        _extract_sigma_values(fit_excitation_samples.get(param_key, np.array([], dtype=float)))
        for _, param_key, _, _ in LONGITUDINAL_TARGETS
    ]
    lateral_values = [
        _extract_sigma_values(fit_excitation_samples.get(param_key, np.array([], dtype=float)))
        for _, param_key, _, _ in LATERAL_TARGETS
    ]
    longitudinal_bins = _build_symmetric_bins(longitudinal_values, num_bins)
    lateral_bins = _build_symmetric_bins(lateral_values, num_bins)

    plot_targets = [
        (param_key, title, color, 'x', longitudinal_bins)
        for _, param_key, title, color in LONGITUDINAL_TARGETS
    ]
    plot_targets += [
        (param_key, title, color, 'y', lateral_bins)
        for _, param_key, title, color in LATERAL_TARGETS
    ]

    for plot_pos, (param_key, title, color, direction, bins) in enumerate(plot_targets):
        sample_entry = fit_excitation_samples.get(param_key, np.array([], dtype=float))
        sigma = _extract_sigma_values(sample_entry)
        plot_bins = _extract_region_edges(sample_entry, sigma)
        if plot_bins is None:
            plot_bins = bins
        axes[plot_pos].hist(
            sigma,
            bins=plot_bins,
            color=color,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )
        axes[plot_pos].axvline(0.0, color='#333333', linestyle='--', linewidth=1.0)
        axes[plot_pos].set_xlim(plot_bins[0], plot_bins[-1])
        axes[plot_pos].set_title(title)
        axes[plot_pos].set_ylabel('Samples')
        axes[plot_pos].set_xlabel('Slip Ratio' if direction == 'x' else 'Slip Angle in rad')
        axes[plot_pos].grid(axis='y', alpha=0.2)

        stats = _summarize_excitation(sigma)
        if stats is None:
            annotation = 'No data'
        else:
            annotation = (
                f"n={stats['count']}\n"
                f"mu={stats['mean']:.3f}, sigma={stats['std']:.3f}\n"
                f"-={stats['neg_share']:.0%}, +={stats['pos_share']:.0%}\n"
                f"0={stats['zero_share']:.0%}, imbalance={stats['imbalance']:.0%}"
            )
        axes[plot_pos].text(
            0.97,
            0.95,
            annotation,
            transform=axes[plot_pos].transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )

    fig.suptitle('Slip Excitation Used For Tire Fitting', fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))


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
        axes[i].set_title(f'Parameter {param_name}')
        axes[i].set_ylabel('Probability Density', fontsize=10)
        axes[i].set_xlim(params_min.__dict__[param_name],
                         params_max.__dict__[param_name])
        fig.legend(labels=used_labels, loc='upper center', bbox_to_anchor=(
            0.5, 0.05), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()


def _clean_plot_points(sigma, load_n, force_n, svi_params, trim_fraction: float = 1.0 / 3.0):
    '''Remove the furthest points from the SVI tire model for cleaner visualization.'''
    sigma = jnp.asarray(sigma)
    load_n = jnp.asarray(load_n)
    force_n = jnp.asarray(force_n)
    if len(sigma) < 3:
        return sigma, load_n, force_n

    safe_load_n = jnp.maximum(jnp.abs(load_n), 1.0e-6)
    measured_force_norm = force_n / safe_load_n
    model_svi_norm = tire_model('MFSimple', sigma, load_n, svi_params) / safe_load_n
    point_distance = jnp.abs(measured_force_norm - model_svi_norm)

    keep_fraction = max(0.0, min(1.0, 1.0 - trim_fraction))
    if keep_fraction <= 0.0:
        return sigma, load_n, force_n
    threshold = float(np.quantile(np.asarray(point_distance), keep_fraction))
    keep_mask = np.asarray(point_distance) <= threshold
    return sigma[keep_mask], load_n[keep_mask], force_n[keep_mask]


def _extract_fit_series(vehicle_states: STMStates, vehicle_forces: STMForces, fit_data: dict | None,
                        state_key: str, param_key: str, direction: str):
    '''Return sigma, load, and force arrays from fit samples when available, otherwise raw states/forces.'''
    if fit_data and param_key in fit_data:
        sigma = _to_numpy(fit_data[param_key].get('sigma', []))
        load_n = _to_numpy(fit_data[param_key].get('load_n', []))
        force_n = _to_numpy(fit_data[param_key].get('force_n', []))
    else:
        sigma = _to_numpy(getattr(getattr(vehicle_states, state_key), f'sigma_{direction}'))
        load_n = _to_numpy(getattr(getattr(vehicle_forces, state_key), 'force_z_n'))
        force_n = _to_numpy(getattr(getattr(vehicle_forces, state_key), f'force_{direction}_n'))
    return sigma, load_n, force_n


def _build_load_regions(load_n, num_regions: int):
    '''Split Fz values into monotonic quantile-based regions and skip empty regions.'''
    load_n = np.asarray(load_n, dtype=float).ravel()
    if load_n.size == 0:
        return []
    if load_n.size == 1:
        eps = max(abs(float(load_n[0])) * 1.0e-6, 1.0e-6)
        return [(load_n[0] - eps, load_n[0] + eps, np.array([True]))]

    quantiles = np.linspace(0.0, 1.0, min(max(num_regions, 1), load_n.size) + 1)
    edges = np.quantile(load_n, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        eps = max(abs(float(edges[0])) * 1.0e-6, 1.0e-6)
        edges = np.array([edges[0] - eps, edges[0] + eps], dtype=float)

    regions = []
    for region_idx in range(len(edges) - 1):
        lower_edge = float(edges[region_idx])
        upper_edge = float(edges[region_idx + 1])
        if region_idx == len(edges) - 2:
            mask = (load_n >= lower_edge) & (load_n <= upper_edge)
        else:
            mask = (load_n >= lower_edge) & (load_n < upper_edge)
        if np.any(mask):
            regions.append((lower_edge, upper_edge, mask))
    return regions


def plot_tire_curves(vehicle_states: STMStates, vehicle_forces: STMForces,
                     tire_params_set_svi: STMTireParams, tire_params_set_nelder: STMTireParams,
                     clean_plots: bool = False, fit_data: dict | None = None):
    ''' Plot the resulting tire curves '''
    column_width = 10
    fig, ax = plt.subplots(3, 2, figsize=(column_width, column_width))
    ax = ax.flatten()
    plot_targets = [(state_key, param_key, title, 'x') for state_key, param_key, title, _ in LONGITUDINAL_TARGETS]
    plot_targets += [(state_key, param_key, title, 'y') for state_key, param_key, title, _ in LATERAL_TARGETS]
    for plot_pos, (state_key, param_key, title, direction) in enumerate(plot_targets):
        if fit_data and param_key in fit_data:
            sigma = jnp.asarray(fit_data[param_key]['sigma'])
            load_n = jnp.asarray(fit_data[param_key]['load_n'])
            force_n = jnp.asarray(fit_data[param_key]['force_n'])
        else:
            sigma_key = f'sigma_{direction}'
            force_key = f'force_{direction}_n'
            sigma = getattr(getattr(vehicle_states, state_key), sigma_key)
            load_n = getattr(getattr(vehicle_forces, state_key), 'force_z_n')
            force_n = getattr(getattr(vehicle_forces, state_key), force_key)
        if len(sigma) == 0:
            continue
        if clean_plots:
            sigma, load_n, force_n = _clean_plot_points(
                sigma,
                load_n,
                force_n,
                getattr(tire_params_set_svi, param_key),
            )
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
        ax[plot_pos].scatter(sigma, force_n / load_n, alpha=1.0, s=0.5, color='#DAD7CB')
        ax[plot_pos].set_title(title)
        ax[plot_pos].set_ylabel('Tire Force / Tire Load')
        ax[plot_pos].set_xlabel('Slip Ratio' if direction == 'x' else 'Slip Angle in rad')
    fig.legend(['SVI', 'Nelder-Mead'], loc='upper center',
               bbox_to_anchor=(0.5, 0.04), ncol=2, fontsize=10, frameon=False)
    plt.tight_layout()


def plot_lateral_load_colored_curves(vehicle_states: STMStates, vehicle_forces: STMForces,
                                     tire_params_set: STMTireParams, fit_data: dict | None = None,
                                     num_load_regions: int = 4):
    '''Plot normalized lateral samples grouped by Fz regions for front and rear axles.'''
    column_width = 10
    fig, axes = plt.subplots(1, 2, figsize=(column_width, 4.6), sharey=True, constrained_layout=True)

    for ax, (state_key, param_key, title, _) in zip(axes, LATERAL_TARGETS):
        sigma, load_n, force_n = _extract_fit_series(vehicle_states, vehicle_forces, fit_data, state_key, param_key, 'y')
        if sigma.size == 0:
            ax.set_title(f'{title}\nNo data')
            ax.set_xlabel('Slip Angle in rad')
            ax.grid(True, alpha=0.25)
            continue

        safe_load_n = np.maximum(np.abs(load_n), 1.0e-6)
        load_regions = _build_load_regions(load_n, num_load_regions)
        cmap = plt.get_cmap('viridis', max(len(load_regions), 1))
        slip_margin = max(0.02, 0.05 * max(np.max(np.abs(sigma)), 1.0e-3))
        slip_plot = jnp.linspace(float(np.min(sigma) - slip_margin), float(np.max(sigma) + slip_margin), 250)

        for region_idx, (lower_edge, upper_edge, mask) in enumerate(load_regions):
            region_sigma = sigma[mask]
            region_force_norm = force_n[mask] / safe_load_n[mask]
            region_load_ref = float(np.median(load_n[mask]))
            region_curve = tire_model(
                'MFSimple',
                slip_plot,
                region_load_ref,
                getattr(tire_params_set, param_key),
            ) / region_load_ref
            color = cmap(region_idx)
            ax.scatter(
                region_sigma,
                region_force_norm,
                s=10,
                alpha=0.65,
                color=color,
                edgecolors='none',
            )
            ax.plot(
                slip_plot,
                region_curve,
                color=color,
                linewidth=2.0,
                label=f'{lower_edge:.0f}-{upper_edge:.0f} N (n={int(np.count_nonzero(mask))})',
            )

        ax.set_title(f'{title}\nNormalized samples grouped by Fz')
        ax.set_xlabel('Slip Angle in rad')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9, title='Fz regions')

    axes[0].set_ylabel('Lateral Force / Vertical Load')
    fig.suptitle('Lateral Tire Curves With Fz-Colored Samples', fontsize=14)


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
