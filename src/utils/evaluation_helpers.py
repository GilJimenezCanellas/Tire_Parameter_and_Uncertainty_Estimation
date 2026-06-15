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
    ('front_axle', 'front_axle_y', 'Front Lateral', '#0072B2'),
    ('rear_axle', 'rear_axle_y', 'Rear Lateral', '#D55E00'),
]


def _to_numpy(values):
    '''Convert arrays to flat NumPy vectors for plotting utilities.'''
    return np.asarray(jnp.asarray(values), dtype=float).ravel()


def _selected_targets(include_longitudinal: bool = True):
    '''Return plotting/evaluation target metadata for the requested fit scope.'''
    return (LONGITUDINAL_TARGETS if include_longitudinal else []) + LATERAL_TARGETS


def _make_subplot_grid(num_targets: int, column_width: float = 10.0, row_height: float = 3.4):
    '''Create a compact 2-column grid and hide unused axes.'''
    num_targets = max(int(num_targets), 1)
    num_cols = 2 if num_targets > 1 else 1
    num_rows = int(np.ceil(num_targets / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(column_width, max(row_height, row_height * num_rows)))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[num_targets:]:
        axis.set_visible(False)
    return fig, axes


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


def _padded_limits(values, min_span: float = 1.0e-6, padding_fraction: float = 0.05):
    '''Return finite axis limits with padding, or None when no finite values exist.'''
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    span = max(upper - lower, min_span)
    padding = span * padding_fraction
    return lower - padding, upper + padding


def _combine_limits(limits):
    '''Combine existing axis-limit tuples into one finite range.'''
    finite_limits = [limit for limit in limits if limit is not None]
    if not finite_limits:
        return None
    return min(limit[0] for limit in finite_limits), max(limit[1] for limit in finite_limits)


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


def plot_excitation_histograms(excitation_source, num_bins: int = 30, include_longitudinal: bool = True):
    '''Plot slip-ratio and slip-angle excitation histograms used in tire fitting.'''
    if isinstance(excitation_source, STMStates):
        fit_excitation_samples = build_fit_excitation_samples(excitation_source)
    else:
        fit_excitation_samples = excitation_source

    lateral_values = [
        _extract_sigma_values(fit_excitation_samples.get(param_key, np.array([], dtype=float)))
        for _, param_key, _, _ in LATERAL_TARGETS
    ]
    lateral_bins = _build_symmetric_bins(lateral_values, num_bins)

    plot_targets = []
    if include_longitudinal:
        longitudinal_values = [
            _extract_sigma_values(fit_excitation_samples.get(param_key, np.array([], dtype=float)))
            for _, param_key, _, _ in LONGITUDINAL_TARGETS
        ]
        longitudinal_bins = _build_symmetric_bins(longitudinal_values, num_bins)
        plot_targets.extend(
            (param_key, title, color, 'x', longitudinal_bins)
            for _, param_key, title, color in LONGITUDINAL_TARGETS
        )
    plot_targets += [
        (param_key, title, color, 'y', lateral_bins)
        for _, param_key, title, color in LATERAL_TARGETS
    ]
    fig, axes = _make_subplot_grid(len(plot_targets), column_width=10.0)

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


def plot_bell_curves(params: STMTireParams, std_params: STMTireParams, params_min: MFSimpleParams,
                     params_max: MFSimpleParams, include_longitudinal: bool = True,
                     fit_flags_by_target: dict | None = None):
    ''' Plot the parameter distributions as bell curves '''
    if not params:
        print("No parameters to plot.")
        return
    param_names = ['B', 'C', 'D', 'E']
    column_width = 6.5
    plot_targets = _selected_targets(include_longitudinal)
    tum_color = {param_key: color for _, param_key, _, color in plot_targets}
    fig, axes = plt.subplots(2, 2, figsize=(column_width, column_width), sharex=False)
    axes = axes.flatten()
    legend_handles = {}
    for i, param_name in enumerate(param_names):
        fixed_annotations = []
        for _, key, _, _ in plot_targets:
            mean = getattr(params, key).__dict__[param_name]
            std_dev = getattr(std_params, key).__dict__[param_name]
            fit_flags = (fit_flags_by_target or {}).get(key, {})
            is_optimized = bool(fit_flags.get(param_name, std_dev > 0.0))
            if is_optimized and np.isfinite(std_dev) and std_dev > 0.0:
                x = jnp.linspace(params_min.__dict__[param_name],
                                 params_max.__dict__[param_name], 100)
                y = norm.pdf(x, mean, std_dev)
                y_sum = jnp.sum(y)
                if np.isfinite(float(y_sum)) and float(y_sum) > 0.0:
                    y = y / y_sum
                line, = axes[i].plot(x, y, color=tum_color[key], label=key)
                legend_handles.setdefault(key, line)
            else:
                line = axes[i].axvline(
                    mean,
                    color=tum_color[key],
                    linestyle='--' if not is_optimized else ':',
                    linewidth=1.7,
                    alpha=0.95,
                    label=key,
                )
                legend_handles.setdefault(key, line)
                status = 'fixed' if not is_optimized else 'zero std'
                fixed_annotations.append(f"{key}: {mean:.4g} ({status})")
        axes[i].set_title(f'Parameter {param_name}')
        axes[i].set_ylabel('Probability Density', fontsize=10)
        axes[i].set_xlim(params_min.__dict__[param_name],
                         params_max.__dict__[param_name])
        axes[i].grid(True, alpha=0.3)
        if fixed_annotations:
            axes[i].text(
                0.03,
                0.95,
                '\n'.join(fixed_annotations),
                transform=axes[i].transAxes,
                ha='left',
                va='top',
                fontsize=8,
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
            )
    fig.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(3, max(1, len(legend_handles))),
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))


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
                     tire_params_set_svi: STMTireParams, tire_params_set_nelder: STMTireParams | None = None,
                     clean_plots: bool = False, fit_data: dict | None = None,
                     include_longitudinal: bool = True):
    ''' Plot the resulting tire curves '''
    plot_targets = []
    if include_longitudinal:
        plot_targets.extend((state_key, param_key, title, 'x') for state_key, param_key, title, _ in LONGITUDINAL_TARGETS)
    plot_targets += [(state_key, param_key, title, 'y') for state_key, param_key, title, _ in LATERAL_TARGETS]
    fig, ax = _make_subplot_grid(len(plot_targets), column_width=10.0)
    lateral_axes = []
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
        if tire_params_set_nelder is not None:
            ax[plot_pos].plot(
                slip_plot,
                tire_model('MFSimple', slip_plot, load_ref, getattr(tire_params_set_nelder, param_key)) / load_ref,
                color='#E37222',
            )
        ax[plot_pos].scatter(sigma, force_n / load_n, alpha=1.0, s=0.5, color='#DAD7CB')
        ax[plot_pos].set_title(title)
        ax[plot_pos].set_ylabel('Tire Force / Tire Load')
        ax[plot_pos].set_xlabel('Slip Ratio' if direction == 'x' else 'Slip Angle in rad')
        ax[plot_pos].grid(True, alpha=0.3)
        if direction == 'y':
            lateral_axes.append(ax[plot_pos])
    lateral_xlim = _combine_limits([axis.get_xlim() for axis in lateral_axes])
    lateral_ylim = _combine_limits([axis.get_ylim() for axis in lateral_axes])
    for lateral_axis in lateral_axes:
        if lateral_xlim is not None:
            lateral_axis.set_xlim(*lateral_xlim)
        if lateral_ylim is not None:
            lateral_axis.set_ylim(*lateral_ylim)
    plt.tight_layout()


def plot_lateral_load_colored_curves(vehicle_states: STMStates, vehicle_forces: STMForces,
                                     tire_params_set: STMTireParams | None = None, fit_data: dict | None = None,
                                     num_load_regions: int = 4):
    '''Plot front/rear lateral friction coefficient against slip angle with Fz encoded as color.

    When balanced fit samples are passed through fit_data, the plot shows exactly
    the samples used by the tire fit. Otherwise it falls back to all filtered
    post-rejection state/force samples.
    '''
    del tire_params_set, num_load_regions

    column_width = 10
    fig, axes = plt.subplots(1, 2, figsize=(column_width, 4.8), sharey=True, constrained_layout=True)

    lateral_series = []
    for state_key, param_key, title, _ in LATERAL_TARGETS:
        sigma, load_n, force_n = _extract_fit_series(vehicle_states, vehicle_forces, fit_data, state_key, param_key, 'y')
        finite_mask = (
            np.isfinite(sigma)
            & np.isfinite(load_n)
            & np.isfinite(force_n)
            & (np.abs(load_n) > 1.0e-9)
        )
        sigma = sigma[finite_mask]
        load_n = load_n[finite_mask]
        mu_y = force_n[finite_mask] / load_n
        lateral_series.append((title, sigma, load_n, mu_y))

    load_sets = [load_n for _, _, load_n, _ in lateral_series if load_n.size]
    all_loads = np.concatenate(load_sets) if load_sets else np.array([], dtype=float)
    sigma_limits = _combine_limits(_padded_limits(sigma) for _, sigma, _, _ in lateral_series)
    mu_limits = _combine_limits(_padded_limits(mu_y) for _, _, _, mu_y in lateral_series)
    color_norm = None
    if all_loads.size:
        color_norm = plt.Normalize(float(np.min(all_loads)), float(np.max(all_loads)))

    scatter_handle = None
    for ax, (title, sigma, load_n, mu_y) in zip(axes, lateral_series):
        if sigma.size == 0:
            ax.set_title(f'{title}\nNo data')
            ax.set_xlabel('Slip Angle [rad]')
            ax.grid(True, alpha=0.25)
            continue

        scatter_handle = ax.scatter(
            sigma,
            mu_y,
            c=load_n,
            cmap='viridis',
            norm=color_norm,
            s=8,
            alpha=0.75,
            edgecolors='none',
        )
        ax.axhline(0.0, color='#333333', linestyle='--', linewidth=0.8, alpha=0.65)
        ax.axvline(0.0, color='#333333', linestyle='--', linewidth=0.8, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel('Slip Angle [rad]')
        if sigma_limits is not None:
            ax.set_xlim(*sigma_limits)
        if mu_limits is not None:
            ax.set_ylim(*mu_limits)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Lateral Friction Coefficient mu_y = Fy / Fz [-]')
    if scatter_handle is not None:
        colorbar = fig.colorbar(scatter_handle, ax=axes, shrink=0.95, pad=0.02)
        colorbar.set_label('Vertical Load Fz [N]')
    fig.suptitle('Lateral mu_y vs Slip Angle Colored By Fz', fontsize=14)


def eval_force_errors(vehicle_states: STMStates, vehicle_forces: STMForces,
                      tire_params_set: STMTireParams, include_longitudinal: bool = True):
    ''' Evaluate the force errors of the tire model '''
    print(f"{'Target':<24}{'Direction':<20}{'Mean':<20}{'Max':<20}")
    print('-' * 80)
    if include_longitudinal:
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
