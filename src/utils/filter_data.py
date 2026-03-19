''' Data Filtering and Offset Correction '''
from dataclasses import fields, is_dataclass
import os

from tqdm import tqdm
import numpy as np
from jax import numpy as jnp
from sklearn.cluster import KMeans
from scipy.stats import zscore
from scipy.signal import savgol_filter, resample, iirfilter, sosfilt
from scipy.ndimage import gaussian_filter1d

from data_types.config import Config
from data_types.sensordata import CorData, ImuData, FilteredData
from data_types.vehiclestates import STMStates, STMForces

from src.utils.datamanager import load_raw_data, load_filtered_data

def fitler_data(filtertype: str, settings: dict, data):
    ''' Filter data with different filter types '''
    match filtertype:
        case 'None':
            print('No Filter used')
        case 'moving_average':
            kernel = jnp.ones(settings['kernelsize']) / settings['kernelsize']
            for key in data.__dataclass_fields__:
                setattr(data, key, jnp.convolve(getattr(data, key), kernel, mode='same'))
        case 'butterworth':
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
            fs = settings['sampling_freq']  # Sampling frequency
            order = settings['filter_order']  # Filter order
            cutoff_freqs = [settings['freqs_lower'], settings['freqs_upper']]
            sos = iirfilter(order, Wn=cutoff_freqs, fs=fs, btype=settings['butter_type'],
                            ftype="butter", output="sos")
            for key in data.__dataclass_fields__:
                setattr(data, key, sosfilt(sos, getattr(data, key)))
        case 'savgol':
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
            # It specifies the number of data points to use in each local regression
            window_length = settings['window_length']
            poly_order = settings['order']
            for key in data.__dataclass_fields__:
                setattr(data, key, savgol_filter(getattr(data, key),
                                                 window_length=window_length, polyorder=poly_order, mode='nearest'))
        case 'gaussian':
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html
            sigma = settings['sigma']  # standard deviation for Gaussian kernel
            for key in data.__dataclass_fields__:
                setattr(data, key, gaussian_filter1d(getattr(data, key), sigma))
    return data


def vel_offset_correction(lambda_vel: float, data: CorData):
    ''' Corrects the velocity data according to the sensor offset '''
    data.vel_cog_x_mps = data.vel_cog_x_mps * \
        jnp.cos(lambda_vel) + data.vel_cog_y_mps * jnp.sin(lambda_vel)
    data.vel_cog_y_mps = - data.vel_cog_x_mps * \
        jnp.sin(lambda_vel) + data.vel_cog_y_mps * jnp.cos(lambda_vel)
    return data


def imu_offset_correction(median_ax: float, median_ay: float, lambda_ax: float, lambda_ay: float, yaw_rate_off: float, az_off: float, data: ImuData):
    ''' Corrects the IMU data according to the sensor offset '''
    data.acc_cog_x_mps2 = (data.acc_cog_x_mps2 -
                           median_ax) / jnp.cos(lambda_ax)
    data.acc_cog_y_mps2 = (data.acc_cog_y_mps2 -
                           median_ay) / jnp.cos(lambda_ay)
    data.yaw_rate_radps = data.yaw_rate_radps - yaw_rate_off
    data.acc_cog_z_mps2 = data.acc_cog_z_mps2 - az_off
    return data

def preprocess_data(conf: Config, all_data: FilteredData = FilteredData()):
    ''' Preprocess all sensor data with predifined settings '''
    for run_name in tqdm(conf.run_names, desc="Loading raw data"):
        data_cor, data_gen, data_imu = load_raw_data(conf, run_name)
        data_cor = vel_offset_correction(conf.lambda_v, data_cor)
        data_imu = imu_offset_correction(conf.ax_median, conf.ay_median,
                                         conf.lambda_ax, conf.lambda_ay,
                                         conf.yaw_rate_off, conf.az_off, data_imu)
        data_gen = fitler_data(conf.filter_gen, conf.settings_gen, data_gen)
        data_cor = fitler_data(conf.filter_cor, conf.settings_cor, data_cor)
        data_imu = fitler_data(conf.filter_imu, conf.settings_imu, data_imu)
        for key in data_cor.__dataclass_fields__:
            setattr(data_cor, key, resample(getattr(data_cor, key), len(data_gen.time)))
        for key in data_imu.__dataclass_fields__:
            setattr(data_imu, key, resample(getattr(data_imu, key), len(data_gen.time)))
        sensordata = FilteredData()
        sensordata.gen_data = data_gen
        sensordata.cor_data = data_cor
        sensordata.imu_data = data_imu
        # Create consecutive time vector
        sensordata.gen_data.time = jnp.linspace(
            0, len(sensordata.gen_data.time) / conf.sampling_freq_hz, len(sensordata.gen_data.time))
        if conf.imu_acc_z_fix:
            sensordata.imu_data.acc_cog_z_mps2 = jnp.ones_like(
                sensordata.imu_data.acc_cog_z_mps2) * 9.81
        # Filter for low speed data
        mask = jnp.array(sensordata.cor_data.vel_cog_x_mps) > conf.low_speed_filter_mps
        if conf.gear_change_filter:
            # Identify where the gear changes
            gear_changes = jnp.diff(sensordata.gen_data.gear) != 0
            gear_change_indices = jnp.where(gear_changes)[0]
            # Create a mask for gear changes and the following 0.2 seconds
            for idx in gear_change_indices:
                shift_oscillation_delay_s = 0.2
                idx_oscillation_delay = int(shift_oscillation_delay_s * conf.sampling_freq_hz)
                mask = mask.at[idx:idx+idx_oscillation_delay].set(False)
        # Apply the mask to all fields in sensordata
        for field in fields(sensordata):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(sensordata, field.name), key, jnp.array(
                    getattr(getattr(sensordata, field.name), key))[mask])
        if hasattr(all_data, 'gen_data'):  # check if all_data has been initialized
            for field in fields(all_data):
                for key in field.type.__dataclass_fields__:
                    dat = jnp.array(getattr(getattr(all_data, field.name), key))
                    dat = jnp.concatenate(
                        (dat, jnp.array(getattr(getattr(sensordata, field.name), key))))
                    setattr(getattr(all_data, field.name), key, dat)
        else:
            all_data = sensordata
    print('Preprocessing finished')
    return all_data

def filter_vhl_data(vhl_states: STMStates, vhl_forces: STMForces, threshold: float = 2,
                    return_mask: bool = False):
    ''' Filter states and forces for outliers with a maximum standard deviation threshold (2 times default)'''
    mask = jnp.ones_like(vhl_states.beta, dtype=bool)
    fit_targets = [
        ('wheel_fl', 'x'),
        ('wheel_fr', 'x'),
        ('wheel_rl', 'x'),
        ('wheel_rr', 'x'),
        ('front_axle', 'y'),
        ('rear_axle', 'y'),
    ]
    for axle, key in fit_targets:
        sigma_values = getattr(getattr(vhl_states, axle), 'sigma_' + key)
        force_values = getattr(getattr(vhl_forces, axle), 'force_' + key + '_n')
        # Cluster sigma values into 20 clusters
        n_clusters = min(20, len(sigma_values))
        if n_clusters < 2:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sigma_values.reshape(-1, 1))
        clusters = kmeans.labels_
        for cluster in range(len(kmeans.cluster_centers_)):
            cluster_mask = (clusters == cluster)
            if jnp.sum(cluster_mask) < 2:
                continue
            cluster_force_values = force_values[cluster_mask]
            tire_data = np.column_stack((sigma_values[cluster_mask], cluster_force_values))
            z_scores = jnp.abs(zscore(tire_data, axis=0))
            mask = mask.at[cluster_mask].set(
                mask[cluster_mask] & (z_scores < threshold).all(axis=1))
    for field in fields(vhl_states):
        if is_dataclass(field.type):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(vhl_states, field.name), key, jnp.array(
                    getattr(getattr(vhl_states, field.name), key))[mask])
        else:
            val = getattr(vhl_states, field.name)
            if isinstance(val, (jnp.ndarray, np.ndarray)) and len(val) == len(mask):
                setattr(vhl_states, field.name, jnp.array(val)[mask])
    for field in fields(vhl_forces):
        if is_dataclass(field.type):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(vhl_forces, field.name), key, jnp.array(
                    getattr(getattr(vhl_forces, field.name), key))[mask])
    print('Filtering of outliers finished')
    if return_mask:
        return vhl_states, vhl_forces, np.asarray(mask, dtype=bool)
    return vhl_states, vhl_forces

def reject_transient_data(sensordata: FilteredData, vhl_states: STMStates, vhl_forces: STMForces, 
                          max_yaw_accel_radps2: float = 0.5, 
                          max_steer_vel_radps: float = 0.15,
                          return_mask: bool = False):
    ''' Filter out highly transient data points based on angular acceleration and steering velocity. '''
    
    # 1. Calculate steering velocity (delta_dot)
    delta_dot = jnp.gradient(sensordata.gen_data.delta_f_rad, sensordata.gen_data.time)
    
    # 2. Create the steady-state boolean mask
    mask = (jnp.abs(vhl_states.dd_psi) < max_yaw_accel_radps2) & \
           (jnp.abs(delta_dot) < max_steer_vel_radps)
    
    # 3. Apply mask to vhl_states (both top-level and nested)
    for field in fields(vhl_states):
        if is_dataclass(field.type):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(vhl_states, field.name), key, 
                        jnp.array(getattr(getattr(vhl_states, field.name), key))[mask])
        else:
            # Handle top-level arrays like beta and dd_psi
            val = getattr(vhl_states, field.name)
            if isinstance(val, (jnp.ndarray, np.ndarray)) and len(val) == len(mask):
                setattr(vhl_states, field.name, jnp.array(val)[mask])
                
    # 4. Apply mask to vhl_forces (nested)
    for field in fields(vhl_forces):
        if is_dataclass(field.type):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(vhl_forces, field.name), key, 
                        jnp.array(getattr(getattr(vhl_forces, field.name), key))[mask])

    dropped_pts = len(mask) - jnp.sum(mask)
    print(f"Transient rejection: Dropped {dropped_pts} points. {jnp.sum(mask)} steady-state points remaining.")

    if return_mask:
        return vhl_states, vhl_forces, np.asarray(mask, dtype=bool)
    return vhl_states, vhl_forces


def _robust_abs_scale(values: np.ndarray, percentile: float = 75.0, floor: float = 1.0e-6) -> float:
    '''Return a robust normalization factor for absolute-valued signals.'''
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return floor
    scale = float(np.percentile(np.abs(values), percentile))
    return max(scale, floor)


def _build_balancing_edges(sigma_values: np.ndarray, num_regions: int, tail_quantile: float) -> np.ndarray:
    '''Construct symmetric slip regions with open-ended tails for rare extreme excitations.'''
    sigma_values = np.asarray(sigma_values, dtype=float)
    if sigma_values.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    max_abs = float(np.max(np.abs(sigma_values)))
    if not np.isfinite(max_abs) or max_abs < 1.0e-6:
        return np.array([-np.inf, 0.0, np.inf], dtype=float)
    num_regions = max(int(num_regions), 3)
    tail_limit = float(np.quantile(np.abs(sigma_values), tail_quantile))
    tail_limit = min(max(tail_limit, 1.0e-6), max_abs)
    inner_region_count = max(num_regions - 2, 1)
    inner_edges = np.linspace(-tail_limit, tail_limit, inner_region_count + 1)
    return np.concatenate(([-np.inf], inner_edges, [np.inf]))


def select_balanced_fit_samples(sigma_values, force_values, load_values, yaw_accel_values, steer_vel_values,
                                target_count: int, num_regions: int | None = None, tail_quantile: float = 0.9,
                                target_points_per_region: int = 125, max_regions: int = 12,
                                yaw_weight: float = 1.0, steer_weight: float = 1.0):
    '''Select a balanced per-target subset by keeping the least-transient points in each slip region.'''
    sigma_values = np.asarray(sigma_values, dtype=float).ravel()
    force_values = np.asarray(force_values, dtype=float).ravel()
    load_values = np.asarray(load_values, dtype=float).ravel()
    yaw_accel_values = np.asarray(yaw_accel_values, dtype=float).ravel()
    steer_vel_values = np.asarray(steer_vel_values, dtype=float).ravel()

    valid_mask = np.isfinite(sigma_values) & np.isfinite(force_values) & np.isfinite(load_values)
    valid_mask &= np.isfinite(yaw_accel_values) & np.isfinite(steer_vel_values)
    sigma_values = sigma_values[valid_mask]
    force_values = force_values[valid_mask]
    load_values = load_values[valid_mask]
    yaw_accel_values = yaw_accel_values[valid_mask]
    steer_vel_values = steer_vel_values[valid_mask]

    if sigma_values.size == 0:
        return {
            'sigma': jnp.array([]),
            'force_n': jnp.array([]),
            'load_n': jnp.array([]),
            'region_edges': np.array([-np.inf, np.inf], dtype=float),
            'region_counts_before': [],
            'region_counts_after': [],
            'transient_score': jnp.array([]),
        }

    target_count = max(1, min(int(target_count), sigma_values.size))
    if num_regions is None:
        num_regions = int(np.clip(round(target_count / max(target_points_per_region, 1)), 6, max_regions))
    num_regions = min(max(int(num_regions), 3), sigma_values.size)
    region_edges = _build_balancing_edges(sigma_values, num_regions, tail_quantile)

    yaw_scale = _robust_abs_scale(yaw_accel_values)
    steer_scale = _robust_abs_scale(steer_vel_values)
    transient_score = yaw_weight * np.abs(yaw_accel_values) / yaw_scale
    transient_score += steer_weight * np.abs(steer_vel_values) / steer_scale

    region_candidate_indices = []
    region_counts_before = []
    for region_idx in range(len(region_edges) - 1):
        lower_edge = region_edges[region_idx]
        upper_edge = region_edges[region_idx + 1]
        if region_idx == len(region_edges) - 2:
            region_mask = (sigma_values >= lower_edge) & (sigma_values <= upper_edge)
        else:
            region_mask = (sigma_values >= lower_edge) & (sigma_values < upper_edge)
        candidate_indices = np.where(region_mask)[0]
        candidate_indices = candidate_indices[np.argsort(transient_score[candidate_indices], kind='stable')]
        region_candidate_indices.append(candidate_indices)
        region_counts_before.append(int(candidate_indices.size))

    quotas = np.array([min(target_count // len(region_candidate_indices), count)
                       for count in region_counts_before], dtype=int)
    remaining_budget = target_count - int(np.sum(quotas))
    while remaining_budget > 0:
        eligible = [idx for idx, candidate_indices in enumerate(region_candidate_indices)
                    if quotas[idx] < len(candidate_indices)]
        if not eligible:
            break
        chosen_region = min(eligible, key=lambda idx: (quotas[idx], -region_counts_before[idx], idx))
        quotas[chosen_region] += 1
        remaining_budget -= 1

    selected_indices = [
        candidate_indices[:quotas[idx]]
        for idx, candidate_indices in enumerate(region_candidate_indices)
        if quotas[idx] > 0
    ]
    if selected_indices:
        selected_indices = np.sort(np.concatenate(selected_indices))
    else:
        selected_indices = np.array([], dtype=int)

    return {
        'sigma': jnp.array(sigma_values[selected_indices]),
        'force_n': jnp.array(force_values[selected_indices]),
        'load_n': jnp.array(load_values[selected_indices]),
        'region_edges': region_edges,
        'region_counts_before': region_counts_before,
        'region_counts_after': quotas.tolist(),
        'transient_score': jnp.array(transient_score[selected_indices]),
    }
