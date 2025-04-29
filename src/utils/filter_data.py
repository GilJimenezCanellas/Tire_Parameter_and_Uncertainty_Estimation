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


def imu_offset_correction(lambda_ax: float, lambda_ay: float, yaw_rate_off: float, az_off: float, data: ImuData):
    ''' Corrects the IMU data according to the sensor offset '''
    data.acc_cog_x_mps2 = (data.acc_cog_x_mps2 -
                           jnp.median(data.acc_cog_x_mps2)) / jnp.cos(lambda_ax)
    data.acc_cog_y_mps2 = (data.acc_cog_y_mps2 -
                           jnp.median(data.acc_cog_y_mps2)) / jnp.cos(lambda_ay)
    data.yaw_rate_radps = data.yaw_rate_radps - yaw_rate_off
    data.acc_cog_z_mps2 = data.acc_cog_z_mps2 + (9.81 - az_off)
    return data

def preprocess_data(conf: Config, all_data: FilteredData = FilteredData()):
    ''' Preprocess all sensor data with predifined settings '''
    for run_name in tqdm(conf.run_names, desc="Loading raw data"):
        data_cor, data_gen, data_imu = load_raw_data(conf, run_name)
        data_cor = vel_offset_correction(conf.lambda_v, data_cor)
        data_imu = imu_offset_correction(conf.lambda_ax, conf.lambda_ay,
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


def filter_vhl_data(vhl_states: STMStates, vhl_forces: STMForces, threshold: float = 2):
    ''' Filter states and forces for outliers with a maximum standard deviation threshold (2 times default)'''
    mask = jnp.ones_like(vhl_states.beta, dtype=bool)
    for axle in ['front_axle', 'rear_axle']:
        for key in ['x', 'y']:
            sigma_values = getattr(getattr(vhl_states, axle), 'sigma_' + key)
            force_values = getattr(getattr(vhl_forces, axle), 'force_' + key + '_n')
            # Cluster sigma values into 20 clusters
            kmeans = KMeans(n_clusters=20, random_state=0).fit(sigma_values.reshape(-1, 1))
            clusters = kmeans.labels_
            for cluster in range(len(kmeans.cluster_centers_)):
                cluster_mask = (clusters == cluster)
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
    for field in fields(vhl_forces):
        if is_dataclass(field.type):
            for key in field.type.__dataclass_fields__:
                setattr(getattr(vhl_forces, field.name), key, jnp.array(
                    getattr(getattr(vhl_forces, field.name), key))[mask])
    print('Filtering of outliers finished')
    return vhl_states, vhl_forces
