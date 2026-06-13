"""AMZ tire fitting pipeline helpers.

This module contains reusable pipeline logic only. Command-line entry points,
input staging, configuration files, vehicle parameters, and output management
are owned by the integrating repository.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

from data_types.sensordata import CorData, FilteredData, GenData, ImuData
from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from src.param_fitting.param_fitting import calc_force_shift, tire_param_fitting
from src.utils.calc_vhl_states import (
    calc_vhl_forces,
    calc_vhl_states,
    validate_required_sensor_signals,
)
from src.utils.datamanager import load_params
from src.utils.filter_data import (
    filter_vhl_data,
    fitler_data,
    imu_offset_correction,
    reject_transient_data,
    select_balanced_fit_samples,
    vel_offset_correction,
)
from src.utils.tiremodels import tire_model


LONGITUDINAL_FIT_TARGETS = [
    ("wheel_fl", "x"),
    ("wheel_fr", "x"),
    ("wheel_rl", "x"),
    ("wheel_rr", "x"),
]
LATERAL_FIT_TARGETS = [
    ("front_axle", "y"),
    ("rear_axle", "y"),
]
FIT_TARGETS = LONGITUDINAL_FIT_TARGETS + LATERAL_FIT_TARGETS
BALANCE_TAIL_QUANTILE = 0.9
BALANCE_TARGET_POINTS_PER_REGION = 250
BALANCE_MAX_REGIONS = 12


@dataclass
class RosbagSignalSeries:
    """Time-aligned scalar signals extracted from one ROS bag topic."""
    time_s: np.ndarray
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class RosbagTopicSpec:
    """Fixed AMZ MCAP topic and message type."""
    topic: str
    type_name: str


AMZ_ROSBAG_TOPICS = {
    "velocity": RosbagTopicSpec(
        "/vcu/nera/velocity_estimation",
        "autonomous_msgs_nera/msg/VelocityEstimation",
    ),
    "torque": RosbagTopicSpec(
        "/vcu/nera/torque_data",
        "autonomous_msgs_nera/msg/TorqueData",
    ),
    "steering": RosbagTopicSpec(
        "/vcu/nera/steering_feedback",
        "autonomous_msgs_nera/msg/DoubleStamped",
    ),
}


def selected_fit_targets(fit_longitudinal: bool = False) -> list[tuple[str, str]]:
    """Return tire targets requested for parameter fitting."""
    return (LONGITUDINAL_FIT_TARGETS if fit_longitudinal else []) + LATERAL_FIT_TARGETS


def _clip_shift_component(name: str, value: float, params_min: MFSimpleParams,
                          params_max: MFSimpleParams) -> tuple[float, bool]:
    """Clip one force-shift component to the configured parameter bounds."""
    lower_bound = float(getattr(params_min, name))
    upper_bound = float(getattr(params_max, name))
    if lower_bound > upper_bound:
        raise ValueError(
            f"Invalid {name} bounds: min {lower_bound} is greater than max {upper_bound}."
        )
    clipped_value = float(np.clip(value, lower_bound, upper_bound))
    return clipped_value, not np.isclose(clipped_value, float(value), rtol=0.0, atol=1.0e-12)


def _clip_lateral_force_shift(vertical_shift_n: float, horizontal_shift_rad: float,
                              params_min: MFSimpleParams,
                              params_max: MFSimpleParams) -> tuple[float, float, bool]:
    """Clip lateral S_V and S_H estimates before they are used by the fitters."""
    clipped_vertical_shift_n, vertical_was_clipped = _clip_shift_component(
        "S_V", vertical_shift_n, params_min, params_max
    )
    clipped_horizontal_shift_rad, horizontal_was_clipped = _clip_shift_component(
        "S_H", horizontal_shift_rad, params_min, params_max
    )
    return (
        clipped_vertical_shift_n,
        clipped_horizontal_shift_rad,
        vertical_was_clipped or horizontal_was_clipped,
    )


def calc_total_lateral_force_body_n_from_pacejka(sensordata: FilteredData, vhl_states, vhl_forces,
                                                 tire_params_set: STMTireParams) -> jnp.array:
    """Estimate body-frame lateral force from measured slip angles, Fz, and fitted Pacejka parameters."""
    delta_f_rad = sensordata.gen_data.delta_f_rad
    force_y_front_tire_n = tire_model(
        "MFSimple",
        vhl_states.front_axle.sigma_y,
        vhl_forces.front_axle.force_z_n,
        tire_params_set.front_axle_y,
    )
    force_y_rear_tire_n = tire_model(
        "MFSimple",
        vhl_states.rear_axle.sigma_y,
        vhl_forces.rear_axle.force_z_n,
        tire_params_set.rear_axle_y,
    )
    force_y_front_body_n = jnp.cos(delta_f_rad) * force_y_front_tire_n
    return force_y_front_body_n + force_y_rear_tire_n


def calc_total_longitudinal_force_body_n_from_wheel_forces(sensordata: FilteredData, vhl_forces) -> jnp.array:
    """Sum wheel longitudinal forces in the body x-direction using front steering angle."""
    delta_f_rad = sensordata.gen_data.delta_f_rad
    force_x_front_body_n = jnp.cos(delta_f_rad) * (
        vhl_forces.wheel_fl.force_x_n + vhl_forces.wheel_fr.force_x_n
    )
    force_x_rear_body_n = vhl_forces.wheel_rl.force_x_n + vhl_forces.wheel_rr.force_x_n
    return force_x_front_body_n + force_x_rear_body_n


def calc_total_longitudinal_force_body_n_from_pacejka(sensordata: FilteredData, vhl_states, vhl_forces,
                                                      tire_params_set: STMTireParams) -> jnp.array:
    """Estimate body-frame longitudinal force from wheel slip ratios, wheel loads, and fitted tire models."""
    delta_f_rad = sensordata.gen_data.delta_f_rad
    force_x_fl_tire_n = tire_model(
        "MFSimple",
        vhl_states.wheel_fl.sigma_x,
        vhl_forces.wheel_fl.force_z_n,
        tire_params_set.wheel_fl_x,
    )
    force_x_fr_tire_n = tire_model(
        "MFSimple",
        vhl_states.wheel_fr.sigma_x,
        vhl_forces.wheel_fr.force_z_n,
        tire_params_set.wheel_fr_x,
    )
    force_x_rl_tire_n = tire_model(
        "MFSimple",
        vhl_states.wheel_rl.sigma_x,
        vhl_forces.wheel_rl.force_z_n,
        tire_params_set.wheel_rl_x,
    )
    force_x_rr_tire_n = tire_model(
        "MFSimple",
        vhl_states.wheel_rr.sigma_x,
        vhl_forces.wheel_rr.force_z_n,
        tire_params_set.wheel_rr_x,
    )
    force_x_front_body_n = jnp.cos(delta_f_rad) * (force_x_fl_tire_n + force_x_fr_tire_n)
    force_x_rear_body_n = force_x_rl_tire_n + force_x_rr_tire_n
    return force_x_front_body_n + force_x_rear_body_n


def smooth_signal(signal: jnp.array, window_length: int, polyorder: int) -> jnp.array:
    """Apply a safe Savitzky-Golay smoothing to a 1D signal."""
    signal_np = np.asarray(signal, dtype=float)
    if signal_np.size <= polyorder + 1:
        return jnp.array(signal_np)
    window_length = min(window_length, signal_np.size if signal_np.size % 2 == 1 else signal_np.size - 1)
    if window_length <= polyorder:
        window_length = polyorder + 1
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder or window_length < 3:
        return jnp.array(signal_np)
    return jnp.array(savgol_filter(signal_np, window_length=window_length, polyorder=polyorder, mode="nearest"))


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Return a zero-mean, unit-variance view of a signal for correlation."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    signal = signal - np.mean(signal)
    scale = np.std(signal)
    if scale < 1.0e-9:
        return np.zeros_like(signal)
    return signal / scale


def _estimate_sample_shift(reference: np.ndarray, target: np.ndarray, max_shift_samples: int) -> int:
    """Estimate integer sample delay between two signals using bounded cross-correlation."""
    reference_norm = _normalize_signal(reference)
    target_norm = _normalize_signal(target)
    if reference_norm.size == 0 or target_norm.size == 0:
        return 0
    corr = np.correlate(target_norm, reference_norm, mode="full")
    lags = np.arange(-len(reference_norm) + 1, len(target_norm))
    valid = np.abs(lags) <= max_shift_samples
    if not np.any(valid):
        return 0
    return int(lags[valid][int(np.argmax(corr[valid]))])


def _shift_signal(signal: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift a 1D signal by an integer number of samples using linear interpolation."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0 or shift_samples == 0:
        return signal
    sample_idx = np.arange(signal.size, dtype=float)
    return np.interp(sample_idx + shift_samples, sample_idx, signal, left=signal[0], right=signal[-1])


def align_filtered_data_signals(filtered: FilteredData, max_shift_s: float = 0.25) -> list[tuple[str, int, float]]:
    """Align key multi-sensor signals with bounded sample shifts and report the applied delays."""
    time_s = np.asarray(filtered.gen_data.time, dtype=float)
    if time_s.size < 3:
        return []

    dt_s = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return []
    max_shift_samples = max(1, int(round(max_shift_s / dt_s)))

    alignment_specs = [
        (
            "imu_data.yaw_rate_radps",
            np.asarray(filtered.imu_data.yaw_rate_radps, dtype=float),
            "imu_data.acc_cog_y_mps2",
            np.asarray(filtered.imu_data.acc_cog_y_mps2, dtype=float),
            False,
        ),
        (
            "imu_data.yaw_rate_radps",
            np.asarray(filtered.imu_data.yaw_rate_radps, dtype=float),
            "cor_data.vel_cog_y_mps",
            np.asarray(filtered.cor_data.vel_cog_y_mps, dtype=float),
            False,
        ),
        (
            "imu_data.yaw_rate_radps",
            np.asarray(filtered.imu_data.yaw_rate_radps, dtype=float),
            "gen_data.delta_f_rad",
            np.asarray(filtered.gen_data.delta_f_rad, dtype=float),
            True,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.gradient(np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float), time_s),
            "imu_data.acc_cog_x_mps2",
            np.asarray(filtered.imu_data.acc_cog_x_mps2, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_wheel_fl_radps",
            np.asarray(filtered.gen_data.omega_wheel_fl_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_wheel_fr_radps",
            np.asarray(filtered.gen_data.omega_wheel_fr_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_wheel_rl_radps",
            np.asarray(filtered.gen_data.omega_wheel_rl_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_wheel_rr_radps",
            np.asarray(filtered.gen_data.omega_wheel_rr_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_m_fl_radps",
            np.asarray(filtered.gen_data.omega_m_fl_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_m_fr_radps",
            np.asarray(filtered.gen_data.omega_m_fr_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_m_rl_radps",
            np.asarray(filtered.gen_data.omega_m_rl_radps, dtype=float),
            False,
        ),
        (
            "cor_data.vel_cog_x_mps",
            np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
            "gen_data.omega_m_rr_radps",
            np.asarray(filtered.gen_data.omega_m_rr_radps, dtype=float),
            False,
        ),
    ]

    applied_shifts: list[tuple[str, int, float]] = []
    for reference_name, reference_signal, target_name, target_signal, use_gradient in alignment_specs:
        reference_for_corr = np.gradient(reference_signal, time_s) if use_gradient else reference_signal
        target_for_corr = np.gradient(target_signal, time_s) if use_gradient else target_signal
        shift_samples = _estimate_sample_shift(reference_for_corr, target_for_corr, max_shift_samples)
        shifted_signal = _shift_signal(target_signal, shift_samples)

        section_name, field_name = target_name.split(".", maxsplit=1)
        section = getattr(filtered, section_name)
        setattr(section, field_name, jnp.array(shifted_signal))
        applied_shifts.append((f"{target_name} aligned to {reference_name}", shift_samples, shift_samples * dt_s))

    return applied_shifts


def _signal_has_new_sample(signal: np.ndarray) -> np.ndarray:
    """Return a mask that is true where the signal changed compared to the previous sample."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.array([], dtype=bool)
    if signal.size == 1:
        return np.array([True], dtype=bool)

    diffs = np.abs(np.diff(signal))
    scale = max(float(np.nanmax(np.abs(signal))), 1.0)
    threshold = 1.0e-6 * scale
    updated = np.concatenate(([True], diffs > threshold))
    return updated


def keep_only_fresh_measurement_rows(filtered: FilteredData) -> tuple[FilteredData, dict[str, int]]:
    """Drop rows where monitored signals are held/repeated instead of carrying a new sample."""
    tracked_signals = {
        "cor_data.vel_cog_x_mps": np.asarray(filtered.cor_data.vel_cog_x_mps, dtype=float),
        "cor_data.vel_cog_y_mps": np.asarray(filtered.cor_data.vel_cog_y_mps, dtype=float),
        "imu_data.acc_cog_x_mps2": np.asarray(filtered.imu_data.acc_cog_x_mps2, dtype=float),
        "imu_data.acc_cog_y_mps2": np.asarray(filtered.imu_data.acc_cog_y_mps2, dtype=float),
        "imu_data.yaw_rate_radps": np.asarray(filtered.imu_data.yaw_rate_radps, dtype=float),
        "gen_data.delta_f_rad": np.asarray(filtered.gen_data.delta_f_rad, dtype=float),
        "gen_data.omega_wheel_fl_radps": np.asarray(filtered.gen_data.omega_wheel_fl_radps, dtype=float),
        "gen_data.omega_wheel_fr_radps": np.asarray(filtered.gen_data.omega_wheel_fr_radps, dtype=float),
        "gen_data.omega_wheel_rl_radps": np.asarray(filtered.gen_data.omega_wheel_rl_radps, dtype=float),
        "gen_data.omega_wheel_rr_radps": np.asarray(filtered.gen_data.omega_wheel_rr_radps, dtype=float),
        "gen_data.omega_m_fl_radps": np.asarray(filtered.gen_data.omega_m_fl_radps, dtype=float),
        "gen_data.omega_m_fr_radps": np.asarray(filtered.gen_data.omega_m_fr_radps, dtype=float),
        "gen_data.omega_m_rl_radps": np.asarray(filtered.gen_data.omega_m_rl_radps, dtype=float),
        "gen_data.omega_m_rr_radps": np.asarray(filtered.gen_data.omega_m_rr_radps, dtype=float),
    }
    if not tracked_signals:
        return filtered, {}

    per_signal_updates = {
        name: _signal_has_new_sample(signal) for name, signal in tracked_signals.items()
    }
    keep_mask = np.logical_and.reduce(list(per_signal_updates.values()))
    if keep_mask.size == 0:
        return filtered, {}
    keep_mask[0] = True

    if np.all(keep_mask):
        return filtered, {name: 0 for name in tracked_signals}

    for section_name in ("cor_data", "imu_data", "gen_data"):
        section = getattr(filtered, section_name)
        for field in fields(section):
            values = np.asarray(getattr(section, field.name))
            if values.ndim == 1 and values.shape[0] == keep_mask.shape[0]:
                setattr(section, field.name, jnp.array(values[keep_mask]))

    dropped_counts = {
        name: int(np.count_nonzero(~update_mask))
        for name, update_mask in per_signal_updates.items()
    }
    dropped_counts["rows_removed_total"] = int(np.count_nonzero(~keep_mask))
    dropped_counts["rows_kept_total"] = int(np.count_nonzero(keep_mask))
    return filtered, dropped_counts


def plot_lateral_estimation(sensordata: FilteredData, vhl_states, vhl_forces, vhl_params,
                            tire_params_set: STMTireParams) -> None:
    """Visualize measured vs Pacejka-estimated total lateral force."""
    time_s = sensordata.gen_data.time
    measured_total_lateral_force_n = vhl_params.mass_kg * sensordata.imu_data.acc_cog_y_mps2
    estimated_total_lateral_force_n = calc_total_lateral_force_body_n_from_pacejka(
        sensordata, vhl_states, vhl_forces, tire_params_set
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    ax.plot(time_s, measured_total_lateral_force_n, label="Measured m * ay", color="#0065BD")
    ax.plot(time_s, estimated_total_lateral_force_n, label="Pacejka-estimated total Fy", color="#E37222")
    ax.set_title("Measured vs Pacejka-estimated Total Lateral Force")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_longitudinal_estimation(sensordata: FilteredData, vhl_states, vhl_forces, vhl_params,
                                 tire_params_set: STMTireParams | None = None,
                                 include_pacejka: bool = True) -> None:
    """Visualize total longitudinal force from acceleration, torque-based wheel forces, and optional tire models."""
    time_s = sensordata.gen_data.time
    measured_total_longitudinal_force_n = vhl_params.mass_kg * sensordata.imu_data.acc_cog_x_mps2
    torque_based_total_longitudinal_force_n = calc_total_longitudinal_force_body_n_from_wheel_forces(
        sensordata, vhl_forces
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True)
    ax.plot(time_s, measured_total_longitudinal_force_n, label="Measured m * ax", color="#0065BD")
    ax.plot(
        time_s,
        torque_based_total_longitudinal_force_n,
        label="Sum Fx from torque and angular acceleration",
        color="#E37222",
    )
    if include_pacejka and tire_params_set is not None:
        pacejka_total_longitudinal_force_n = calc_total_longitudinal_force_body_n_from_pacejka(
            sensordata, vhl_states, vhl_forces, tire_params_set
        )
        ax.plot(
            time_s,
            pacejka_total_longitudinal_force_n,
            label="Sum Fx from slip ratio + Fz + fitted tire models",
            color="#A2AD00",
        )
    ax.set_title("Measured vs Estimated Total Longitudinal Force")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _message_time_s(msg, fallback_timestamp_ns: int) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = float(getattr(stamp, "sec", 0.0))
        nanosec = float(getattr(stamp, "nanosec", 0.0))
        if sec != 0.0 or nanosec != 0.0:
            return sec + nanosec * 1.0e-9
    return float(fallback_timestamp_ns) * 1.0e-9


def _median_sample_time_s(time_s: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float).ravel()
    if time_s.size < 2:
        raise ValueError("At least two samples are required to determine sampling time.")
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        raise ValueError("Time vector does not contain positive finite sample intervals.")
    return float(np.median(diffs))


def _odd_window_length(candidate: int, signal_size: int, poly_order: int) -> int:
    if signal_size <= poly_order + 1:
        return signal_size
    window_length = max(int(candidate), poly_order + 2, 3)
    if window_length % 2 == 0:
        window_length += 1
    max_window = signal_size if signal_size % 2 == 1 else signal_size - 1
    window_length = min(window_length, max_window)
    if window_length <= poly_order:
        window_length = poly_order + 1
        if window_length % 2 == 0:
            window_length += 1
    return max(window_length, 3)


def _filter_settings_for_time(filter_type: str, settings: dict, time_s: np.ndarray,
                              nominal_sampling_freq_hz: float, signal_size: int) -> dict:
    scaled_settings = dict(settings)
    dt_s = _median_sample_time_s(time_s)
    actual_sampling_freq_hz = 1.0 / dt_s
    if filter_type == "savgol" and "window_length" in scaled_settings:
        nominal_sampling_freq_hz = float(nominal_sampling_freq_hz)
        configured_window_length = int(scaled_settings["window_length"])
        if nominal_sampling_freq_hz > 0.0:
            window_duration_s = configured_window_length / nominal_sampling_freq_hz
            window_length = int(round(window_duration_s * actual_sampling_freq_hz))
        else:
            window_length = configured_window_length
        scaled_settings["window_length"] = _odd_window_length(
            window_length,
            signal_size,
            int(scaled_settings.get("order", 0)),
        )
    elif filter_type == "butterworth":
        scaled_settings["sampling_freq"] = actual_sampling_freq_hz
    return scaled_settings


def _filter_data_for_time(filter_type: str, settings: dict, data, time_s: np.ndarray, conf):
    signal_size = len(np.asarray(time_s).ravel())
    if signal_size < 3:
        return data
    if filter_type == "savgol" and signal_size <= int(settings.get("order", 0)) + 1:
        return data
    scaled_settings = _filter_settings_for_time(
        filter_type,
        settings,
        time_s,
        getattr(conf, "sampling_freq_hz", 0.0),
        signal_size,
    )
    return fitler_data(filter_type, scaled_settings, data)


def _copy_masked_sections(data_cor: CorData, data_imu: ImuData, data_gen: GenData, mask: np.ndarray) -> FilteredData:
    filtered = FilteredData()
    for section_name, source in (
        ("cor_data", data_cor),
        ("imu_data", data_imu),
        ("gen_data", data_gen),
    ):
        destination = getattr(filtered, section_name)
        for field in fields(source):
            values = np.asarray(getattr(source, field.name))
            if values.ndim == 1 and values.shape[0] == mask.shape[0]:
                setattr(destination, field.name, jnp.array(values[mask]))
    return filtered


def _required_input_matrix(data_cor: CorData, data_imu: ImuData, data_gen: GenData,
                           include_motor_speeds: bool) -> np.ndarray:
    required_signals = [
        np.asarray(data_cor.time),
        np.asarray(data_cor.vel_cog_x_mps),
        np.asarray(data_cor.vel_cog_y_mps),
        np.asarray(data_imu.acc_cog_x_mps2),
        np.asarray(data_imu.acc_cog_y_mps2),
        np.asarray(data_imu.yaw_rate_radps),
        np.asarray(data_gen.delta_f_rad),
        np.asarray(data_gen.omega_wheel_fl_radps),
        np.asarray(data_gen.omega_wheel_fr_radps),
        np.asarray(data_gen.omega_wheel_rl_radps),
        np.asarray(data_gen.omega_wheel_rr_radps),
        np.asarray(data_gen.t_m_fl_nm),
        np.asarray(data_gen.t_m_fr_nm),
        np.asarray(data_gen.t_m_rl_nm),
        np.asarray(data_gen.t_m_rr_nm),
    ]
    if include_motor_speeds:
        required_signals.extend(
            [
                np.asarray(data_gen.omega_m_fl_radps),
                np.asarray(data_gen.omega_m_fr_radps),
                np.asarray(data_gen.omega_m_rl_radps),
                np.asarray(data_gen.omega_m_rr_radps),
            ]
        )
    return np.column_stack(required_signals)


def _finalize_filtered_sections(data_cor: CorData, data_imu: ImuData, data_gen: GenData, conf,
                                *, include_motor_speeds: bool = True,
                                align_signals: bool = True,
                                filter_fresh_measurements: bool = True) -> FilteredData:
    data_cor = vel_offset_correction(conf.lambda_v, data_cor)
    data_imu = imu_offset_correction(
        conf.ax_median,
        conf.ay_median,
        conf.lambda_ax,
        conf.lambda_ay,
        conf.yaw_rate_off,
        conf.az_off,
        data_imu,
    )
    data_gen = _filter_data_for_time(conf.filter_gen, conf.settings_gen, data_gen, data_gen.time, conf)
    data_cor = _filter_data_for_time(conf.filter_cor, conf.settings_cor, data_cor, data_cor.time, conf)
    data_imu = _filter_data_for_time(conf.filter_imu, conf.settings_imu, data_imu, data_imu.time, conf)
    if conf.imu_acc_z_fix:
        data_imu.acc_cog_z_mps2 = jnp.ones_like(data_imu.acc_cog_z_mps2) * 9.81

    finite_mask = np.isfinite(
        _required_input_matrix(data_cor, data_imu, data_gen, include_motor_speeds)
    ).all(axis=1)
    speed_mask = np.asarray(data_cor.vel_cog_x_mps) > conf.low_speed_filter_mps
    filtered = _copy_masked_sections(data_cor, data_imu, data_gen, finite_mask & speed_mask)

    validate_required_sensor_signals(
        filtered,
        include_motor_speeds=include_motor_speeds,
        include_force_inputs=True,
    )

    if align_signals:
        applied_shifts = align_filtered_data_signals(filtered)
        if applied_shifts:
            print("-" * 80)
            print("Signal alignment summary")
            for description, shift_samples, shift_seconds in applied_shifts:
                print(f"{description:<60} shift={shift_samples:+4d} samples ({shift_seconds:+.4f} s)")

    if filter_fresh_measurements:
        filtered, freshness_summary = keep_only_fresh_measurement_rows(filtered)
        if freshness_summary:
            print("-" * 80)
            print("Fresh-measurement row filtering")
            print(
                f"Kept {freshness_summary['rows_kept_total']} rows, "
                f"removed {freshness_summary['rows_removed_total']} rows."
            )
            for signal_name, removed_count in freshness_summary.items():
                if signal_name.startswith("rows_"):
                    continue
                print(f"{signal_name:<60} repeated_rows={removed_count}")

    validate_required_sensor_signals(
        filtered,
        include_motor_speeds=include_motor_speeds,
        include_force_inputs=True,
    )
    return filtered


def _detect_rosbag_storage_id(uri: Path, storage_id: str | None) -> str:
    if storage_id:
        return storage_id
    if uri.is_file():
        if uri.suffix == ".mcap":
            return "mcap"
        if uri.suffix == ".db3":
            return "sqlite3"
    if uri.is_dir():
        for child in uri.iterdir():
            if child.suffix == ".mcap":
                return "mcap"
            if child.suffix == ".db3":
                return "sqlite3"
    return ""


def _validate_amz_rosbag_topics(topic_types: dict[str, str]) -> dict[str, str]:
    missing_topics = [
        spec.topic
        for spec in AMZ_ROSBAG_TOPICS.values()
        if spec.topic not in topic_types
    ]
    if missing_topics:
        available = "\n".join(f"{topic}: {type_name}" for topic, type_name in sorted(topic_types.items()))
        raise ValueError(
            "The AMZ MCAP parser expects fixed VCU topics from the reference bag. "
            f"Missing topics: {', '.join(missing_topics)}. Available topics:\n{available}"
        )

    mismatched_types = [
        f"{spec.topic}: expected {spec.type_name}, found {topic_types[spec.topic]}"
        for spec in AMZ_ROSBAG_TOPICS.values()
        if topic_types[spec.topic] != spec.type_name
    ]
    if mismatched_types:
        raise ValueError(
            "The AMZ MCAP parser expects the message types from the reference bag:\n"
            + "\n".join(mismatched_types)
        )

    return {role: spec.topic for role, spec in AMZ_ROSBAG_TOPICS.items()}


def _extract_amz_rosbag_values(role: str, msg) -> dict[str, float]:
    if role == "velocity":
        return {
            "vel_x": float(msg.velocities.x),
            "vel_y": float(msg.velocities.y),
            "yaw_rate": float(msg.velocities.theta),
            "acc_x": float(msg.accelerations.x),
            "acc_y": float(msg.accelerations.y),
        }

    if role == "torque":
        return {
            "feedback_t_m_fl": float(msg.feedback_t_m_fl),
            "feedback_t_m_fr": float(msg.feedback_t_m_fr),
            "feedback_t_m_rl": float(msg.feedback_t_m_rl),
            "feedback_t_m_rr": float(msg.feedback_t_m_rr),
            "reference_t_m_fl": float(msg.reference_t_m_fl),
            "reference_t_m_fr": float(msg.reference_t_m_fr),
            "reference_t_m_rl": float(msg.reference_t_m_rl),
            "reference_t_m_rr": float(msg.reference_t_m_rr),
        }

    if role == "steering":
        return {"steer": float(msg.data)}

    raise ValueError(f"Unknown AMZ ROS bag role: {role}")


def _series_from_rosbag_samples(samples: list[tuple[float, dict[str, float]]], role: str) -> RosbagSignalSeries:
    if not samples:
        raise ValueError(f"No usable {role} samples were found in the bag.")
    rows_by_time = {}
    keys = set()
    for time_s, values in samples:
        if not np.isfinite(time_s):
            continue
        rows_by_time[float(time_s)] = values
        keys.update(values)
    if not rows_by_time:
        raise ValueError(f"No finite {role} timestamps were found in the bag.")
    time_s = np.array(sorted(rows_by_time), dtype=float)
    series_values = {
        key: np.array([rows_by_time[time].get(key, np.nan) for time in time_s], dtype=float)
        for key in sorted(keys)
    }
    finite_mask = np.isfinite(time_s)
    for values in series_values.values():
        finite_mask &= np.isfinite(values)
    time_s = time_s[finite_mask]
    series_values = {key: values[finite_mask] for key, values in series_values.items()}
    if time_s.size < 2:
        raise ValueError(f"At least two finite {role} samples are required.")
    return RosbagSignalSeries(time_s=time_s, values=series_values)


def _read_rosbag_signal_series(data_file: Path,
                               storage_id: str | None = None) -> tuple[dict[str, RosbagSignalSeries], dict[str, str]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise ImportError(
            "ROS bag input requires ROS 2 Python packages. Source the ROS environment "
            "and make sure rosbag2_py, rclpy, and rosidl_runtime_py are importable."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=str(data_file),
        storage_id=_detect_rosbag_storage_id(data_file, storage_id),
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    resolved_topics = _validate_amz_rosbag_topics(topic_types)
    role_by_topic = {topic: role for role, topic in resolved_topics.items()}
    try:
        msg_classes = {
            topic: get_message(topic_types[topic])
            for topic in role_by_topic
        }
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        required_types = ", ".join(sorted({topic_types[topic] for topic in role_by_topic}))
        raise ImportError(
            "ROS bag input can open the bag, but cannot load the generated Python "
            "message classes needed to deserialize it. Source a colcon overlay that "
            "contains these message packages and was built with the same ROS "
            f"distribution and Python ABI as this process. Required message types: {required_types}."
        ) from exc

    samples_by_role = {role: [] for role in resolved_topics}
    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()
        role = role_by_topic.get(topic)
        if role is None:
            continue
        try:
            msg = deserialize_message(serialized_data, msg_classes[topic])
        except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise ImportError(
                "ROS bag input loaded the message class but failed to deserialize "
                f"topic '{topic}' of type '{topic_types[topic]}'. This usually means "
                "the generated ROS Python bindings were built for a different ROS "
                "distribution or Python version than the process running the fit script."
            ) from exc
        values = _extract_amz_rosbag_values(role, msg)
        samples_by_role[role].append((_message_time_s(msg, timestamp_ns), values))

    return (
        {role: _series_from_rosbag_samples(samples, role) for role, samples in samples_by_role.items()},
        resolved_topics,
    )


def _interpolate_series(series: RosbagSignalSeries, target_time_s: np.ndarray,
                        field_names: tuple[str, ...], label: str) -> dict[str, np.ndarray]:
    values = {}
    for field_name in field_names:
        if field_name not in series.values:
            raise ValueError(f"Required field '{field_name}' was not found in {label} samples.")
        values[field_name] = np.interp(target_time_s, series.time_s, series.values[field_name])
    return values


def _choose_rosbag_torque_source(torque_values: dict[str, np.ndarray], requested_source: str) -> str:
    if requested_source not in {"auto", "feedback", "reference"}:
        raise ValueError("rosbag torque source must be 'auto', 'feedback', or 'reference'.")
    if requested_source != "auto":
        return requested_source

    feedback_keys = ("feedback_t_m_fl", "feedback_t_m_fr", "feedback_t_m_rl", "feedback_t_m_rr")
    reference_keys = ("reference_t_m_fl", "reference_t_m_fr", "reference_t_m_rl", "reference_t_m_rr")
    feedback_has_signal = any(np.any(np.abs(torque_values[key]) > 1.0e-9) for key in feedback_keys)
    reference_has_signal = any(np.any(np.abs(torque_values[key]) > 1.0e-9) for key in reference_keys)
    if feedback_has_signal:
        return "feedback"
    if reference_has_signal:
        return "reference"
    return "feedback"


def mat_array(mat_data: dict, *names: str, default: np.ndarray | None = None) -> np.ndarray:
    """Return the first available signal as a flat float array."""
    for name in names:
        if name in mat_data:
            return np.asarray(mat_data[name], dtype=float).squeeze()
    if default is not None:
        return np.asarray(default, dtype=float).squeeze()
    raise KeyError(f"None of the signals were found: {names}")


def build_filtered_data(data_file: Path, conf, vhl_params) -> FilteredData:
    """Map AMZ run data into the estimator's filtered sensor dataclass."""
    data_mat = loadmat(data_file, squeeze_me=True, struct_as_record=False)

    time = mat_array(data_mat, "Time")
    vel_x = mat_array(data_mat, "v_X_VE", "v_X_OVS", "V_x")
    vel_y = mat_array(data_mat, "v_Y_VE", "v_Y_OVS")
    yaw_rate = mat_array(data_mat, "omega_Z_VE", "omega_Z_INS", "omega_Z_OVS")
    acc_x = mat_array(data_mat, "a_X_VE", "a_X_OVS", "a_X_INS", "a_x_VE_direct")
    acc_y = mat_array(data_mat, "a_Y_VE", "a_Y_OVS", "a_Y_INS")
    # steer = mat_array(data_mat, "steering_target_rad")
    steer_fr = mat_array(data_mat, "delta_W_FR")
    steer_fl = mat_array(data_mat, "delta_W_FL")
    steer = (steer_fr + steer_fl) / 2

    omega_m_fl = mat_array(data_mat, "omega_M_FL")
    omega_m_fr = mat_array(data_mat, "omega_M_FR")
    omega_m_rl = mat_array(data_mat, "omega_M_RL")
    omega_m_rr = mat_array(data_mat, "omega_M_RR")
    t_m_fl = mat_array(data_mat, "T_M_FL")
    t_m_fr = mat_array(data_mat, "T_M_FR")
    t_m_rl = mat_array(data_mat, "T_M_RL")
    t_m_rr = mat_array(data_mat, "T_M_RR")
    safe_gear_ratio = max(abs(float(vhl_params.gear_ratio)), 1.0e-6)

    lengths = [
        len(signal)
        for signal in (
            time,
            vel_x,
            vel_y,
            acc_x,
            acc_y,
            yaw_rate,
            steer,
            omega_m_fl,
            omega_m_fr,
            omega_m_rl,
            omega_m_rr,
            t_m_fl,
            t_m_fr,
            t_m_rl,
            t_m_rr,
        )
    ]
    size = min(lengths)
    time = time[:size] - time[0]
    vel_x = vel_x[:size]
    vel_y = vel_y[:size]
    acc_x = acc_x[:size]
    acc_y = acc_y[:size]
    yaw_rate = yaw_rate[:size]
    steer = steer[:size]
    omega_m_fl = omega_m_fl[:size]
    omega_m_fr = omega_m_fr[:size]
    omega_m_rl = omega_m_rl[:size]
    omega_m_rr = omega_m_rr[:size]
    t_m_fl = t_m_fl[:size]
    t_m_fr = t_m_fr[:size]
    t_m_rl = t_m_rl[:size]
    t_m_rr = t_m_rr[:size]
    omega_fl = omega_m_fl / safe_gear_ratio
    omega_fr = omega_m_fr / safe_gear_ratio
    omega_rl = omega_m_rl / safe_gear_ratio
    omega_rr = omega_m_rr / safe_gear_ratio

    data_cor = CorData(
        time=jnp.array(time),
        vel_cog_x_mps=jnp.array(vel_x),
        vel_cog_y_mps=jnp.array(vel_y),
    )
    data_imu = ImuData(
        time=jnp.array(time),
        acc_cog_x_mps2=jnp.array(acc_x),
        acc_cog_y_mps2=jnp.array(acc_y),
        acc_cog_z_mps2=jnp.ones(size) * 9.81,
        yaw_rate_radps=jnp.array(yaw_rate),
    )
    data_gen = GenData(
        time=jnp.array(time),
        delta_f_rad=jnp.array(steer),
        omega_wheel_fl_radps=jnp.array(omega_fl),
        omega_wheel_fr_radps=jnp.array(omega_fr),
        omega_wheel_rl_radps=jnp.array(omega_rl),
        omega_wheel_rr_radps=jnp.array(omega_rr),
        omega_m_fl_radps=jnp.array(omega_m_fl),
        omega_m_fr_radps=jnp.array(omega_m_fr),
        omega_m_rl_radps=jnp.array(omega_m_rl),
        omega_m_rr_radps=jnp.array(omega_m_rr),
        t_m_fl_nm=jnp.array(t_m_fl),
        t_m_fr_nm=jnp.array(t_m_fr),
        t_m_rl_nm=jnp.array(t_m_rl),
        t_m_rr_nm=jnp.array(t_m_rr),
        gear=jnp.zeros(size),
    )

    return _finalize_filtered_sections(
        data_cor,
        data_imu,
        data_gen,
        conf,
        include_motor_speeds=True,
        align_signals=True,
        filter_fresh_measurements=True,
    )


def build_filtered_data_from_rosbag(data_file: Path, conf, vhl_params, *,
                                    storage_id: str | None = None,
                                    torque_source: str = "auto") -> FilteredData:
    """Map AMZ ROS 2 MCAP data into the estimator's filtered sensor dataclass.

    MCAP logs currently do not contain wheel-speed feedback. The builder
    synthesizes rolling wheel speeds from vehicle velocity only to keep lateral
    tire-state calculations well-defined; longitudinal tire fitting is rejected
    because real longitudinal slip is unavailable.
    """
    if bool(getattr(conf, "fit_longitudinal", False)):
        raise ValueError(
            "Longitudinal tire fitting is not available for ROS bag inputs because "
            "the bags do not contain wheel-speed feedback."
        )

    series_by_role, resolved_topics = _read_rosbag_signal_series(data_file, storage_id)
    setattr(conf, "rosbag_topics", resolved_topics)

    velocity_series = series_by_role["velocity"]
    torque_series = series_by_role["torque"]
    steering_series = series_by_role["steering"]
    overlap_start_s = max(
        float(velocity_series.time_s[0]),
        float(torque_series.time_s[0]),
        float(steering_series.time_s[0]),
    )
    overlap_end_s = min(
        float(velocity_series.time_s[-1]),
        float(torque_series.time_s[-1]),
        float(steering_series.time_s[-1]),
    )
    if overlap_end_s <= overlap_start_s:
        raise ValueError("ROS bag topics do not have an overlapping time interval.")

    base_mask = (velocity_series.time_s >= overlap_start_s) & (velocity_series.time_s <= overlap_end_s)
    base_time_abs_s = velocity_series.time_s[base_mask]
    if base_time_abs_s.size < 2:
        dt_s = _median_sample_time_s(velocity_series.time_s)
        base_time_abs_s = np.arange(overlap_start_s, overlap_end_s + 0.5 * dt_s, dt_s)
    if base_time_abs_s.size < 2:
        raise ValueError("ROS bag overlap interval does not contain enough samples.")

    velocity_values = _interpolate_series(
        velocity_series,
        base_time_abs_s,
        ("vel_x", "vel_y", "yaw_rate"),
        "velocity",
    )
    vel_x = velocity_values["vel_x"]
    vel_y = velocity_values["vel_y"]
    yaw_rate = velocity_values["yaw_rate"]
    if "acc_x" in velocity_series.values and "acc_y" in velocity_series.values:
        acceleration_values = _interpolate_series(
            velocity_series,
            base_time_abs_s,
            ("acc_x", "acc_y"),
            "velocity",
        )
        acc_x = acceleration_values["acc_x"]
        acc_y = acceleration_values["acc_y"]
    else:
        acc_x = np.gradient(vel_x, base_time_abs_s)
        acc_y = np.gradient(vel_y, base_time_abs_s)

    steering_values = _interpolate_series(steering_series, base_time_abs_s, ("steer",), "steering")
    steer = steering_values["steer"]

    torque_values = _interpolate_series(
        torque_series,
        base_time_abs_s,
        (
            "feedback_t_m_fl",
            "feedback_t_m_fr",
            "feedback_t_m_rl",
            "feedback_t_m_rr",
            "reference_t_m_fl",
            "reference_t_m_fr",
            "reference_t_m_rl",
            "reference_t_m_rr",
        ),
        "torque",
    )
    selected_torque_source = _choose_rosbag_torque_source(torque_values, torque_source)
    setattr(conf, "rosbag_torque_source_requested", torque_source)
    setattr(conf, "rosbag_torque_source_used", selected_torque_source)
    setattr(conf, "longitudinal_force_mode", "ideal_torque")

    time = base_time_abs_s - base_time_abs_s[0]
    size = len(time)
    safe_front_radius_m = max(abs(float(vhl_params.r_tire_unloaded_front_m)), 1.0e-6)
    safe_rear_radius_m = max(abs(float(vhl_params.r_tire_unloaded_rear_m)), 1.0e-6)
    safe_gear_ratio = max(abs(float(vhl_params.gear_ratio)), 1.0e-6)

    vel_y_front_mps = vel_y + yaw_rate * vhl_params.l_front_m
    vel_y_rear_mps = vel_y - yaw_rate * vhl_params.l_rear_m
    vel_x_front_left_mps = vel_x - yaw_rate * vhl_params.tw_front_m / 2
    vel_x_front_right_mps = vel_x + yaw_rate * vhl_params.tw_front_m / 2
    vel_x_rear_left_mps = vel_x - yaw_rate * vhl_params.tw_rear_m / 2
    vel_x_rear_right_mps = vel_x + yaw_rate * vhl_params.tw_rear_m / 2
    omega_fl = (
        np.cos(steer) * vel_x_front_left_mps + np.sin(steer) * vel_y_front_mps
    ) / safe_front_radius_m
    omega_fr = (
        np.cos(steer) * vel_x_front_right_mps + np.sin(steer) * vel_y_front_mps
    ) / safe_front_radius_m
    omega_rl = vel_x_rear_left_mps / safe_rear_radius_m
    omega_rr = vel_x_rear_right_mps / safe_rear_radius_m

    data_cor = CorData(
        time=jnp.array(time),
        vel_cog_x_mps=jnp.array(vel_x),
        vel_cog_y_mps=jnp.array(vel_y),
    )
    data_imu = ImuData(
        time=jnp.array(time),
        acc_cog_x_mps2=jnp.array(acc_x),
        acc_cog_y_mps2=jnp.array(acc_y),
        acc_cog_z_mps2=jnp.ones(size) * 9.81,
        yaw_rate_radps=jnp.array(yaw_rate),
    )
    data_gen = GenData(
        time=jnp.array(time),
        delta_f_rad=jnp.array(steer),
        omega_wheel_fl_radps=jnp.array(omega_fl),
        omega_wheel_fr_radps=jnp.array(omega_fr),
        omega_wheel_rl_radps=jnp.array(omega_rl),
        omega_wheel_rr_radps=jnp.array(omega_rr),
        omega_m_fl_radps=jnp.array(omega_fl * safe_gear_ratio),
        omega_m_fr_radps=jnp.array(omega_fr * safe_gear_ratio),
        omega_m_rl_radps=jnp.array(omega_rl * safe_gear_ratio),
        omega_m_rr_radps=jnp.array(omega_rr * safe_gear_ratio),
        t_m_fl_nm=jnp.array(torque_values[f"{selected_torque_source}_t_m_fl"]),
        t_m_fr_nm=jnp.array(torque_values[f"{selected_torque_source}_t_m_fr"]),
        t_m_rl_nm=jnp.array(torque_values[f"{selected_torque_source}_t_m_rl"]),
        t_m_rr_nm=jnp.array(torque_values[f"{selected_torque_source}_t_m_rr"]),
        gear=jnp.zeros(size),
    )

    return _finalize_filtered_sections(
        data_cor,
        data_imu,
        data_gen,
        conf,
        include_motor_speeds=False,
        align_signals=False,
        filter_fresh_measurements=False,
    )


def fit_tire_parameters(conf, sensordata, vhl_params=None):
    """Fit tire parameters for already loaded sensor data.

    Vehicle parameters can be passed directly by callers that manage all
    inputs externally. When omitted, conf.vehicle_parameter_names must point
    to an external TOML file.
    """
    params_min = MFSimpleParams(**conf.params_min)
    params_max = MFSimpleParams(**conf.params_max)
    params_init = MFSimpleParams(**conf.params_init)
    if vhl_params is None:
        vhl_params = load_params(str(conf.vehicle_parameter_names))

    vhl_states = calc_vhl_states(sensordata, vhl_params)
    vhl_forces = calc_vhl_forces(
        conf.model,
        sensordata,
        vhl_states,
        vhl_params,
        longitudinal_force_mode=getattr(conf, "longitudinal_force_mode", "wheel_dynamics"),
    )

    # Low pass filtering
    force_filter_type = 'savgol' # or 'butterworth', 'moving_average', 'gaussian'
    force_filter_settings = {
        'window_length': 51,  # Must be odd
        'order': 3,           # Polynomial order
        # If using butterworth, you'd use these instead:
        # 'sampling_freq': 100, 
        # 'filter_order': 4, 
        # 'freqs_upper': 10.0, 
        # 'butter_type': 'lowpass'
    }
    for field in fields(vhl_forces):
        sub_dataclass = getattr(vhl_forces, field.name)
        if is_dataclass(sub_dataclass):
            setattr(
                vhl_forces,
                field.name,
                _filter_data_for_time(
                    force_filter_type,
                    force_filter_settings,
                    sub_dataclass,
                    sensordata.gen_data.time,
                    conf,
                ),
            )

    delta_dot = jnp.gradient(sensordata.gen_data.delta_f_rad, sensordata.gen_data.time)

    # Transient rejection
    vhl_states, vhl_forces, steady_mask = reject_transient_data(
        sensordata, vhl_states, vhl_forces, 
        max_yaw_accel_radps2=4.0, 
        max_steer_vel_radps=1.0,
        return_mask=True,
    )
    delta_dot = jnp.array(np.asarray(delta_dot)[steady_mask])

    # Outlier rejection
    if conf.vhl_data_filter:
        vhl_states, vhl_forces, outlier_mask = filter_vhl_data(vhl_states, vhl_forces, 1.8, return_mask=True)
        delta_dot = jnp.array(np.asarray(delta_dot)[outlier_mask])

    tire_params_set_svi = STMTireParams()
    std_params_set_svi = STMTireParams()
    tire_params_set_nelder = STMTireParams()
    fit_excitation_samples = {}
    target_sample_points = max(conf.svi_options["sample_points"], conf.nelder_options["sample_points"])
    fit_longitudinal = bool(getattr(conf, "fit_longitudinal", False))
    fit_targets = selected_fit_targets(fit_longitudinal)
    if not fit_longitudinal:
        print("Longitudinal wheel tire fitting disabled; torque-based Fx is still calculated.")

    for state_key, direction in fit_targets:
        sigma_raw = jnp.array(getattr(getattr(vhl_states, state_key), f"sigma_{direction}"))
        force_n_raw = jnp.array(
            getattr(getattr(vhl_forces, state_key), f"force_{direction}_n")
        )
        load_n_raw = jnp.array(getattr(vhl_forces, state_key).force_z_n)
        fit_flags = getattr(conf, f"fit_flags_{state_key}_{direction}")
        param_key = f"{state_key}_{direction}"
        balanced_samples = select_balanced_fit_samples(
            sigma_raw,
            force_n_raw,
            load_n_raw,
            vhl_states.dd_psi,
            delta_dot,
            target_count=target_sample_points,
            tail_quantile=BALANCE_TAIL_QUANTILE,
            target_points_per_region=BALANCE_TARGET_POINTS_PER_REGION,
            max_regions=BALANCE_MAX_REGIONS,
        )
        sigma = jnp.array(balanced_samples["sigma"])
        force_n = jnp.array(balanced_samples["force_n"])
        load_n = jnp.array(balanced_samples["load_n"])
        fit_excitation_samples[param_key] = balanced_samples

        print(
            f"Balanced selection - {state_key} {direction}: "
            f"kept {len(sigma)}/{len(sigma_raw)} samples "
            f"across {len(balanced_samples['region_counts_after'])} regions"
        )
        print(f"Region occupancy before: {balanced_samples['region_counts_before']}")
        print(f"Region occupancy kept:   {balanced_samples['region_counts_after']}")

        try:
            shift_source = "balanced"
            shift_s_v, shift_s_h = calc_force_shift(sigma, force_n)
        except ValueError:
            shift_source = "raw"
            shift_s_v, shift_s_h = calc_force_shift(sigma_raw, force_n_raw)

        if direction == "y":
            unclipped_s_v = shift_s_v
            unclipped_s_h = shift_s_h
            shift_s_v, shift_s_h, shift_was_clipped = _clip_lateral_force_shift(
                shift_s_v, shift_s_h, params_min, params_max
            )
            if shift_was_clipped:
                print(
                    f"Lateral force shift clipped - {state_key}: "
                    f"S_V {unclipped_s_v:.6g} -> {shift_s_v:.6g}, "
                    f"S_H {unclipped_s_h:.6g} -> {shift_s_h:.6g}"
                )

        params_init.S_V = shift_s_v
        params_init.S_H = shift_s_h
        print(
            f"Force shift - {state_key} {direction} ({shift_source} samples): "
            f"S_V={params_init.S_V:.6g}, S_H={params_init.S_H:.6g}"
        )
        print(f"Fitting - {state_key} {direction}")
        svi_options = dict(conf.svi_options)
        nelder_options = dict(conf.nelder_options)
        svi_options["sample_points"] = len(sigma)
        nelder_options["sample_points"] = len(sigma)
        params_svi, std_svi, _ = tire_param_fitting(
            "SVI",
            "MFSimple",
            sigma,
            force_n,
            load_n,
            params_init,
            params_min,
            params_max,
            svi_options,
            fit_flags,
        )
        params_nelder, _, _ = tire_param_fitting(
            "Nelder",
            "MFSimple",
            sigma,
            force_n,
            load_n,
            params_init,
            params_min,
            params_max,
            nelder_options,
            fit_flags,
        )
        print("SVI:")
        print(params_svi)
        print(std_svi)
        print("Nelder:")
        print(params_nelder)

        setattr(tire_params_set_svi, param_key, params_svi)
        setattr(std_params_set_svi, param_key, std_svi)
        setattr(tire_params_set_nelder, param_key, params_nelder)

    return (
        vhl_params,
        vhl_states,
        vhl_forces,
        params_min,
        params_max,
        tire_params_set_svi,
        std_params_set_svi,
        tire_params_set_nelder,
        fit_excitation_samples,
    )
