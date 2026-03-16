"""Fit tire parameters from AMZ MATLAB data using the bundled estimator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter
from dataclasses import fields, is_dataclass

REPO_ROOT = Path(__file__).resolve().parents[1]
TIRE_LIB_ROOT = REPO_ROOT / "Tire_Parameter_and_Uncertainty_Estimation"
if str(TIRE_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(TIRE_LIB_ROOT))

from data_types.sensordata import CorData, FilteredData, GenData, ImuData
from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from src.param_fitting.param_fitting import calc_force_shift, tire_param_fitting
from src.utils.calc_vhl_states import calc_vhl_forces, calc_vhl_states
from src.utils.datamanager import load_config, load_params, save_dataclass_to_csv
from src.utils.evaluation_helpers import (
    eval_force_errors,
    plot_bell_curves,
    plot_tire_curves,
)
from src.utils.filter_data import (
    filter_vhl_data,
    fitler_data,
    imu_offset_correction,
    vel_offset_correction,
    reject_transient_data,
)
from src.utils.tiremodels import tire_model


DEFAULT_DATA_FILE = (
    REPO_ROOT
    / ".."
    / "inputs"
    / "August"
    # / "DV_trackdrive"
    # / "2025-08-05_18-08-02_run_21"
    # / "2025-08-05_18-08-02_DV_trackdrive_csv_test_v2_data.mat"
    / "EV_autoX"
    / "2025-08-23_15-42-21_FSG_autoX_luan"
    / "2025-08-23_15-42-21_EV_autox_FSG_autoX_luan_data.mat"
)
DEFAULT_CONFIG_FILE = TIRE_LIB_ROOT / "setup" / "config.toml"
DEFAULT_VEHICLE_PARAMS = TIRE_LIB_ROOT / "setup" / "vehicle_parameters.toml"
FIT_TARGETS = [
    ("wheel_fl", "x"),
    ("wheel_fr", "x"),
    ("wheel_rl", "x"),
    ("wheel_rr", "x"),
    ("front_axle", "y"),
    ("rear_axle", "y"),
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate tire parameters from AMZ .mat data."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to the AMZ *_data.mat file.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Estimator config TOML to use as a baseline.",
    )
    parser.add_argument(
        "--vehicle-params",
        type=Path,
        default=DEFAULT_VEHICLE_PARAMS,
        help="Vehicle parameter TOML file.",
    )
    parser.add_argument(
        "--output-folder",
        default="tire_params_amz_dv",
        help="Folder name created under outputs/.",
    )
    parser.add_argument(
        "--low-speed-filter",
        type=float,
        default=2.0,
        help="Discard samples below this longitudinal speed in m/s.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=1500,
        help="Sample points passed to the optimizers.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plots.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable CSV output logging.",
    )
    return parser.parse_args()


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
    acc_x = mat_array(data_mat, "a_X_VE", "a_X_OVS", "a_X_INS")
    acc_y = mat_array(data_mat, "a_Y_VE", "a_Y_OVS", "a_Y_INS")
    # steer = mat_array(data_mat, "steering_target_rad")
    steer_fr = np.asarray(data_mat["delta_W_FR"]).squeeze()
    steer_fl = np.asarray(data_mat["delta_W_FL"]).squeeze()
    steer = (steer_fr + steer_fl) / 2

    omega_m_fl = mat_array(data_mat, "omega_M_FL", default=np.zeros_like(time))
    omega_m_fr = mat_array(data_mat, "omega_M_FR", default=np.zeros_like(time))
    omega_m_rl = mat_array(data_mat, "omega_M_RL", default=np.zeros_like(time))
    omega_m_rr = mat_array(data_mat, "omega_M_RR", default=np.zeros_like(time))
    t_m_fl = mat_array(data_mat, "T_M_FL", default=np.zeros_like(time))
    t_m_fr = mat_array(data_mat, "T_M_FR", default=np.zeros_like(time))
    t_m_rl = mat_array(data_mat, "T_M_RL", default=np.zeros_like(time))
    t_m_rr = mat_array(data_mat, "T_M_RR", default=np.zeros_like(time))
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
    data_gen = fitler_data(conf.filter_gen, conf.settings_gen, data_gen)
    data_cor = fitler_data(conf.filter_cor, conf.settings_cor, data_cor)
    data_imu = fitler_data(conf.filter_imu, conf.settings_imu, data_imu)
    if conf.imu_acc_z_fix:
        data_imu.acc_cog_z_mps2 = jnp.ones_like(data_imu.acc_cog_z_mps2) * 9.81

    finite_mask = np.isfinite(
        np.column_stack(
            [
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
                np.asarray(data_gen.omega_m_fl_radps),
                np.asarray(data_gen.omega_m_fr_radps),
                np.asarray(data_gen.omega_m_rl_radps),
                np.asarray(data_gen.omega_m_rr_radps),
                np.asarray(data_gen.t_m_fl_nm),
                np.asarray(data_gen.t_m_fr_nm),
                np.asarray(data_gen.t_m_rl_nm),
                np.asarray(data_gen.t_m_rr_nm),
            ]
        )
    ).all(axis=1)
    speed_mask = np.asarray(data_cor.vel_cog_x_mps) > conf.low_speed_filter_mps
    mask = finite_mask & speed_mask

    filtered = FilteredData()
    filtered.cor_data.time = jnp.array(np.asarray(data_cor.time)[mask])
    filtered.cor_data.vel_cog_x_mps = jnp.array(np.asarray(data_cor.vel_cog_x_mps)[mask])
    filtered.cor_data.vel_cog_y_mps = jnp.array(np.asarray(data_cor.vel_cog_y_mps)[mask])

    filtered.imu_data.time = jnp.array(np.asarray(data_imu.time)[mask])
    filtered.imu_data.acc_cog_x_mps2 = jnp.array(np.asarray(data_imu.acc_cog_x_mps2)[mask])
    filtered.imu_data.acc_cog_y_mps2 = jnp.array(np.asarray(data_imu.acc_cog_y_mps2)[mask])
    filtered.imu_data.acc_cog_z_mps2 = jnp.array(np.asarray(data_imu.acc_cog_z_mps2)[mask])
    filtered.imu_data.yaw_rate_radps = jnp.array(np.asarray(data_imu.yaw_rate_radps)[mask])

    filtered.gen_data.time = jnp.array(np.asarray(data_gen.time)[mask])
    filtered.gen_data.delta_f_rad = jnp.array(np.asarray(data_gen.delta_f_rad)[mask])
    filtered.gen_data.omega_wheel_fl_radps = jnp.array(np.asarray(data_gen.omega_wheel_fl_radps)[mask])
    filtered.gen_data.omega_wheel_fr_radps = jnp.array(np.asarray(data_gen.omega_wheel_fr_radps)[mask])
    filtered.gen_data.omega_wheel_rl_radps = jnp.array(np.asarray(data_gen.omega_wheel_rl_radps)[mask])
    filtered.gen_data.omega_wheel_rr_radps = jnp.array(np.asarray(data_gen.omega_wheel_rr_radps)[mask])
    filtered.gen_data.omega_m_fl_radps = jnp.array(np.asarray(data_gen.omega_m_fl_radps)[mask])
    filtered.gen_data.omega_m_fr_radps = jnp.array(np.asarray(data_gen.omega_m_fr_radps)[mask])
    filtered.gen_data.omega_m_rl_radps = jnp.array(np.asarray(data_gen.omega_m_rl_radps)[mask])
    filtered.gen_data.omega_m_rr_radps = jnp.array(np.asarray(data_gen.omega_m_rr_radps)[mask])
    filtered.gen_data.t_m_fl_nm = jnp.array(np.asarray(data_gen.t_m_fl_nm)[mask])
    filtered.gen_data.t_m_fr_nm = jnp.array(np.asarray(data_gen.t_m_fr_nm)[mask])
    filtered.gen_data.t_m_rl_nm = jnp.array(np.asarray(data_gen.t_m_rl_nm)[mask])
    filtered.gen_data.t_m_rr_nm = jnp.array(np.asarray(data_gen.t_m_rr_nm)[mask])
    filtered.gen_data.gear = jnp.array(np.asarray(data_gen.gear)[mask])
    return filtered


def configure_estimator(args: argparse.Namespace):
    """Load the upstream config and override run-specific defaults."""
    conf = load_config(str(args.config_file))
    conf.mode = "fitting"
    conf.filetype = "filtered"
    conf.low_speed_filter_mps = args.low_speed_filter
    conf.enable_plotting = not args.no_plot
    conf.enable_logging = not args.no_log
    conf.output_folder_path = str(REPO_ROOT)
    conf.output_folder = args.output_folder
    conf.nelder_options["sample_points"] = args.sample_points
    conf.svi_options["sample_points"] = args.sample_points
    conf.run_names = [args.data_file.stem]
    return conf


def fit_tire_parameters(conf, sensordata):
    """Run the same fitting loop used by the bundled estimator."""
    params_min = MFSimpleParams(**conf.params_min)
    params_max = MFSimpleParams(**conf.params_max)
    params_init = MFSimpleParams(**conf.params_init)
    vhl_params = load_params(str(conf.vehicle_parameter_names))

    vhl_states = calc_vhl_states(sensordata, vhl_params)
    vhl_forces = calc_vhl_forces(conf.model, sensordata, vhl_states, vhl_params)

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
            setattr(vhl_forces, field.name, fitler_data(force_filter_type, force_filter_settings, sub_dataclass))

    # Transient rejection
    vhl_states, vhl_forces = reject_transient_data(
        sensordata, vhl_states, vhl_forces, 
        max_yaw_accel_radps2=0.4, 
        max_steer_vel_radps=0.2
    )

    # Outlier rejection
    if conf.vhl_data_filter:
        vhl_states, vhl_forces = filter_vhl_data(vhl_states, vhl_forces)

    tire_params_set_svi = STMTireParams()
    std_params_set_svi = STMTireParams()
    tire_params_set_nelder = STMTireParams()

    for state_key, direction in FIT_TARGETS:
        sigma = jnp.array(getattr(getattr(vhl_states, state_key), f"sigma_{direction}"))
        force_n = jnp.array(
            getattr(getattr(vhl_forces, state_key), f"force_{direction}_n")
        )
        load_n = jnp.array(getattr(vhl_forces, state_key).force_z_n)
        fit_flags = getattr(conf, f"fit_flags_{state_key}_{direction}")

        params_init.S_V, params_init.S_H = calc_force_shift(sigma, force_n)
        print(f"Fitting - {state_key} {direction}")
        params_svi, std_svi, _ = tire_param_fitting(
            "SVI",
            "MFSimple",
            sigma,
            force_n,
            load_n,
            params_init,
            params_min,
            params_max,
            conf.svi_options,
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
            conf.nelder_options,
            fit_flags,
        )
        print("SVI:")
        print(params_svi)
        print(std_svi)
        print("Nelder:")
        print(params_nelder)

        setattr(tire_params_set_svi, f"{state_key}_{direction}", params_svi)
        setattr(std_params_set_svi, f"{state_key}_{direction}", std_svi)
        setattr(tire_params_set_nelder, f"{state_key}_{direction}", params_nelder)

    return (
        vhl_params,
        vhl_states,
        vhl_forces,
        params_min,
        params_max,
        tire_params_set_svi,
        std_params_set_svi,
        tire_params_set_nelder,
    )


def maybe_save_results(conf, tire_params_set_svi, std_params_set_svi, tire_params_set_nelder):
    """Save fitted parameter sets to CSV files."""
    if not conf.enable_logging:
        return

    save_dataclass_to_csv(
        tire_params_set_svi,
        conf.output_folder_path,
        conf.output_folder,
        "tire_params_svi.csv",
    )
    save_dataclass_to_csv(
        std_params_set_svi,
        conf.output_folder_path,
        conf.output_folder,
        "std_params_svi.csv",
    )
    save_dataclass_to_csv(
        tire_params_set_nelder,
        conf.output_folder_path,
        conf.output_folder,
        "tire_params_nelder.csv",
    )


def main() -> None:
    args = parse_args()
    conf = configure_estimator(args)
    conf.vehicle_parameter_names = str(args.vehicle_params)
    vhl_params = load_params(str(conf.vehicle_parameter_names))

    sensordata = build_filtered_data(args.data_file, conf, vhl_params)
    (
        _vhl_params,
        vhl_states,
        vhl_forces,
        params_min, 
        params_max,
        tire_params_set_svi,
        std_params_set_svi,
        tire_params_set_nelder,
    ) = fit_tire_parameters(conf, sensordata)

    maybe_save_results(
        conf, tire_params_set_svi, std_params_set_svi, tire_params_set_nelder
    )

    print("-" * 80)
    print("SVI Force Errors")
    eval_force_errors(vhl_states, vhl_forces, tire_params_set_svi)
    print("-" * 80)
    print("Nelder Force Errors")
    eval_force_errors(vhl_states, vhl_forces, tire_params_set_nelder)
    print("-" * 80)

    if conf.enable_plotting:
        vhl_states_plot = calc_vhl_states(sensordata, _vhl_params)
        vhl_forces_plot = calc_vhl_forces(conf.model, sensordata, vhl_states_plot, _vhl_params)
        plot_bell_curves(
            tire_params_set_svi, std_params_set_svi, params_min, params_max
        )
        plot_tire_curves(
            vhl_states, vhl_forces, tire_params_set_svi, tire_params_set_nelder
        )
        plot_lateral_estimation(
            sensordata,
            vhl_states_plot,
            vhl_forces_plot,
            _vhl_params,
            tire_params_set_svi,
        )
        plt.show()


if __name__ == "__main__":
    main()
