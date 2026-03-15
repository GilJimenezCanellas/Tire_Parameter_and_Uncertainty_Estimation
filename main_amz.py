"""Fit tire parameters from AMZ MATLAB data using the bundled estimator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
TIRE_LIB_ROOT = REPO_ROOT / "Tire_Parameter_and_Uncertainty_Estimation"
if str(TIRE_LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(TIRE_LIB_ROOT))

from data_types.sensordata import FilteredData
from data_types.vehicleparameters import MFSimpleParams, STMTireParams
from src.param_fitting.param_fitting import calc_force_shift, tire_param_fitting
from src.utils.calc_vhl_states import calc_vhl_forces, calc_vhl_states
from src.utils.datamanager import load_config, load_params, save_dataclass_to_csv
from src.utils.evaluation_helpers import (
    eval_force_errors,
    plot_bell_curves,
    plot_tire_curves,
)
from src.utils.filter_data import filter_vhl_data


DEFAULT_DATA_FILE = (
    REPO_ROOT
    / ".."
    / "inputs"
    / "August"
    / "DV_trackdrive"
    / "2025-08-05_18-08-02_run_21"
    / "2025-08-05_18-08-02_DV_trackdrive_csv_test_v2_data.mat"
)
DEFAULT_CONFIG_FILE = TIRE_LIB_ROOT / "setup" / "config_amz_dv.toml"
DEFAULT_VEHICLE_PARAMS = TIRE_LIB_ROOT / "setup" / "vehicle_parameters_amz_dv.toml"


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


def mat_array(mat_data: dict, *names: str) -> np.ndarray:
    """Return the first available signal as a flat float array."""
    for name in names:
        if name in mat_data:
            return np.asarray(mat_data[name], dtype=float).squeeze()
    raise KeyError(f"None of the signals were found: {names}")


def build_filtered_data(data_file: Path, low_speed_filter_mps: float) -> FilteredData:
    """Map AMZ run data into the estimator's filtered sensor dataclass."""
    data_mat = loadmat(data_file, squeeze_me=True, struct_as_record=False)

    time = mat_array(data_mat, "Time")
    vel_x = mat_array(data_mat, "v_X_OVS", "v_X_VE", "V_x")
    vel_y = mat_array(data_mat, "v_Y_OVS", "v_Y_VE")
    acc_x = mat_array(data_mat, "a_X_OVS", "a_X_INS", "a_X_VE")
    acc_y = mat_array(data_mat, "a_Y_OVS", "a_Y_INS", "a_Y_VE")
    yaw_rate = mat_array(data_mat, "omega_Z_OVS", "omega_Z_INS", "omega_Z_VE")
    # steer = mat_array(data_mat, "steering_target_rad")
    steer_fr = np.asarray(data_mat["delta_W_FR"]).squeeze()
    steer_fl = np.asarray(data_mat["delta_W_FL"]).squeeze()
    steer = (steer_fr + steer_fl) / 2

    wheel_speed_fl = mat_array(data_mat, "v_X_W_FL")
    wheel_speed_fr = mat_array(data_mat, "v_X_W_FR")
    wheel_speed_rl = mat_array(data_mat, "v_X_W_RL")
    wheel_speed_rr = mat_array(data_mat, "v_X_W_RR")
    rolling_radius = 0.203 # possibility of variable rolling radius

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
            wheel_speed_fl,
            wheel_speed_fr,
            wheel_speed_rl,
            wheel_speed_rr,
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
    wheel_speed_fl = wheel_speed_fl[:size]
    wheel_speed_fr = wheel_speed_fr[:size]
    wheel_speed_rl = wheel_speed_rl[:size]
    wheel_speed_rr = wheel_speed_rr[:size]
    omega_fl = wheel_speed_fl / rolling_radius
    omega_fr = wheel_speed_fr / rolling_radius
    omega_rl = wheel_speed_rl / rolling_radius
    omega_rr = wheel_speed_rr / rolling_radius

    # The estimator expects a positive vertical acceleration for static load.
    acc_z = np.full(size, 9.81, dtype=float)

    finite_mask = np.isfinite(
        np.column_stack(
            [
                time,
                vel_x,
                vel_y,
                acc_x,
                acc_y,
                yaw_rate,
                steer,
                omega_fl,
                omega_fr,
                omega_rl,
                omega_rr,
            ]
        )
    ).all(axis=1)
    speed_mask = vel_x > low_speed_filter_mps
    mask = finite_mask & speed_mask

    filtered = FilteredData()
    filtered.cor_data.time = jnp.array(time[mask])
    filtered.cor_data.vel_cog_x_mps = jnp.array(vel_x[mask])
    filtered.cor_data.vel_cog_y_mps = jnp.array(vel_y[mask])

    filtered.imu_data.time = jnp.array(time[mask])
    filtered.imu_data.acc_cog_x_mps2 = jnp.array(acc_x[mask])
    filtered.imu_data.acc_cog_y_mps2 = jnp.array(acc_y[mask])
    filtered.imu_data.acc_cog_z_mps2 = jnp.array(acc_z[mask])
    filtered.imu_data.yaw_rate_radps = jnp.array(yaw_rate[mask])

    filtered.gen_data.time = jnp.array(time[mask])
    filtered.gen_data.delta_f_rad = jnp.array(steer[mask])
    filtered.gen_data.omega_wheel_fl_radps = jnp.array(omega_fl[mask])
    filtered.gen_data.omega_wheel_fr_radps = jnp.array(omega_fr[mask])
    filtered.gen_data.omega_wheel_rl_radps = jnp.array(omega_rl[mask])
    filtered.gen_data.omega_wheel_rr_radps = jnp.array(omega_rr[mask])
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
    if conf.vhl_data_filter:
        vhl_states, vhl_forces = filter_vhl_data(vhl_states, vhl_forces)

    tire_params_set_svi = STMTireParams()
    std_params_set_svi = STMTireParams()
    tire_params_set_nelder = STMTireParams()

    for axle in ("front_axle", "rear_axle"):
        for direction in ("x", "y"):
            sigma = jnp.array(getattr(getattr(vhl_states, axle), f"sigma_{direction}"))
            force_n = jnp.array(
                getattr(getattr(vhl_forces, axle), f"force_{direction}_n")
            )
            load_n = jnp.array(getattr(vhl_forces, axle).force_z_n)
            fit_flags = getattr(conf, f"fit_flags_{axle}_{direction}")

            params_init.S_V, params_init.S_H = calc_force_shift(sigma, force_n)
            print(f"Fitting - {axle} {direction}")
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

            setattr(tire_params_set_svi, f"{axle}_{direction}", params_svi)
            setattr(std_params_set_svi, f"{axle}_{direction}", std_svi)
            setattr(tire_params_set_nelder, f"{axle}_{direction}", params_nelder)

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

    sensordata = build_filtered_data(args.data_file, conf.low_speed_filter_mps)
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
        plot_bell_curves(
            tire_params_set_svi, std_params_set_svi, params_min, params_max
        )
        plot_tire_curves(
            vhl_states, vhl_forces, tire_params_set_svi, tire_params_set_nelder
        )
        plt.show()


if __name__ == "__main__":
    main()
