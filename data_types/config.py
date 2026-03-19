""" Dataclasses for the used configuration settings """
from dataclasses import dataclass


@dataclass
class Config():
    """ Dataclass for storing configuration data """
    ### Input File Settings ###
    run_names: list
    filetype: str
    signal_names: str
    vehicle_parameter_names: str
    model: str
    filetype: str
    file_path: str
    mode: str
    ### Preprocessing Settings ###
    filter_gen: str
    settings_gen: dict
    filter_cor: str
    settings_cor: dict
    filter_imu: str
    settings_imu: dict
    low_speed_filter_mps: float
    gear_change_filter: bool
    imu_acc_z_fix: bool
    sampling_freq_hz: float
    vhl_data_filter: bool
    save_filtered_data: bool
    ### Sensor Offsets ###
    lambda_v: float
    lambda_ax: float
    lambda_ay: float
    ax_median: float
    ay_median: float
    az_off: float
    yaw_rate_off: float
    ### Fitting Settings ###
    nelder_options: dict
    svi_options: dict
    params_min: dict
    params_max: dict
    params_init: dict
    fit_flags_wheel_fl_x: dict
    fit_flags_wheel_fr_x: dict
    fit_flags_wheel_rl_x: dict
    fit_flags_wheel_rr_x: dict
    fit_flags_front_axle_y: dict
    fit_flags_rear_axle_y: dict
    ### Evaluation Settings ###
    output_folder_path: str
    output_folder: str
    enable_plotting: bool
    enable_logging: bool
    clean_plots: bool
