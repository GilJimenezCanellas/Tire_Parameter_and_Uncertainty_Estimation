''' Calculate Vehicle States and Forces '''
import jax.numpy as jnp

from data_types.sensordata import FilteredData
from data_types.vehiclestates import STMForces, STMStates
from data_types.vehicleparameters import VhlParams


def calc_axle_loads(sensordata: FilteredData, vhlparams: VhlParams):
    ''' Calculate Axle Loads '''
    load_cog_n = vhlparams.mass_kg * sensordata.imu_data.acc_cog_z_mps2
    load_aero_front_n = -0.5 * vhlparams.roh_air_kgpm3 * vhlparams.a_vehicle_m2 * \
        vhlparams.cl_front * sensordata.cor_data.vel_cog_x_mps**2
    load_aero_rear_n = -0.5 * vhlparams.roh_air_kgpm3 * vhlparams.a_vehicle_m2 * \
        vhlparams.cl_rear * sensordata.cor_data.vel_cog_x_mps**2
    load_transfer_pitch_n = vhlparams.mass_kg * vhlparams.cog_z_m * \
        sensordata.imu_data.acc_cog_x_mps2 / \
        (vhlparams.l_front_m + vhlparams.l_rear_m)
    load_front_n = (load_cog_n * vhlparams.l_rear_m /
                    (vhlparams.l_front_m + vhlparams.l_rear_m)) + load_aero_front_n - load_transfer_pitch_n
    load_rear_n = (load_cog_n * vhlparams.l_front_m /
                   (vhlparams.l_front_m + vhlparams.l_rear_m)) + load_aero_rear_n + load_transfer_pitch_n
    return load_front_n, load_rear_n


def calc_slip(axle_speed_radps: jnp.array, radius_dynamic_m: jnp.array, vel_tire_x_mps: jnp.array,
              vel_tire_y_mps: jnp.array):
    ''' Calculate tire slip '''
    vel_tire_x_mps = jnp.clip(vel_tire_x_mps, a_min=2)
    sigma_x = (axle_speed_radps * radius_dynamic_m - vel_tire_x_mps) / jnp.maximum(vel_tire_x_mps, axle_speed_radps * radius_dynamic_m)
    sigma_y = -jnp.arctan(vel_tire_y_mps / vel_tire_x_mps)
    return sigma_x, sigma_y


def _get_signal(sensordata: FilteredData, signal_name: str) -> jnp.array:
    section_name, field_name = signal_name.split(".", maxsplit=1)
    return getattr(getattr(sensordata, section_name), field_name)


def _validate_signal(signal_name: str, signal: jnp.array, expected_length: int | None,
                     require_nonzero: bool = True) -> None:
    values = jnp.asarray(signal)
    if values.ndim == 0:
        values = values.reshape((1,))
    if values.size == 0:
        raise ValueError(f"Required signal '{signal_name}' is empty.")
    if expected_length is not None and values.shape[0] != expected_length:
        raise ValueError(
            f"Required signal '{signal_name}' has {values.shape[0]} samples, "
            f"expected {expected_length}."
        )
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError(f"Required signal '{signal_name}' contains NaN or infinite values.")
    if require_nonzero and bool(jnp.allclose(values, 0.0)):
        raise ValueError(f"Required signal '{signal_name}' is all zero.")


def validate_required_sensor_signals(sensordata: FilteredData, *, include_motor_speeds: bool = False,
                                     include_force_inputs: bool = False) -> None:
    """Validate filtered signals required by tire state and force estimation."""
    time = jnp.asarray(sensordata.gen_data.time)
    if time.ndim == 0:
        time = time.reshape((1,))
    _validate_signal("gen_data.time", time, None, require_nonzero=True)
    if time.shape[0] < 2:
        raise ValueError("Required signal 'gen_data.time' must contain at least two samples.")
    if not bool(jnp.all(jnp.diff(time) > 0.0)):
        raise ValueError("Required signal 'gen_data.time' must be strictly increasing.")

    required_signals = [
        "cor_data.time",
        "imu_data.time",
        "cor_data.vel_cog_x_mps",
        "cor_data.vel_cog_y_mps",
        "imu_data.acc_cog_x_mps2",
        "imu_data.acc_cog_y_mps2",
        "imu_data.acc_cog_z_mps2",
        "imu_data.yaw_rate_radps",
        "gen_data.delta_f_rad",
        "gen_data.omega_wheel_fl_radps",
        "gen_data.omega_wheel_fr_radps",
        "gen_data.omega_wheel_rl_radps",
        "gen_data.omega_wheel_rr_radps",
    ]
    if include_force_inputs:
        required_signals.extend(
            [
                "gen_data.t_m_fl_nm",
                "gen_data.t_m_fr_nm",
                "gen_data.t_m_rl_nm",
                "gen_data.t_m_rr_nm",
            ]
        )
    if include_motor_speeds:
        required_signals.extend(
            [
                "gen_data.omega_m_fl_radps",
                "gen_data.omega_m_fr_radps",
                "gen_data.omega_m_rl_radps",
                "gen_data.omega_m_rr_radps",
            ]
        )

    expected_length = time.shape[0]
    for signal_name in required_signals:
        require_nonzero = signal_name not in {"cor_data.time", "imu_data.time"}
        _validate_signal(
            signal_name,
            _get_signal(sensordata, signal_name),
            expected_length,
            require_nonzero=require_nonzero,
        )


def calc_ang_acc(signal_radps: jnp.array, time_s: jnp.array):
    '''Calculate angular acceleration from a filtered angular velocity signal.'''
    if len(signal_radps) == 0:
        return jnp.array([])
    if len(signal_radps) == 1:
        return jnp.zeros_like(signal_radps)
    return jnp.gradient(signal_radps, time_s)


def calc_wheel_force_from_torque(motor_torque_nm: jnp.array, wheel_speed_radps: jnp.array, time_s: jnp.array,
                                 gear_ratio: float, wheel_inertia_kgm2: float, wheel_radius_m: jnp.array):
    '''Estimate wheel longitudinal force from rotational dynamics.'''
    wheel_acc_radps2 = calc_ang_acc(wheel_speed_radps, time_s)
    wheel_torque_nm = motor_torque_nm * gear_ratio
    safe_radius_m = jnp.maximum(jnp.abs(wheel_radius_m), 1.0e-6)
    return (wheel_torque_nm - wheel_inertia_kgm2 * wheel_acc_radps2 - 35) / safe_radius_m


def calc_wheel_force_from_ideal_torque(motor_torque_nm: jnp.array, gear_ratio: float,
                                       wheel_radius_m: jnp.array):
    '''Estimate wheel longitudinal force with an ideal torque-to-ground-force model.'''
    wheel_torque_nm = motor_torque_nm * gear_ratio
    safe_radius_m = jnp.maximum(jnp.abs(wheel_radius_m), 1.0e-6)
    return wheel_torque_nm / safe_radius_m


def calc_vhl_states(sensordata: FilteredData, vhlparams: VhlParams, stm_states: STMStates = STMStates()):
    ''' Calculate Vehicle States '''
    validate_required_sensor_signals(sensordata)
    stm_states.beta = jnp.arctan(sensordata.cor_data.vel_cog_y_mps /
                                 (jnp.clip(sensordata.cor_data.vel_cog_x_mps, a_min=2)))
    stm_states.dd_psi = jnp.gradient(sensordata.imu_data.yaw_rate_radps, sensordata.gen_data.time)
    load_front_n, load_rear_n = calc_axle_loads(sensordata, vhlparams)
    stm_states.radius_dyn_front_m = jnp.ones_like(load_front_n) * vhlparams.r_tire_unloaded_front_m
    stm_states.radius_dyn_rear_m = jnp.ones_like(load_rear_n) * vhlparams.r_tire_unloaded_rear_m
    # Transform lateral velocity to axle frame
    vel_y_front_mps = sensordata.cor_data.vel_cog_y_mps + \
        sensordata.imu_data.yaw_rate_radps * vhlparams.l_front_m
    vel_y_rear_mps = sensordata.cor_data.vel_cog_y_mps - \
        sensordata.imu_data.yaw_rate_radps * vhlparams.l_rear_m
    vel_x_front_left_mps = sensordata.cor_data.vel_cog_x_mps - \
        sensordata.imu_data.yaw_rate_radps * vhlparams.tw_front_m / 2
    vel_x_front_right_mps = sensordata.cor_data.vel_cog_x_mps + \
        sensordata.imu_data.yaw_rate_radps * vhlparams.tw_front_m / 2
    vel_x_rear_left_mps = sensordata.cor_data.vel_cog_x_mps - \
        sensordata.imu_data.yaw_rate_radps * vhlparams.tw_rear_m / 2
    vel_x_rear_right_mps = sensordata.cor_data.vel_cog_x_mps + \
        sensordata.imu_data.yaw_rate_radps * vhlparams.tw_rear_m / 2
    # Transform velocity to tire frame
    vel_x_tireframe_front_mps = jnp.cos(sensordata.gen_data.delta_f_rad) * sensordata.cor_data.vel_cog_x_mps + \
        jnp.sin(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    vel_y_tireframe_front_mps = -jnp.sin(sensordata.gen_data.delta_f_rad) * sensordata.cor_data.vel_cog_x_mps + \
        jnp.cos(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    
    vel_x_tireframe_front_left_mps = jnp.cos(sensordata.gen_data.delta_f_rad) * vel_x_front_left_mps + \
        jnp.sin(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    vel_y_tireframe_front_left_mps = -jnp.sin(sensordata.gen_data.delta_f_rad) * vel_x_front_left_mps + \
        jnp.cos(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    
    vel_x_tireframe_front_right_mps = jnp.cos(sensordata.gen_data.delta_f_rad) * vel_x_front_right_mps + \
        jnp.sin(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    vel_y_tireframe_front_right_mps = -jnp.sin(sensordata.gen_data.delta_f_rad) * vel_x_front_right_mps + \
        jnp.cos(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    
    stm_states.wheel_fl.sigma_x, stm_states.wheel_fl.sigma_y = \
        calc_slip(sensordata.gen_data.omega_wheel_fl_radps,
                  stm_states.radius_dyn_front_m, vel_x_tireframe_front_left_mps, vel_y_tireframe_front_left_mps)
    stm_states.wheel_fr.sigma_x, stm_states.wheel_fr.sigma_y = \
        calc_slip(sensordata.gen_data.omega_wheel_fr_radps,
                  stm_states.radius_dyn_front_m, vel_x_tireframe_front_right_mps, vel_y_tireframe_front_right_mps)
    stm_states.wheel_rl.sigma_x, stm_states.wheel_rl.sigma_y = \
        calc_slip(sensordata.gen_data.omega_wheel_rl_radps,
                  stm_states.radius_dyn_rear_m, vel_x_rear_left_mps, vel_y_rear_mps)
    stm_states.wheel_rr.sigma_x, stm_states.wheel_rr.sigma_y = \
        calc_slip(sensordata.gen_data.omega_wheel_rr_radps,
                  stm_states.radius_dyn_rear_m, vel_x_rear_right_mps, vel_y_rear_mps)
    stm_states.front_axle.sigma_x = (stm_states.wheel_fl.sigma_x + stm_states.wheel_fr.sigma_x) / 2
    stm_states.front_axle.sigma_y = (stm_states.wheel_fl.sigma_y + stm_states.wheel_fr.sigma_y) / 2
    stm_states.rear_axle.sigma_x = (stm_states.wheel_rl.sigma_x + stm_states.wheel_rr.sigma_x) / 2
    stm_states.rear_axle.sigma_y = (stm_states.wheel_rl.sigma_y + stm_states.wheel_rr.sigma_y) / 2
    return stm_states


def calc_vhl_forces(model: str, sensordata: FilteredData, vhlstates: STMStates, vhlparams: VhlParams,
                    stm_forces: STMForces = STMForces(), longitudinal_force_mode: str = "wheel_dynamics"):
    ''' Calculate Vehicle and Tire Forces '''
    validate_required_sensor_signals(sensordata, include_force_inputs=True)
    force_drag_n = 0.5 * vhlparams.roh_air_kgpm3 * vhlparams.a_vehicle_m2 * vhlparams.cw * \
        sensordata.cor_data.vel_cog_x_mps**2
    # Forces on center of gravity
    stm_forces.cog.force_z_n = vhlparams.mass_kg * sensordata.imu_data.acc_cog_z_mps2
    stm_forces.cog.force_x_n = vhlparams.mass_kg * \
        sensordata.imu_data.acc_cog_x_mps2 + force_drag_n
    stm_forces.cog.force_y_n = vhlparams.mass_kg * sensordata.imu_data.acc_cog_y_mps2
    # Axle Loads
    load_front_n, load_rear_n = calc_axle_loads(sensordata, vhlparams)
    stm_forces.front_axle.force_z_n = load_front_n
    stm_forces.rear_axle.force_z_n = load_rear_n
    # Axle Force calculation
    # similar to Farroni T.R.I.C.K.‐Tire/Road Interaction Characterization & Knowledge
    time_s = sensordata.gen_data.time
    if longitudinal_force_mode == "wheel_dynamics":
        wheel_force_fl_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_fl_nm,
            sensordata.gen_data.omega_wheel_fl_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_front_m,
        )
        wheel_force_fr_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_fr_nm,
            sensordata.gen_data.omega_wheel_fr_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_front_m,
        )
        wheel_force_rl_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_rl_nm,
            sensordata.gen_data.omega_wheel_rl_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_rear_m,
        )
        wheel_force_rr_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_rr_nm,
            sensordata.gen_data.omega_wheel_rr_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_rear_m,
        )
    elif longitudinal_force_mode == "ideal_torque":
        wheel_force_fl_n = calc_wheel_force_from_ideal_torque(
            sensordata.gen_data.t_m_fl_nm, vhlparams.gear_ratio, vhlstates.radius_dyn_front_m
        )
        wheel_force_fr_n = calc_wheel_force_from_ideal_torque(
            sensordata.gen_data.t_m_fr_nm, vhlparams.gear_ratio, vhlstates.radius_dyn_front_m
        )
        wheel_force_rl_n = calc_wheel_force_from_ideal_torque(
            sensordata.gen_data.t_m_rl_nm, vhlparams.gear_ratio, vhlstates.radius_dyn_rear_m
        )
        wheel_force_rr_n = calc_wheel_force_from_ideal_torque(
            sensordata.gen_data.t_m_rr_nm, vhlparams.gear_ratio, vhlstates.radius_dyn_rear_m
        )
    else:
        raise ValueError(
            "Unknown longitudinal force mode: "
            f"{longitudinal_force_mode}. Expected 'wheel_dynamics' or 'ideal_torque'."
        )
    stm_forces.wheel_fl.force_x_n = wheel_force_fl_n
    stm_forces.wheel_fr.force_x_n = wheel_force_fr_n
    stm_forces.wheel_rl.force_x_n = wheel_force_rl_n
    stm_forces.wheel_rr.force_x_n = wheel_force_rr_n
    force_x_front_n = wheel_force_fl_n + wheel_force_fr_n
    stm_forces.rear_axle.force_x_n = wheel_force_rl_n + wheel_force_rr_n

    # Consider Torque Vectoring
    tv_yaw_moment_nm = ((wheel_force_fr_n - wheel_force_fl_n) * (vhlparams.tw_front_m / 2) * jnp.cos(sensordata.gen_data.delta_f_rad) +
                        (wheel_force_rr_n - wheel_force_rl_n) * (vhlparams.tw_rear_m / 2))

    force_y_front_n = (stm_forces.cog.force_y_n * vhlparams.l_rear_m + 
                       vhlparams.izz_kgm2 * vhlstates.dd_psi - tv_yaw_moment_nm) / (vhlparams.l_front_m + vhlparams.l_rear_m)
    
    stm_forces.rear_axle.force_y_n = stm_forces.cog.force_y_n - force_y_front_n
    front_load_transfer_n = force_y_front_n * vhlparams.cog_z_m / vhlparams.tw_front_m
    rear_load_transfer_n = stm_forces.rear_axle.force_y_n * vhlparams.cog_z_m / vhlparams.tw_rear_m
    stm_forces.wheel_fl.force_z_n = jnp.clip(load_front_n / 2 - front_load_transfer_n, a_min=1.0)
    stm_forces.wheel_fr.force_z_n = jnp.clip(load_front_n / 2 + front_load_transfer_n, a_min=1.0)
    stm_forces.wheel_rl.force_z_n = jnp.clip(load_rear_n / 2 - rear_load_transfer_n, a_min=1.0)
    stm_forces.wheel_rr.force_z_n = jnp.clip(load_rear_n / 2 + rear_load_transfer_n, a_min=1.0)
    # Calculate Fx and Fy in tire coordinate system acc. Bechtloff
    # "Schätzung des Schwimmwinkels und fahrdynamischer Parameter zur Verbesserung modellbasierter Fahrdynamikregelungen"
    stm_forces.front_axle.force_x_n = force_x_front_n
    stm_forces.front_axle.force_y_n = (1 / (jnp.cos(sensordata.gen_data.delta_f_rad))) * \
        (force_y_front_n - jnp.sin(sensordata.gen_data.delta_f_rad) * force_x_front_n)
    stm_forces.wheel_fl.force_y_n = stm_forces.front_axle.force_y_n / 2
    stm_forces.wheel_fr.force_y_n = stm_forces.front_axle.force_y_n / 2
    stm_forces.wheel_rl.force_y_n = stm_forces.rear_axle.force_y_n / 2
    stm_forces.wheel_rr.force_y_n = stm_forces.rear_axle.force_y_n / 2
    return stm_forces
