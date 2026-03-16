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


def calc_tire_radius(axle_load_n: jnp.array, radius_tire_unloaded_m: jnp.array, tire_speed_expansion_mpradps2: float,
                     omega_axle_radps: jnp.array, tire_load_stiffness_npm: float):
    '''
    Calc dynamic Tire Radius according to M. Schabauer et al., “Experimental Investigation
    and Semi-physical Modelling of the Influence of Rotational Speed on the Vertical Tyre
    Stiffness and Tyre Radii
    '''
    radius_def_load = axle_load_n / (2 * tire_load_stiffness_npm)
    radius_def_speed = tire_speed_expansion_mpradps2 * \
        (abs(omega_axle_radps)) ** 2
    return radius_tire_unloaded_m - radius_def_load / 3 + 2 * radius_def_speed / 3


def calc_ang_acc(signal_radps: jnp.array, time_s: jnp.array):
    '''Calculate angular acceleration from a filtered angular velocity signal.'''
    if len(signal_radps) == 0:
        return jnp.array([])
    if len(signal_radps) == 1:
        return jnp.zeros_like(signal_radps)
    return jnp.gradient(signal_radps, time_s)


def has_motor_force_inputs(sensordata: FilteredData):
    '''Check whether filtered motor speed signals are available for wheel-wise Fx estimation.'''
    omega_m_signals = [
        sensordata.gen_data.omega_m_fl_radps,
        sensordata.gen_data.omega_m_fr_radps,
        sensordata.gen_data.omega_m_rl_radps,
        sensordata.gen_data.omega_m_rr_radps,
    ]
    torque_signals = [
        sensordata.gen_data.t_m_fl_nm,
        sensordata.gen_data.t_m_fr_nm,
        sensordata.gen_data.t_m_rl_nm,
        sensordata.gen_data.t_m_rr_nm,
    ]
    wheel_speed_signals = [
        sensordata.gen_data.omega_wheel_fl_radps,
        sensordata.gen_data.omega_wheel_fr_radps,
        sensordata.gen_data.omega_wheel_rl_radps,
        sensordata.gen_data.omega_wheel_rr_radps,
    ]
    if any(len(signal) == 0 for signal in omega_m_signals + torque_signals):
        return False
    motor_speed_available = any(not jnp.allclose(signal, 0.0) for signal in omega_m_signals)
    wheel_speed_available = any(not jnp.allclose(signal, 0.0) for signal in wheel_speed_signals)
    return motor_speed_available or not wheel_speed_available


def calc_wheel_force_from_torque(motor_torque_nm: jnp.array, motor_speed_radps: jnp.array, time_s: jnp.array,
                                 gear_ratio: float, wheel_inertia_kgm2: float, wheel_radius_m: jnp.array):
    '''Estimate wheel longitudinal force from rotational dynamics.'''
    safe_gear_ratio = jnp.maximum(jnp.abs(gear_ratio), 1.0e-6)
    wheel_speed_radps = motor_speed_radps / safe_gear_ratio
    wheel_acc_radps2 = calc_ang_acc(wheel_speed_radps, time_s)
    wheel_torque_nm = motor_torque_nm * gear_ratio
    safe_radius_m = jnp.maximum(jnp.abs(wheel_radius_m), 1.0e-6)
    return (wheel_torque_nm - wheel_inertia_kgm2 * wheel_acc_radps2 - 35) / safe_radius_m


def calc_vhl_states(sensordata: FilteredData, vhlparams: VhlParams, stm_states: STMStates = STMStates()):
    ''' Calculate Vehicle States '''
    stm_states.beta = jnp.arctan(sensordata.cor_data.vel_cog_y_mps /
                                 (jnp.clip(sensordata.cor_data.vel_cog_x_mps, a_min=2)))
    stm_states.dd_psi = jnp.gradient(sensordata.imu_data.yaw_rate_radps, sensordata.gen_data.time)
    load_front_n, load_rear_n = calc_axle_loads(sensordata, vhlparams)
    stm_states.radius_dyn_front_m = 0.203 #\
        # calc_tire_radius(load_front_n, vhlparams.r_tire_unloaded_front_m,
        #                  vhlparams.tire_speed_expansion_front_mpradps2,
        #                  (sensordata.gen_data.omega_wheel_fl_radps +
        #                   sensordata.gen_data.omega_wheel_fr_radps) / 2,
        #                  vhlparams.tire_load_stiffness_front_npm)
    stm_states.radius_dyn_rear_m = 0.203 #\
        # calc_tire_radius(load_rear_n, vhlparams.r_tire_unloaded_rear_m,
        #                  vhlparams.tire_speed_expansion_rear_mpradps2,
        #                  (sensordata.gen_data.omega_wheel_rl_radps +
        #                   sensordata.gen_data.omega_wheel_rr_radps) / 2,
        #                  vhlparams.tire_load_stiffness_rear_npm)
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


def calc_vhl_forces(model: str, sensordata: FilteredData, vhlstates: STMStates, vhlparams: VhlParams, stm_forces: STMForces = STMForces()):
    ''' Calculate Vehicle and Tire Forces '''
    force_drag_n = 0.5 * vhlparams.roh_air_kgpm3 * vhlparams.a_vehicle_m2 * vhlparams.cw * \
        sensordata.cor_data.vel_cog_x_mps**2
    # Forces on center of gravity
    stm_forces.cog.force_z_n = vhlparams.mass_kg * sensordata.imu_data.acc_cog_z_mps2
    force_roll_n = vhlparams.tire_roll_resistance * stm_forces.cog.force_z_n
    stm_forces.cog.force_x_n = vhlparams.mass_kg * \
        sensordata.imu_data.acc_cog_x_mps2 + force_drag_n + force_roll_n
    stm_forces.cog.force_y_n = vhlparams.mass_kg * sensordata.imu_data.acc_cog_y_mps2
    # Axle Loads
    load_front_n, load_rear_n = calc_axle_loads(sensordata, vhlparams)
    stm_forces.front_axle.force_z_n = load_front_n
    stm_forces.rear_axle.force_z_n = load_rear_n
    # Axle Force calculation
    # similar to Farroni T.R.I.C.K.‐Tire/Road Interaction Characterization & Knowledge
    use_motor_force_inputs = has_motor_force_inputs(sensordata)
    force_x_front_n = jnp.zeros(len(sensordata.gen_data.time))
    force_y_front_n = jnp.zeros(len(sensordata.gen_data.time))
    if use_motor_force_inputs:
        time_s = sensordata.gen_data.time
        wheel_force_fl_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_fl_nm,
            sensordata.gen_data.omega_m_fl_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_front_m,
        )
        wheel_force_fr_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_fr_nm,
            sensordata.gen_data.omega_m_fr_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_front_m,
        )
        wheel_force_rl_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_rl_nm,
            sensordata.gen_data.omega_m_rl_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_rear_m,
        )
        wheel_force_rr_n = calc_wheel_force_from_torque(
            sensordata.gen_data.t_m_rr_nm,
            sensordata.gen_data.omega_m_rr_radps,
            time_s,
            vhlparams.gear_ratio,
            vhlparams.wheel_inertia_kgm2,
            vhlstates.radius_dyn_rear_m,
        )
        stm_forces.wheel_fl.force_x_n = wheel_force_fl_n
        stm_forces.wheel_fr.force_x_n = wheel_force_fr_n
        stm_forces.wheel_rl.force_x_n = wheel_force_rl_n
        stm_forces.wheel_rr.force_x_n = wheel_force_rr_n
        force_x_front_n = wheel_force_fl_n + wheel_force_fr_n
        stm_forces.rear_axle.force_x_n = wheel_force_rl_n + wheel_force_rr_n
    else:
        force_x_front_n = stm_forces.cog.force_x_n * load_front_n / (load_front_n + load_rear_n)
        stm_forces.rear_axle.force_x_n = stm_forces.cog.force_x_n - force_x_front_n
        stm_forces.wheel_fl.force_x_n = force_x_front_n / 2
        stm_forces.wheel_fr.force_x_n = force_x_front_n / 2
        stm_forces.wheel_rl.force_x_n = stm_forces.rear_axle.force_x_n / 2
        stm_forces.wheel_rr.force_x_n = stm_forces.rear_axle.force_x_n / 2

    # Consider Torque Vectoring
    tv_yaw_moment_nm = jnp.zeros(len(sensordata.gen_data.time))
    if use_motor_force_inputs: 
        stm_forces.wheel_fl.force_x_n = wheel_force_fl_n
        stm_forces.wheel_fr.force_x_n = wheel_force_fr_n
        stm_forces.wheel_rl.force_x_n = wheel_force_rl_n
        stm_forces.wheel_rr.force_x_n = wheel_force_rr_n
        force_x_front_n = wheel_force_fl_n + wheel_force_fr_n
        stm_forces.rear_axle.force_x_n = wheel_force_rl_n + wheel_force_rr_n
        
        # Calculate Torque Vectoring Yaw Moment
        tv_yaw_moment_nm = ((wheel_force_fr_n - wheel_force_fl_n) * (vhlparams.tw_front_m / 2) * jnp.cos(sensordata.gen_data.delta_f_rad) +
                            (wheel_force_rr_n - wheel_force_rl_n) * (vhlparams.tw_rear_m / 2))

    # Limited Slip Differential
    lsd_yaw_trq_nm = jnp.zeros(len(sensordata.gen_data.time))
    if model == 'LSD-STM' and not use_motor_force_inputs:
        # According to Gadola et al. "On the Passive Limited Slip Differential for High Performance Vehicle Applications"
        drive_torque_nm = stm_forces.rear_axle.force_x_n * vhlparams.r_tire_unloaded_rear_m
        # clip engine torque to maximum engine brake torque times gear ratio // assumption: max 800 Nm at wheels
        drive_torque_nm = jnp.abs(jnp.clip(drive_torque_nm, -800, None))
        kappa = 0.5  # slip sensitivity coefficient
        sigma_diff = sensordata.gen_data.omega_wheel_rl_radps - sensordata.gen_data.omega_wheel_rr_radps
        coeff_lsd = jnp.where(stm_forces.rear_axle.force_x_n > 0,
                              vhlparams.ratio_lock_drive, vhlparams.ratio_lock_coast)
        torque_lsd_nm = jnp.where(jnp.abs(coeff_lsd * jnp.tanh(kappa * sigma_diff) * drive_torque_nm) < vhlparams.torque_preload_nm,
                                  jnp.tanh(kappa * sigma_diff) * vhlparams.torque_preload_nm,
                                  coeff_lsd * jnp.tanh(kappa * sigma_diff) * drive_torque_nm)
        force_diff = (torque_lsd_nm / vhlparams.r_tire_unloaded_rear_m)
        lsd_yaw_trq_nm = force_diff * vhlparams.tw_rear_m / 2

    force_y_front_n = (stm_forces.cog.force_y_n * vhlparams.l_rear_m + 
                       vhlparams.izz_kgm2 * vhlstates.dd_psi - tv_yaw_moment_nm + 
                       lsd_yaw_trq_nm) / (vhlparams.l_front_m + vhlparams.l_rear_m)
    
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
