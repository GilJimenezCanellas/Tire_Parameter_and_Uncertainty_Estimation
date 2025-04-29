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
    sigma_x = (axle_speed_radps * radius_dynamic_m - vel_tire_x_mps) / vel_tire_x_mps
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


def calc_vhl_states(sensordata: FilteredData, vhlparams: VhlParams, stm_states: STMStates = STMStates()):
    ''' Calculate Vehicle States '''
    stm_states.beta = jnp.arctan(sensordata.cor_data.vel_cog_y_mps /
                                 (jnp.clip(sensordata.cor_data.vel_cog_x_mps, a_min=2)))
    stm_states.dd_psi = jnp.gradient(sensordata.imu_data.yaw_rate_radps)
    load_front_n, load_rear_n = calc_axle_loads(sensordata, vhlparams)
    stm_states.radius_dyn_front_m = \
        calc_tire_radius(load_front_n, vhlparams.r_tire_unloaded_front_m,
                         vhlparams.tire_speed_expansion_front_mpradps2,
                         (sensordata.gen_data.omega_wheel_fl_radps +
                          sensordata.gen_data.omega_wheel_fr_radps) / 2,
                         vhlparams.tire_load_stiffness_front_npm)
    stm_states.radius_dyn_rear_m = \
        calc_tire_radius(load_rear_n, vhlparams.r_tire_unloaded_rear_m,
                         vhlparams.tire_speed_expansion_rear_mpradps2,
                         (sensordata.gen_data.omega_wheel_rl_radps +
                          sensordata.gen_data.omega_wheel_rr_radps) / 2,
                         vhlparams.tire_load_stiffness_rear_npm)
    # Transform lateral velocity to axle frame
    vel_y_front_mps = sensordata.cor_data.vel_cog_y_mps + \
        sensordata.imu_data.yaw_rate_radps * vhlparams.l_front_m
    vel_y_rear_mps = sensordata.cor_data.vel_cog_y_mps - \
        sensordata.imu_data.yaw_rate_radps * vhlparams.l_rear_m
    # Transform velocity to tire frame
    vel_x_tireframe_front_mps = jnp.cos(sensordata.gen_data.delta_f_rad) * sensordata.cor_data.vel_cog_x_mps + \
        jnp.sin(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    vel_y_tireframe_front_mps = -jnp.sin(sensordata.gen_data.delta_f_rad) * sensordata.cor_data.vel_cog_x_mps + \
        jnp.cos(sensordata.gen_data.delta_f_rad) * vel_y_front_mps
    stm_states.front_axle.sigma_x, stm_states.front_axle.sigma_y = \
        calc_slip((sensordata.gen_data.omega_wheel_fl_radps + sensordata.gen_data.omega_wheel_fr_radps) / 2,
                  stm_states.radius_dyn_front_m, vel_x_tireframe_front_mps, vel_y_tireframe_front_mps)
    stm_states.rear_axle.sigma_x, stm_states.rear_axle.sigma_y = \
        calc_slip((sensordata.gen_data.omega_wheel_rl_radps + sensordata.gen_data.omega_wheel_rr_radps) / 2,
                  stm_states.radius_dyn_rear_m, sensordata.cor_data.vel_cog_x_mps, vel_y_rear_mps)
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
    force_x_front_n = jnp.zeros(len(sensordata.gen_data.time))
    force_y_front_n = jnp.zeros(len(sensordata.gen_data.time))
    # Create masks for acceleration and deceleration
    dec_mask = stm_forces.cog.force_x_n < 0.0
    # Calculate values for deceleration
    force_x_front_n = force_x_front_n.at[dec_mask].set(stm_forces.cog.force_x_n[dec_mask] *
                                                       load_front_n[dec_mask] / (load_front_n[dec_mask] + load_rear_n[dec_mask]))
    stm_forces.rear_axle.force_x_n = stm_forces.cog.force_x_n - force_x_front_n

    # Limited Slip Differential
    lsd_yaw_trq_nm = jnp.zeros(len(sensordata.gen_data.time))
    if model == 'LSD-STM':
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

    force_y_front_n = (stm_forces.cog.force_y_n * vhlparams.l_rear_m -
                       vhlparams.izz_kgm2 * vhlstates.dd_psi + lsd_yaw_trq_nm) / (vhlparams.l_front_m + vhlparams.l_rear_m)
    stm_forces.rear_axle.force_y_n = stm_forces.cog.force_y_n - force_y_front_n
    # Calculate Fx and Fy in tire coordinate system acc. Bechtloff
    # "Schätzung des Schwimmwinkels und fahrdynamischer Parameter zur Verbesserung modellbasierter Fahrdynamikregelungen"
    stm_forces.front_axle.force_x_n = force_x_front_n
    stm_forces.front_axle.force_y_n = (1 / (jnp.cos(sensordata.gen_data.delta_f_rad))) * \
        (force_y_front_n - jnp.sin(sensordata.gen_data.delta_f_rad) * force_x_front_n)
    return stm_forces
