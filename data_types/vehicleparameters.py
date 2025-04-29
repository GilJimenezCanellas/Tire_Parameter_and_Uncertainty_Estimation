''' Dataclasses for the used vehicle parameters '''
from dataclasses import dataclass, field


@dataclass
class MFSimpleParams:
    ''' Dataclass for storing MF Simple parameters '''
    B: float = 0
    C: float = 0
    D: float = 0
    E: float = 0
    S_V: float = 0
    S_H: float = 0


@dataclass
class MFCombinedParams:
    ''' Dataclass for storing MF Combined parameters '''
    kappa_lon: float = 0
    kappa_lat: float = 0
    mf_params: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))


@dataclass
class STMTireParams:
    ''' Dataclass for storing STM Tire parameters '''
    front_axle_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    front_axle_y: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    rear_axle_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    rear_axle_y: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))


@dataclass
class VhlParams:
    ''' Dataclass for storing vehicle parameters '''
    ### Vehicle Measurements ###
    l_front_m: float
    l_rear_m: float
    cog_z_m: float
    tw_rear_m: float
    ### Vehicle Intertia ###
    mass_kg: float
    izz_kgm2: float
    ### Tire Properties ###
    r_tire_unloaded_front_m: float
    r_tire_unloaded_rear_m: float
    tire_load_stiffness_front_npm: float
    tire_load_stiffness_rear_npm: float
    tire_speed_expansion_front_mpradps2: float
    tire_speed_expansion_rear_mpradps2: float
    tire_roll_resistance: float
    ### Aero ###
    cw: float
    cl_front: float
    cl_rear: float
    roh_air_kgpm3: float
    a_vehicle_m2: float
    ### Limited Slip Differential ###
    ratio_lock_drive: float
    ratio_lock_coast: float
    torque_preload_nm: float
    slip_sensitivity_coeff: float
