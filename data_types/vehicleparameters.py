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
    wheel_fl_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    wheel_fr_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    wheel_rl_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    wheel_rr_x: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    front_axle_y: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))
    rear_axle_y: MFSimpleParams = field(
        default_factory=lambda: MFSimpleParams(0, 0, 0, 0, 0, 0))


@dataclass
class VhlParams:
    ''' Dataclass for storing vehicle parameters '''
    ### Vehicle Measurements ###
    l_front_m: float
    l_rear_m: float
    tw_front_m: float
    cog_z_m: float
    tw_rear_m: float
    ### Vehicle Intertia ###
    mass_kg: float
    izz_kgm2: float
    ### Tire Properties ###
    r_tire_unloaded_front_m: float
    r_tire_unloaded_rear_m: float
    gear_ratio: float
    wheel_inertia_kgm2: float
    ### Aero ###
    cw: float
    cl_front: float
    cl_rear: float
    roh_air_kgpm3: float
    a_vehicle_m2: float
    ### Data Acquisition ###
    sampling_freq_hz: float = 0.0
