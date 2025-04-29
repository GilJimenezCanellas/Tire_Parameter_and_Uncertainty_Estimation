""" Dataclasses for the used sensor data types"""
from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass()
class CorData:
    """ Dataclass for storing correvit sensor data """
    time: jnp.array = field(default_factory=lambda: jnp.array([]))
    vel_cog_x_mps: jnp.array = field(default_factory=lambda: jnp.array([]))
    vel_cog_y_mps: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass()
class GenData:
    """ Dataclass for storing general sensor data """
    time: jnp.array = field(default_factory=lambda: jnp.array([]))
    delta_f_rad: jnp.array = field(default_factory=lambda: jnp.array([]))
    omega_wheel_fl_radps: jnp.array = field(default_factory=lambda: jnp.array([]))
    omega_wheel_fr_radps: jnp.array = field(default_factory=lambda: jnp.array([]))
    omega_wheel_rl_radps: jnp.array = field(default_factory=lambda: jnp.array([]))
    omega_wheel_rr_radps: jnp.array = field(default_factory=lambda: jnp.array([]))
    gear: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass()
class ImuData:
    """ Dataclass for storing imu sensor data """
    time: jnp.array = field(default_factory=lambda: jnp.array([]))
    acc_cog_x_mps2: jnp.array = field(default_factory=lambda: jnp.array([]))
    acc_cog_y_mps2: jnp.array = field(default_factory=lambda: jnp.array([]))
    acc_cog_z_mps2: jnp.array = field(default_factory=lambda: jnp.array([]))
    yaw_rate_radps: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass()
class FilteredData:
    """ Dataclass for storing filtered sensor data """
    gen_data: GenData = field(default_factory=GenData)
    cor_data: CorData = field(default_factory=CorData)
    imu_data: ImuData = field(default_factory=ImuData)
