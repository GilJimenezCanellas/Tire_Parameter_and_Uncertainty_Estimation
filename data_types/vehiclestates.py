''' Dataclasses for the used vehicle states '''
from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass
class Forces:
    ''' Dataclass for force vectors '''
    force_x_n: jnp.array = field(default_factory=lambda: jnp.array([]))
    force_y_n: jnp.array = field(default_factory=lambda: jnp.array([]))
    force_z_n: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass
class TireStates():
    ''' Dataclass for tire states '''
    sigma_x: jnp.array = field(default_factory=lambda: jnp.array([]))
    sigma_y: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass
class STMStates():
    ''' Dataclass for STM states '''
    beta: jnp.array = field(default_factory=lambda: jnp.array([]))
    dd_psi: jnp.array = field(default_factory=lambda: jnp.array([]))
    front_axle: TireStates = field(default_factory=TireStates)
    rear_axle: TireStates = field(default_factory=TireStates)
    wheel_fl: TireStates = field(default_factory=TireStates)
    wheel_fr: TireStates = field(default_factory=TireStates)
    wheel_rl: TireStates = field(default_factory=TireStates)
    wheel_rr: TireStates = field(default_factory=TireStates)
    radius_dyn_front_m: jnp.array = field(
        default_factory=lambda: jnp.array([]))
    radius_dyn_rear_m: jnp.array = field(default_factory=lambda: jnp.array([]))


@dataclass
class STMForces:
    ''' Dataclass for STM forces '''
    cog: Forces = field(default_factory=Forces)
    front_axle: Forces = field(default_factory=Forces)
    rear_axle: Forces = field(default_factory=Forces)
    wheel_fl: Forces = field(default_factory=Forces)
    wheel_fr: Forces = field(default_factory=Forces)
    wheel_rl: Forces = field(default_factory=Forces)
    wheel_rr: Forces = field(default_factory=Forces)
