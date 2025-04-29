'''Functions for utilized tire models'''
import jax.numpy as jnp

from data_types.vehicleparameters import MFSimpleParams, MFCombinedParams


def mf_simple(slip: jnp.array, load_n: jnp.array, mfsimple_params: MFSimpleParams):
    '''Magic Formula Simple Tire Model'''
    slip_eff = slip - mfsimple_params.S_H
    tire_force = load_n * (mfsimple_params.D * jnp.sin(mfsimple_params.C * jnp.arctan(mfsimple_params.B *
                                                                                      slip_eff - mfsimple_params.E * (mfsimple_params.B * slip_eff - jnp.arctan(mfsimple_params.B * slip_eff)))))
    tire_force_eff = tire_force + mfsimple_params.S_V
    return tire_force_eff

def tire_model(tire_model_name: str, slip: jnp.array, load_n: jnp.array, tire_params):
    '''Wrapper for tire models'''
    if tire_model_name == 'MFSimple':
        return mf_simple(slip, load_n, tire_params)
    raise ValueError('Unknown tire model: ' + tire_model_name)
