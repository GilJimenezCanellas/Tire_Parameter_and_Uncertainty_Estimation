''' Functions to fit the parameters of the tire models to the experimental data. '''
from jax import numpy as jnp, random as jr, local_device_count
from scipy import spatial as scp
from scipy.optimize import minimize
from numpyro import sample, param, plate, set_host_device_count, distributions as dist
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO

from data_types.vehicleparameters import MFSimpleParams

from src.utils.tiremodels import tire_model as tire


def svi_fitting(tire_model: str, sigma: jnp.array, force_n: jnp.array, load_n: jnp.array,
                params_init: MFSimpleParams, params_min: MFSimpleParams, params_max: MFSimpleParams,
                options: dict, fit_flags: dict):
    """Function to fit the tire parameters using the SVI method."""
    def model_mfs(sigma: jnp.array, force_obs: jnp.array = None):
        """Model SVI: Magic Formula Simple tire model"""
        B = sample("B", dist.Uniform(5, 40)) if fit_flags.get("B", True) else params_init.B
        C = sample("C", dist.Uniform(1, 3)) if fit_flags.get("C", True) else params_init.C
        D = sample("D", dist.Uniform(0.1, 2)) if fit_flags.get("D", True) else params_init.D
        E = sample("E", dist.Uniform(-1, 1)) if fit_flags.get("E", True) else params_init.E
        noise_data = param("sigma", 0.1, constraint=dist.constraints.positive)
        tire_params = MFSimpleParams(B=B, C=C, D=D, E=E, S_H=params_init.S_H, S_V=params_init.S_V)
        force_pred = tire("MFSimple", sigma, load_n, tire_params)
        with plate("data", len(sigma)):
            return sample("obs", dist.Normal(force_pred, noise_data), obs=force_obs)

    def guide_mfs(sigma, force_obs=None):
        """Guide SVI: Magic Formula Simple tire model"""
        param_names = [key for key, value in fit_flags.items() if value]
        # Generate the constrained mean vector and covariance matrix
        mean_vector = jnp.array(
            [getattr(params_init, name) for name in param_names]
        )
        lower_bounds = jnp.array([getattr(params_min, name) for name in param_names])
        upper_bounds = jnp.array([getattr(params_max, name) for name in param_names])
        mean_vector = param(
            "mean_vector", mean_vector, constraint=dist.constraints.interval(lower_bounds, upper_bounds)
        )
        cov_matrix = param(
            "cov_matrix", jnp.eye(len(param_names)), constraint=dist.constraints.positive_definite
        )
        params = sample("params", dist.MultivariateNormal(
            mean_vector, cov_matrix), infer={"is_auxiliary": True})
        # Unpack sampled parameters and assign to their respective names
        for i, name in enumerate(param_names):
            sample(name, dist.Delta(params[i]))
        noise_data = param("sigma", 0.1, constraint=dist.constraints.positive)

    svi_mfs = SVI(model_mfs, guide_mfs, Adam(options["step_MFS"]), loss=Trace_ELBO(num_particles=1))
    svi_result_mfs_fn = svi_mfs.run(jr.PRNGKey(0), options["iter_svi_MFS"], sigma, force_n)
    # Retrieve fitted parameters and standard deviations
    mean_vector = svi_result_mfs_fn.params["mean_vector"]
    cov_matrix = svi_result_mfs_fn.params["cov_matrix"]
    loss = svi_result_mfs_fn.losses
    fitted_params = MFSimpleParams()
    std_devs = MFSimpleParams()
    param_names = [key for key, value in fit_flags.items() if value]
    for i, name in enumerate(param_names):
        setattr(fitted_params, name, mean_vector[i].item())
        setattr(std_devs, name, jnp.sqrt(cov_matrix[i, i]).item())
    all_params = ["B", "C", "D", "E"]
    for name in all_params:
        if name not in param_names:
            setattr(fitted_params, name, getattr(params_init, name))
            setattr(std_devs, name, 0.0)
    setattr(fitted_params, "S_H", getattr(params_init, "S_H"))
    setattr(fitted_params, "S_V", getattr(params_init, "S_V"))
    setattr(std_devs, "S_H", 0.0)
    setattr(std_devs, "S_V", 0.0)
    return fitted_params, std_devs, loss


def nelder_fitting(tire_model: str, sigma: jnp.array, force_n: jnp.array, load_n: jnp.array,
                   params_init: MFSimpleParams, params_min: MFSimpleParams, params_max: MFSimpleParams,
                   options: dict, fit_flags: dict):
    ''' Function to fit the tire parameters using the Nelder-Mead method. '''
    start = [getattr(params_init, key) for key in fit_flags if fit_flags[key]]
    bnds = [(getattr(params_min, key), getattr(params_max, key))
            for key in fit_flags if fit_flags[key]]

    def cost(x, loc_data, loc_sigma, loc_load_n, loc_params_init, loc_fit_flags):
        ''' Cost function to minimize. '''
        params = dict(zip(loc_fit_flags.keys(), x))
        B = params.get('B', params_init.B)
        C = params.get('C', params_init.C)
        D = params.get('D', params_init.D)
        E = params.get('E', params_init.E)
        sigma_shift = loc_sigma - loc_params_init.S_H
        forces_calc = loc_load_n * D * \
            jnp.sin(C * jnp.arctan(B * sigma_shift - E *
                    (B * sigma_shift - jnp.arctan(B * sigma_shift))))
        force = forces_calc + loc_params_init.S_V
        return jnp.sum((loc_data - force) ** 2)
    options_nelder = {'maxiter': options["maxiter"]}
    results_y = minimize(cost, x0=start, args=(force_n, sigma, load_n, params_init, fit_flags),
                         bounds=bnds, method='Nelder-Mead', options=options_nelder)
    fitted_params = MFSimpleParams()

    param_names = [key for key, value in fit_flags.items() if value]
    for i, name in enumerate(param_names):
        setattr(fitted_params, name, results_y.x[i].item())
    all_params = ["B", "C", "D", "E"]
    for name in all_params:
        if name not in param_names:
            setattr(fitted_params, name, getattr(params_init, name))
    setattr(fitted_params, "S_H", params_init.S_H)
    setattr(fitted_params, "S_V", params_init.S_V)

    return fitted_params, None, None


def tire_param_fitting(algorithm: str, tire_model: str, sigma: jnp.array, force_n: jnp.array, load_n: jnp.array,
                       params_init: MFSimpleParams, params_min: MFSimpleParams, params_max: MFSimpleParams,
                       options: dict, fit_flags: dict):
    ''' Function to fit the tire parameters using a specified algorithm. '''
    if not any(fit_flags.values()):
        raise ValueError("No parameters to fit - check fit_flags settings.")
    if len(sigma) == 0:
        raise ValueError("No samples available for fitting.")
    # sort sigma, force and load and only select samples to generate an evenly spaced input over the sigma values
    sort_idx = jnp.argsort(sigma)
    sigma = jnp.array(sigma[sort_idx])
    force_n = jnp.array(force_n[sort_idx])
    load_n = jnp.array(load_n[sort_idx])
    sample_points = min(int(options["sample_points"]), len(sigma))
    if sample_points < len(sigma):
        # define reference sigma values
        sigma_ref = jnp.linspace(sigma[0], sigma[-1], sample_points)
        # find nearest values in the reference sigma values
        tree = scp.cKDTree(sigma.reshape(-1, 1))
        _, idx = tree.query(sigma_ref.reshape(-1, 1), k=1)
        idx = idx.flatten()
        sigma = sigma[idx]
        force_n = force_n[idx]
        load_n = load_n[idx]
    set_host_device_count(local_device_count())
    if algorithm == 'SVI':
        return svi_fitting(tire_model, sigma, force_n, load_n, params_init, params_min, params_max, options, fit_flags)
    if algorithm == 'Nelder':
        return nelder_fitting(tire_model, sigma, force_n, load_n, params_init, params_min, params_max, options, fit_flags)
    raise ValueError('Unknown fitting algorithm: ' + algorithm)


def calc_force_shift(sigma: jnp.array, force_n: jnp.array):
    ''' Function to calculate the shift of the tire curve. Returns the vertical and horizontal shift. '''
    sigma_bound = 0.015
    mask = jnp.abs(sigma) <= sigma_bound
    if not jnp.any(mask):
        raise ValueError("No values within the sigma bound.")
    coeff = jnp.polyfit(sigma[mask], force_n[mask], 1)
    horizontal_shift = float(jnp.roots(coeff)[0].real / 2)
    vertical_shift = float(jnp.polyval(coeff, 0) / 2)
    return vertical_shift, horizontal_shift
