# Tire Parameter and Uncertainty Estimation Library

This branch keeps the tire parameter estimation code as an importable library. Runtime entry points, input data, output folders, estimator configs, and vehicle parameter files are intentionally owned by the integrating repository. For `autonomous_2026`, those files live under `tools/sys_id/tires/`.

The library provides:

- dataclasses in `data_types/`
- tire state, force, filtering, fitting, and plotting utilities in `src/utils/` and `src/param_fitting/`
- AMZ MATLAB and ROS 2 MCAP data mapping plus fitting helpers in `src/amz_pipeline.py`

It intentionally does not provide top-level main scripts, bundled input data, bundled output data, or default setup TOMLs. Callers must pass explicit config, vehicle parameter, input, and output paths.

## AMZ Integration API

`src.amz_pipeline` exposes the reusable AMZ pipeline pieces used by `autonomous_2026`:

- `build_filtered_data(data_file, conf, vhl_params)` maps an AMZ `*_data.mat` file into the estimator dataclasses.
- `build_filtered_data_from_rosbag(data_file, conf, vhl_params, ...)` maps an AMZ ROS 2 MCAP bag into the same dataclasses. It reads `/vcu/nera/velocity_estimation` and `/vcu/nera/steering_feedback`. Because current bags do not contain wheel-speed or force feedback, it uses synthesized rolling wheel speeds for lateral state calculation, sets wheel `Fx` to zero for force reconstruction, and rejects longitudinal tire fitting.
- `fit_tire_parameters(conf, sensordata, vhl_params=None)` fits the configured tire model. By default it fits only `front_axle_y` and `rear_axle_y`; set `conf.fit_longitudinal = True` to also fit the four wheel longitudinal targets.
- `plot_lateral_estimation(...)` and `plot_longitudinal_estimation(...)` provide optional diagnostics.

Set `conf.skidpad_mode = True` to add an initial `abs(ay)` threshold and select the least-transient fit samples globally instead of enforcing equal sample quotas across slip-angle regions.

The caller is responsible for loading `conf` and `vhl_params`, choosing output paths, writing results, and deciding whether plots should be shown.

## Dependencies

Install the pinned Python dependencies from `requirements.txt`. In `autonomous_2026`, this is done through `tools/sys_id/tires/requirements.txt`, which delegates to this file from the submodule checkout.

MCAP input also needs a sourced ROS 2 environment with `rosbag2_py`, `rclpy`, `rosidl_runtime_py`, and the generated AMZ message packages used by the bag. Build and source those message packages with the same ROS distribution and Python version used by the fit process.

## References

If you use the provided code in your work, consider citing:

- [Bayesian Optimization-based Tire Parameter and Uncertainty Estimation for Real-World Data](https://arxiv.org/abs/2504.20863)

```bibtex
@misc{goblirsch2025,
      title={Bayesian Optimization-based Tire Parameter and Uncertainty Estimation for Real-World Data},
      author={Sven Goblirsch and Benedikt Ruhland and Johannes Betz and Markus Lienkamp},
      year={2025},
      eprint={2504.20863},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.20863},
}
```
