<div align="center">

<h1> Tire Parameter and Uncertainty Estimation with Bayesian Optimization</h1>

</div>

This repository provides a toolbox for the parametrization a Magic Formula tire model based on real-world driving data. The tire states are calculated based on velocity measurements (for instance from a Correvit Sensor or a Side Slip Angle Estimator), while the tire forces are calculated based on the IMU measurements. For the optimization two estimators are provided: Nelder-Mead and Stochastical-Variational-Inference.

![Tire Parameter and Uncertainty Estimation](/docs/images/Offline_Params.jpg)

## Introduction

Tire parametrization is crucial for adequate performance of vehicle dynamics models (for instance utilized in state estimators or MPCs). However, parametrization aquired from test benches is often too expensive and prone to errors as the correct temperature window as well as road-tire contact are not sufficiently covered. We, therefore, provide a toolbox to parametrize a Magic Formula Simple tire model based on real-driving data. The repository contains functions for data preprocessing, tire force and state calculation and two algorithms to optimize the tire parameters: Nelder-Mead and Stochastical Variational Inference. The latter provides the ability to assess parameter uncertainties and use fixed values in case insufficient tire excitation is present in the recorded data. A more thourough analysis can be found in the paper linked under references.

## List of components

- data_types: contains the used dataclasses within the functions
- inputs: contains the input data files -- exemplary files are included in the folders
- outputs: default folder for storing of the results -- can be changed in the configuration file
- setup: contains the configuration files of the repository
- src: contains the functions for the GUI, parameter fitting, data processing and filtering and tire state and force calculation
- main.py: main script to calculate the tire parameters
- gui_main.py: main script to open the GUI for the parametrization pipeline

## Configuration
 
All configuration files are included in the setup folder.

### config.toml

Configuration of data loading and storing, preprocessing, sensor offsets and results.

Input File Settings
- filetype: filtered / raw -- filtered contains already preprocessed data stored in a single csv file / raw contains three csv files for different sensor measurements (general, correvit, imu) to enable different sampling frequencies for different signal sources (the folder and signal structures can be seen in the exemplary csv files)
- file_path: filepath to the run_names
- run_names: dict storing all considered runs for the optimization
- signal_names: config file name of the signal names
- vehicle_parameter_names: config file name of the vehicle parameters
- model: STM / LSD-STM -- LSD-STM takes the additional yaw moment of a limited slip differential on the rear axle into account
- mode: fitting / evaluation -- fitting is used to fit a new set of parameters / evaluation is used to load parameters form the output_folder_path and assess the performance of a model with those parameters

Preprocessing:
- filter_gen: filter type general signals -- options: None, moving_average, butterworth, savgol, gaussian
- settings_gen: filter settings of the general signals stored in a dict -- more information in src/utils/filter_data: def filter_data
- filter_cor: filter type correvit signals (velocities)
- settings_cor: settings
- filter_imu: filter type imu signals (acceleration)
- settings_imu: setting
- low_speed_filter_mps: minimum vehicle velocity to be considered in the calculations -- all data below this value is cropped to ensure adequate slip calculation
- gear_change_filter: bool - filter out 0.2 seconds after each gear shift to discard the impact of drivetrain oscillations on the wheelspeed signal
- imu_acc_z_fix: bool - fixiate the vertical acceleration to 9.81 mps2 to lower the impact of imu noise
- sampling_freq_hz: set the sampling frequncy of the filtered output signal
- vhl_data_filter: bool - filter out outliers in the final force calculation based on the standard deviation of the data points (preset to double standarddeviation)
- save_filtered_data: bool - save the filtered data in the output folder

Sensor Offsets:
Definition according to F. Farroni "T.R.I.C.K.‐Tire/Road Interaction Characterization & Knowledge - A tool for the evaluation of tire and vehicle performances in outdoor test sessions"
- lambda_v: twist of the velocity measurements
- lambda_ax: twist of the longitudinal acceleration measurments (ax/az)
- lambda_ay: twist of the lateral acceleration measurements (ay/az)
- ax_median: static offset of the longitudinal acceleration measurements
- ay_median: static offset of the lateral acceleration measurements
- az_off: static offset of the vertical acceleration measurements
- yaw_rate_off: static offset of the yaw rate measurements

Fitting:
- nelder_options: dict containing nelder-mead fitting settings - maxiter: maximum number of iterations, sample_points: number of sample points used from data (equally spread over input space)
- svi_options: dict containing svi fitting settings - iter_svi_MFS: number of iterations, step_MFS: step size, sample_points: number of sample points used from data (equally spread over input space)
- params_min: lower parameter bounds considered in the fitting
- params_max: upper parameter bounds considered in the fitting
- params_init: initial values // if fit flags are set to false the parameters will be set to the values specified here
- fit_flags_front_axle_x: fit flags for the parameters: true = considered in fitting process ; false = set to init value -- ensure at least one fit flag is activated
- fit_flags_front_axle_y: fit flags for the parameters: true = considered in fitting process ; false = set to init value -- ensure at least one fit flag is activated
- fit_flags_rear_axle_x: fit flags for the parameters: true = considered in fitting process ; false = set to init value -- ensure at least one fit flag is activated
- fit_flags_rear_axle_y: fit flags for the parameters: true = considered in fitting process ; false = set to init value -- ensure at least one fit flag is activated

Results:
- output_folder_path: output folder path -- defaults to "../outputs/" if "" is selected
- output_folder: output folder name -- defaults to "default" if "" is selected
- enable_plotting: plot tire curves and parameter uncertainties
- enable_logging: store parameters in output folder

### signal_names.toml
Signal name defintion - has to match the raw measurment signal names stored in sensor_data

### vehicle_parameters.toml
Vehicle parameters

Definitions for the tire properties according to M. Schabauer et al. “Experimental Investigation and Semi-physical Modelling of the Influence of Rotational Speed on the Vertical Tyre Stiffness and Tyre Radii"

## Running the Code

The code was developed and tested with Python 3.12 on Windows and Linux OS.

Install dependencies:
pip3 install -r /path_to repository/requirements.txt

Run locally:
- configure all settings in the setup folder
- ensure correct structures of the input csv files (runnames, endings, signal names)
- run the main.py script

GUI:
- ensure correct structures of the input csv files
- run the gui_main.py script
- configure everything in the GUI / default: settings stored in setup
- load real data: select folder sensor_data (iterates over all folders in this folder) ; load filtered data: select folder filtered - iterates over all csv files

## References

If you use the provided code in your work please consider citing our paper:
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

### Contact

- [Sven Goblirsch](mailto:sven.goblirsch@tum.de)