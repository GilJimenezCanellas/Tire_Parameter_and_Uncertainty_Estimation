''' Helper functions for handling preprocessing in the GUI '''
import tkinter as tk

from src.utils.filter_data import preprocess_data


def update_filter_options(settings, filter_type, frame):
    ''' Update the filter options of the selected filter type '''
    for widget in frame.winfo_children():
        widget.destroy()

    def save_settings():
        ''' Save the settings of the selected filter type '''
        settings.clear()
        if filter_type == "moving_average":
            settings["kernelsize"] = int(kernelsize.get()) if kernelsize.get() else 3
        elif filter_type == "butterworth":
            settings["sampling_freq_hz"] = int(
                sampling_freq_hz.get()) if sampling_freq_hz.get() else 100
            settings["filter_order"] = int(filter_order.get()) if filter_order.get() else 3
            settings["cutoff_freq_lower"] = int(
                cutoff_freq_lower.get()) if cutoff_freq_lower.get() else 0
            settings["cutoff_freq_upper"] = int(
                cutoff_freq_upper.get()) if cutoff_freq_upper.get() else 0
            settings["butter_type"] = butter_type.get() if butter_type.get() else 'low'
        elif filter_type == "savgol":
            settings["window_length"] = int(window_length.get()) if window_length.get() else 80
            settings["order"] = int(order.get()) if order.get() else 3
        elif filter_type == "gaussian":
            settings["sigma"] = int(sigma.get()) if sigma.get() else 1
        print("Saved filter settings")

    if filter_type == "moving_average":
        tk.Label(frame, text="Kernel Size:", bg='grey',
                 fg='black').grid(row=0, column=0, sticky="w")
        kernelsize = tk.Entry(frame)
        kernelsize.grid(row=0, column=1, sticky="w")
        kernelsize.insert(0, settings.get("kernelsize", ""))

    elif filter_type == "butterworth":
        tk.Label(frame, text="Sampling Frequency (Hz):", bg='grey',
                 fg='black').grid(row=0, column=0, sticky="w")
        sampling_freq_hz = tk.Entry(frame)
        sampling_freq_hz.grid(row=0, column=1, sticky="w")
        sampling_freq_hz.insert(0, settings.get("sampling_freq_hz", ""))

        tk.Label(frame, text="Filter Order:", bg='grey',
                 fg='black').grid(row=1, column=0, sticky="w")
        filter_order = tk.Entry(frame)
        filter_order.grid(row=1, column=1, sticky="w")
        filter_order.insert(0, settings.get("filter_order", ""))

        tk.Label(frame, text="Cutoff Frequency Lower:", bg='grey',
                 fg='black').grid(row=2, column=0, sticky="w")
        cutoff_freq_lower = tk.Entry(frame)
        cutoff_freq_lower.grid(row=2, column=1, sticky="w")
        cutoff_freq_lower.insert(0, settings.get("cutoff_freq_lower", ""))

        tk.Label(frame, text="Cutoff Frequency Upper:", bg='grey',
                 fg='black').grid(row=3, column=0, sticky="w")
        cutoff_freq_upper = tk.Entry(frame)
        cutoff_freq_upper.grid(row=3, column=1, sticky="w")
        cutoff_freq_upper.insert(0, settings.get("cutoff_freq_upper", ""))

        tk.Label(frame, text="Butter Type:", bg='grey',
                 fg='black').grid(row=4, column=0, sticky="w")
        butter_type = tk.Entry(frame)
        butter_type.grid(row=4, column=1, sticky="w")
        butter_type.insert(0, settings.get("butter_type", ""))

    elif filter_type == "savgol":
        tk.Label(frame, text="Window Length:", bg='grey',
                 fg='black').grid(row=0, column=0, sticky="w")
        window_length = tk.Entry(frame)
        window_length.grid(row=0, column=1, sticky="w")
        window_length.insert(0, settings.get("window_length", ""))

        tk.Label(frame, text="Order:", bg='grey', fg='black').grid(row=1, column=0, sticky="w")
        order = tk.Entry(frame)
        order.grid(row=1, column=1, sticky="w")
        order.insert(0, settings.get("order", ""))

    elif filter_type == "gaussian":
        tk.Label(frame, text="Sigma:", bg='grey', fg='black').grid(row=0, column=0, sticky="w")
        sigma = tk.Entry(frame)
        sigma.grid(row=0, column=1, sticky="w")
        sigma.insert(0, settings.get("sigma", ""))

    print("Filter options updated")
    tk.Button(frame, text="Save", command=save_settings).grid(
        row=0, column=2, columnspan=2, sticky="w")


def update_filter_gen(config, selected_filter, filter_options_frame):
    ''' Update the general filter settings '''
    config.filter_gen = selected_filter
    update_filter_options(config.settings_gen, selected_filter, filter_options_frame)


def update_filter_cor(config, selected_filter, filter_options_frame):
    ''' Update the Correvit filter settings '''
    config.filter_cor = selected_filter
    update_filter_options(config.settings_cor, selected_filter, filter_options_frame)


def update_filter_imu(config, selected_filter, filter_options_frame):
    ''' Update the IMU filter settings '''
    config.filter_imu = selected_filter
    update_filter_options(config.settings_imu, selected_filter, filter_options_frame)


def update_filter_settings(config, filter_options_frame_gen, filter_options_frame_cor, filter_options_frame_imu):
    ''' Update all filter settings '''
    update_filter_gen(config, config.filter_gen, filter_options_frame_gen)
    update_filter_cor(config, config.filter_cor, filter_options_frame_cor)
    update_filter_imu(config, config.filter_imu, filter_options_frame_imu)
    print("Filter settings updated")


def open_preprocessing(config, data):
    ''' Open the preprocessing window '''
    prepro_window = tk.Toplevel()
    prepro_window.title("Preprocessing")
    prepro_window.geometry("800x600")
    prepro_window.grid()
    prepro_window.config(bg='grey')
    # font configuration
    prepro_window.option_add("*Font", "Arial 11")  # set default font
    font_headline = ("Arial", 16, "bold")

    tk.Label(prepro_window, text="Preprocessing",
             font=font_headline).grid(row=0, column=0, sticky="w")

    tk.Label(prepro_window, text="General Filter", font=(
        "Arial", 12, "bold")).grid(row=1, column=0, sticky="w")
    tk.Label(prepro_window, text="Filter Type:").grid(row=2, column=0, sticky="w")
    selected_filter_gen = tk.StringVar()
    selected_filter_gen.set("savgol")
    tk.OptionMenu(prepro_window, selected_filter_gen, "moving_average", "butterworth",
                  "savgol", "gaussian").grid(row=2, column=1, sticky="w")

    tk.Label(prepro_window, text="Correvit Filter", font=(
        "Arial", 12, "bold")).grid(row=3, column=0, sticky="w")
    tk.Label(prepro_window, text="Filter Type:").grid(row=4, column=0, sticky="w")
    selected_filter_cor = tk.StringVar()
    selected_filter_cor.set("savgol")
    tk.OptionMenu(prepro_window, selected_filter_cor, "moving_average", "butterworth",
                  "savgol", "gaussian").grid(row=4, column=1, sticky="w")

    tk.Label(prepro_window, text="IMU Filter", font=(
        "Arial", 12, "bold")).grid(row=5, column=0, sticky="w")
    tk.Label(prepro_window, text="Filter Type:").grid(row=6, column=0, sticky="w")
    selected_filter_imu = tk.StringVar()
    selected_filter_imu.set("savgol")
    tk.OptionMenu(prepro_window, selected_filter_imu, "moving_average", "butterworth",
                  "savgol", "gaussian").grid(row=6, column=1, sticky="w")

    filter_options_frame_gen = tk.Frame(prepro_window, bg='grey')
    filter_options_frame_gen.grid(row=2, column=3, columnspan=2, sticky="w")

    filter_options_frame_cor = tk.Frame(prepro_window, bg='grey')
    filter_options_frame_cor.grid(row=4, column=3, columnspan=2, sticky="w")

    filter_options_frame_imu = tk.Frame(prepro_window, bg='grey')
    filter_options_frame_imu.grid(row=6, column=3, columnspan=2, sticky="w")

    # Initial call to set up the default filter options
    update_filter_options(config.settings_gen, selected_filter_gen.get(), filter_options_frame_gen)
    update_filter_options(config.settings_imu, selected_filter_imu.get(), filter_options_frame_imu)
    update_filter_options(config.settings_cor, selected_filter_cor.get(), filter_options_frame_cor)

    selected_filter_gen.trace_add(
        "write", lambda *args: update_filter_gen(config, selected_filter_gen.get(), filter_options_frame_gen))
    selected_filter_cor.trace_add(
        "write", lambda *args: update_filter_cor(config, selected_filter_cor.get(), filter_options_frame_cor))
    selected_filter_imu.trace_add(
        "write", lambda *args: update_filter_imu(config, selected_filter_imu.get(), filter_options_frame_imu))

    tk.Label(prepro_window, text="Further Settings", font=(
        "Arial", 12, "bold")).grid(row=7, column=0, sticky="w")
    tk.Label(prepro_window, text="Low Speed Filter (m/s):").grid(row=8, column=0, sticky="w")
    low_speed_filter_box = tk.Entry(prepro_window)
    low_speed_filter_box.grid(row=8, column=1, sticky="w")

    def update_low_speed_filter(*args):
        ''' Update the low speed filter setting '''
        try:
            config.low_speed_filter_mps = float(low_speed_filter_box.get())
        except ValueError:
            config.low_speed_filter_mps = 5.0
            print("Invalid input for low speed filter. Setting to default value 5.0 m/s.")

    low_speed_filter_box.bind("<FocusOut>", update_low_speed_filter)

    tk.Label(prepro_window, text="Sampling Frequency (Hz):").grid(row=9, column=0, sticky="w")
    sampling_freq_box = tk.Entry(prepro_window)
    sampling_freq_box.grid(row=9, column=1, sticky="w")

    def update_sampling_freq(*args):
        try:
            config.sampling_freq_hz = int(sampling_freq_box.get())
        except ValueError:
            config.sampling_freq_hz = 100
            print("Invalid input for sampling frequency. Setting to default value 100 Hz.")

    sampling_freq_box.bind("<FocusOut>", update_sampling_freq)

    gear_change_filter_var = tk.BooleanVar()
    tk.Label(prepro_window, text="Gear Change Filter:").grid(row=10, column=0, sticky="w")
    tk.Checkbutton(prepro_window, variable=gear_change_filter_var).grid(
        row=10, column=1, sticky="w")

    def update_gear_change_filter(*args):
        config.gear_change_filter = gear_change_filter_var.get()

    gear_change_filter_var.trace_add('write', update_gear_change_filter)

    imu_acc_z_fix_var = tk.BooleanVar()
    tk.Label(prepro_window, text="IMU Acc Z Fix:").grid(row=11, column=0, sticky="w")
    tk.Checkbutton(prepro_window, variable=imu_acc_z_fix_var).grid(row=11, column=1, sticky="w")

    def update_imu_acc_z_fix(*args):
        config.imu_acc_z_fix = imu_acc_z_fix_var.get()

    imu_acc_z_fix_var.trace_add('write', update_imu_acc_z_fix)

    vhl_data_filter_var = tk.BooleanVar()
    tk.Label(prepro_window, text="VHL Data Filter:").grid(row=12, column=0, sticky="w")
    tk.Checkbutton(prepro_window, variable=vhl_data_filter_var).grid(row=12, column=1, sticky="w")

    def update_vhl_data_filter(*args):
        config.vhl_data_filter = vhl_data_filter_var.get()

    vhl_data_filter_var.trace_add('write', update_vhl_data_filter)

    def set_sensor_offsets(configal):
        ''' Set the sensor offsets '''
        def save_values():
            ''' Save the sensor offsets '''
            configal.lambda_v = float(entry_lambda_v.get())
            configal.lambda_ax = float(entry_lambda_ax.get())
            configal.lambda_ay = float(entry_lambda_ay.get())
            configal.ax_median = float(entry_ax_median.get())
            configal.ay_median = float(entry_ay_median.get())
            configal.az_off = float(entry_az_off.get())
            configal.yaw_rate_off = float(entry_yaw_rate_off.get())
            print("Sensor offsets saved")
            offset_window.destroy()

        offset_window = tk.Toplevel()
        offset_window.title("Set Sensor Offsets")

        tk.Label(offset_window, text="lambda_v").grid(row=0, column=0)
        entry_lambda_v = tk.Entry(offset_window)
        entry_lambda_v.grid(row=0, column=1)
        entry_lambda_v.insert(0, "0.0")

        tk.Label(offset_window, text="lambda_ax").grid(row=1, column=0)
        entry_lambda_ax = tk.Entry(offset_window)
        entry_lambda_ax.grid(row=1, column=1)
        entry_lambda_ax.insert(0, "0.0")

        tk.Label(offset_window, text="lambda_ay").grid(row=2, column=0)
        entry_lambda_ay = tk.Entry(offset_window)
        entry_lambda_ay.grid(row=2, column=1)
        entry_lambda_ay.insert(0, "0.0")

        tk.Label(offset_window, text="ax_median").grid(row=3, column=0)
        entry_ax_median = tk.Entry(offset_window)
        entry_ax_median.grid(row=3, column=1)
        entry_ax_median.insert(0, "0.0")

        tk.Label(offset_window, text="ay_median").grid(row=4, column=0)
        entry_ay_median = tk.Entry(offset_window)
        entry_ay_median.grid(row=4, column=1)
        entry_ay_median.insert(0, "0.0")

        tk.Label(offset_window, text="az_off").grid(row=5, column=0)
        entry_az_off = tk.Entry(offset_window)
        entry_az_off.grid(row=5, column=1)
        entry_az_off.insert(0, "0.0")

        tk.Label(offset_window, text="yaw_rate_off").grid(row=6, column=0)
        entry_yaw_rate_off = tk.Entry(offset_window)
        entry_yaw_rate_off.grid(row=6, column=1)
        entry_yaw_rate_off.insert(0, "0.0")

        tk.Button(offset_window, text="Save", command=save_values).grid(
            row=7, column=0, columnspan=2)

    tk.Label(prepro_window, text="Set Sensor Offsets:").grid(row=14, column=0, sticky="w")
    tk.Button(prepro_window, text='Set Sensor Offsets',
              command=lambda: set_sensor_offsets(config)).grid(row=14, column=1, sticky='w')

    tk.Label(prepro_window, text="Preprocess Data", font=(
        "Arial", 12, "bold")).grid(row=15, column=0, sticky="w")
    tk.Label(prepro_window, text="Preprocess Data:").grid(row=16, column=0, sticky="w")
    tk.Button(prepro_window, text='Preprocess Data',
              command=lambda: preprocess_data(config, data)).grid(row=16, column=1, sticky='w')
    prepro_window.mainloop()
