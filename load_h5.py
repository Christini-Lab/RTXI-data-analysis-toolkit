import h5py
import pdb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy import signal
import random


# 2. Read in the file

# 3. The following function is used to get the keys of an h5 object
def get_keys(f):
    return [key for key in f.keys()]


def load_h5(path):
    return h5py.File(path, 'r')


def extract_channel_data(data_h5, trial_number):
    trial_str = f'Trial{trial_number}'
    data = data_h5[trial_str]['Synchronous Data']['Channel Data'][()]
    return data


def plot_V_and_I(data):
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout()

    axes[0].title.set_text('Voltage (V)')
    axes[0].plot(data['Time (s)'], data['Voltage (V)'])

    axes[1].title.set_text('Current (A)')
    axes[1].plot(data['Time (s)'], data['Current'])
    plt.show()


def get_time_data(data_h5, trial_number):
    total_time, period = get_time_and_period(data_h5, trial_number)
    ch_data = extract_channel_data(data_h5, trial_number)

    V = get_channel_data(ch_data, 1)
    time_array = np.arange(0, len(V)) * period

    return time_array


def get_time_and_period(data_h5, trial_number):
    start_time, end_time = start_end_time(data_h5, trial_number)
    trial_str = f'Trial{trial_number}'
    total_time = (end_time - start_time) / 1E9
    period = data_h5[trial_str]['Period (ns)'][()] / 1E9

    return total_time, period


def start_end_time(data_h5, trial_number):
    trial_str = f'Trial{trial_number}'
    start_time = data_h5[trial_str]['Timestamp Start (ns)'][()]
    end_time = data_h5[trial_str]['Timestamp Stop (ns)'][()]
    return start_time, end_time


def get_channel_data(ch_data, channel_V):
    return ch_data[:, channel_V - 1]


def get_tags(data_h5, trial_number):
    tags = data_h5['Tags']
    start_time, end_time = start_end_time(data_h5, trial_number)
    col = ['Time', 'Description']
    tag_data = pd.DataFrame(columns=col)

    for tag_n in range(1, len(tags)):
        current_tag = tags[f'Tag {tag_n}'][()][0].decode('UTF-8')
        time_of_tag = float(current_tag[0:current_tag.find(',')])
        description = current_tag[(current_tag.find(',') + 1):]
        if time_of_tag > start_time and time_of_tag < end_time:
            new_tag = [(time_of_tag - start_time) / 1E9, description]
            tag_data = tag_data.append( \
                pd.Series(new_tag, index=col), ignore_index=True)

    return tag_data

    total_time, period = get_time_and_period(data_h5, trial_number)


def subsample_data(exp_data, start_sample, end_sample):
    ix_start = np.abs(exp_data['Time (s)'] - start_sample).idxmin()
    ix_end = np.abs(exp_data['Time (s)'] - end_sample).idxmin()

    exp_data_sub = exp_data.loc[ix_start:ix_end].copy()
    exp_data_sub.loc[:, 'Time (s)'] = exp_data_sub.loc[:, 'Time (s)'] - start_sample

    return exp_data_sub


def get_exp_as_df(data_h5, trial_number):
    """I was going to save the time, voltage and current as a csv,
    but decided not to, because there can be >3million points in 
    the h5 dataset. If you want to make comparisons between trials or
    experiments, call this multiple times.
    """
    ch_data = extract_channel_data(data_h5, trial_number)
    ch_metadata = get_keys(data_h5[f'Trial{trial_number}']['Synchronous Data'])

    output_channel = int([s for s in ch_metadata if 'Output' in s][0][0])
    input_channel = int([s for s in ch_metadata if 'Input' in s][0][0])

    output_data = get_channel_data(ch_data, output_channel)
    input_data = get_channel_data(ch_data, input_channel)

    if np.mean(np.abs(output_data)) < 1E-6:
        current_data = output_data
        voltage_data = input_data
    else:
        current_data = input_data
        voltage_data = output_data

    t_data = get_time_data(data_h5, trial_number)
    d_as_frame = pd.DataFrame({'Time (s)': t_data,
                               'Voltage (V)': voltage_data,
                               'Current': current_data})
    # to save:
    #    data_path=h5_file_path[0:h5_file_path.find('.h5')-1]
    #    file_path=f'{data_path}/trial_{trial_num}.csv'
    return d_as_frame


def save_SAP_to_csv(h5_file_path, sap_data, is_returned, label=''):
    data_path = h5_file_path[0:h5_file_path.find('.h5') - 1]
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    files_in_directory = get_files_in_directory(data_path)

    max_number = 0
    for file_name in files_in_directory:
        current_number = int(file_name[file_name.find('sap_') + 4: \
                                       file_name.find('_qq')])
        if current_number > max_number:
            max_number = current_number

    file_path = f'{data_path}/sap_{max_number + 1}_qq_{label}.csv'
    sap_data.to_csv(file_path)
    print(f'Saved to {file_path}')
    if is_returned:
        return sap_data


def plot_all_aps(h5_file_path):
    data_path = h5_file_path[0:h5_file_path.find('.h5') - 1]
    sap_files = get_files_in_directory(data_path)
    fig, axes = plt.subplots(1, 1)
    fig.tight_layout()

    for sap_file in sap_files:
        ap_data = pd.read_csv(sap_file)
        ap_zeroed = zero_ap_data(ap_data)
        plt.plot(ap_zeroed['Time (s)'], ap_zeroed['Voltage (V)'])

    plt.show()


def zero_ap_data(sap_data):
    v_max_idx = sap_data['Voltage (V)'].idxmax()
    zeroed_ap = sap_data.copy()
    zeroed_ap['Time (s)'] = zeroed_ap['Time (s)'] - \
                            zeroed_ap['Time (s)'].iloc[v_max_idx]

    return zeroed_ap


def get_files_in_directory(directory):
    files_in_directory = []
    for r, d, f in os.walk(directory):
        for file in f:
            if 'sap' in file:
                files_in_directory.append(os.path.join(r, file))
    return files_in_directory


def get_ap_amplitude(ap_data, is_plotted=False):
    ap_amplitude = ap_data['Voltage (V)'].max() - ap_data['Voltage (V)'].min()

    index = ap_data['Voltage (V)'].idxmax()
    time_at_max = ap_data['Time (s)'].loc[index]
    voltage_at_max = ap_data['Voltage (V)'].max()
    voltage_at_min = ap_data['Voltage (V)'].min()
    index2 = ap_data['Voltage (V)'].idxmin()
    time_at_min = ap_data['Time (s)'].loc[index2]
    if (is_plotted):
        plt.plot(ap_data['Time (s)'], ap_data['Voltage (V)'])
        plt.plot([time_at_min], [voltage_at_min], marker='o', markersize=5, color='red')
        plt.plot([time_at_max], [voltage_at_max], marker='o', markersize=5, color='red')
        plt.plot([time_at_max, time_at_max], [voltage_at_min, voltage_at_max], 'g-')
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage(V)')
    return ap_amplitude


def get_ap_duration(sap_data, depolarization_percent, repolarization_percent, does_plot=False):
    ap_data_copy = sap_data.reset_index()
    voltage = ap_data_copy['Voltage (V)']
    time = ap_data_copy['Time (s)']
    ap_data_pre_max = ap_data_copy[:(voltage.idxmax() - time.idxmin())]
    ap_data_post_max = ap_data_copy[(voltage.idxmax() - time.idxmin()):]
    ap_data_max_to_min = ap_data_post_max[:ap_data_post_max['Voltage (V)'].idxmin()]
    voltage_mid = ((voltage.max() - voltage[time.idxmin()]) * depolarization_percent) + voltage[time.idxmin()]
    voltage_mid_loc = (ap_data_pre_max['Voltage (V)'] - voltage_mid).abs().idxmin()
    amplitude = get_ap_amplitude(ap_data_copy)
    voltage_90 = voltage.min() + (amplitude * (1 - repolarization_percent))
    voltage_90_loc = (ap_data_max_to_min['Voltage (V)'] - voltage_90).abs().idxmin()
    time_start = time.loc[voltage_mid_loc]
    time_end = time.loc[voltage_90_loc]
    ap_duration = time_end - time_start

    if does_plot:
        print('Action Potential Duration is', ap_duration)
        plot_single_ap(ap_data_copy)
        plt.plot([time_start, time_end], [voltage_mid, voltage_mid], 'r-')
        plt.plot([time_start, time_end], [voltage_mid, voltage_90], 'yo')

    return ap_duration


def get_single_ap(ap_data, ap_number, does_plot=False):
    voltage = ap_data['Voltage (V)']
    time = ap_data['Time (s)']
    voltage_local_max = find_voltage_peaks(voltage)
    if ap_number == 0:
        ap_number = random.randint(1,(len(voltage_local_max)-1))
    cycle_start = voltage_local_max[ap_number - 1]
    cycle_end = voltage_local_max[ap_number]
    single_ap_max = ap_data[cycle_start:cycle_end]
    cycle_time = single_ap_max['Time (s)']
    cycle_length = len(cycle_time)
    if cycle_length > 25000:
        ap_start = cycle_start - 5000
    else:
        ap_start = cycle_start - int(cycle_length / 4)
    ap_data_post_max = ap_data[(voltage.idxmax() - time.idxmin()):]
    if (ap_data_post_max['Voltage (V)'].idxmin() - cycle_start) > 6000:
        ap_end = cycle_start + 7500
    else:
        ap_end = ap_start + cycle_length
    single_ap = ap_data[ap_start:ap_end]

    if does_plot:
        plot_single_ap(single_ap)

    return single_ap


def get_ap_shape_factor_points(ap_data, repolarization_percent, does_plot=False):
    voltage = ap_data['Voltage (V)']
    time = ap_data['Time (s)']
    ap_data_post_max = ap_data[(voltage.idxmax() - time.idxmin()):(voltage.idxmin() - time.idxmin())]
    amplitude = get_ap_amplitude(ap_data)
    voltage_90 = voltage.min() + (amplitude * (1 - repolarization_percent))
    voltage_90_loc = (ap_data_post_max['Voltage (V)'] - voltage_90).abs().idxmin()
    time_end = time.loc[voltage_90_loc]

    if does_plot:
        plot_single_ap(ap_data)
        plt.plot([time_end], [voltage_90], 'yo')
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage(V)')

    return time_end, voltage_90


def get_ap_shape_factor(ap_data):
    APD_30 = get_ap_duration(ap_data, 0.3)
    APD_40 = get_ap_duration(ap_data, 0.4)
    APD_70 = get_ap_duration(ap_data, 0.7)
    APD_80 = get_ap_duration(ap_data, 0.8)
    ap_shape_factor = (APD_30 - APD_40) / (APD_70 - APD_80)
    return ap_shape_factor


def get_cycle_lengths(ap_data, does_plot = False):
    voltage = ap_data['Voltage (V)']
    voltage_local_max = find_voltage_peaks(voltage)
    cycle_lengths = []
    for x in range(len(voltage_local_max) - 1):
        cycle_start = voltage_local_max[x]
        cycle_end = voltage_local_max[x + 1]
        single_ap_max = ap_data[cycle_start:cycle_end]
        cycle_time = single_ap_max['Time (s)']
        if len(cycle_time) > 25000:
            cycle_lengths.append(cycle_lengths[-1])
        else:
            cycle_lengths.append(len(cycle_time))

    if does_plot:
        plt.plot(cycle_lengths)

    return cycle_lengths


def plot_single_ap(sap_data):
    plt.plot(sap_data['Time (s)'], sap_data['Voltage (V)'])

def find_voltage_peaks(voltage):
    voltage_peaks = np.ndarray.tolist(list(signal.find_peaks(voltage, distance=5000, prominence=.03, height=0))[0])

    return voltage_peaks

def get_various_aps(ap_data, does_plot=False):
    cycle_lengths = get_cycle_lengths(ap_data)
    if len(cycle_lengths) < 5:
        number_of_aps = len(cycle_lengths)
    else:
        number_of_aps = 5
    aps = []
    locs = []
    bloc_size = len(cycle_lengths) / number_of_aps
    for x in range(number_of_aps):
        random_cycle_loc = int(random.uniform((bloc_size * x), (bloc_size * (x + 1))))
        locs.append(random_cycle_loc)
        aps.append(get_single_ap(ap_data, random_cycle_loc))

    if does_plot:
        for x in range(number_of_aps):
            aps_copy = zero_ap_data(aps[x].reset_index())
            plot_single_ap(aps_copy)
    print(locs)

    return aps


def get_ap_vmax(ap_data, percent_up):
    time = ap_data['Time (s)']
    voltage = ap_data['Voltage (V)']
    time_begin = ap_data['Time (s)'].idxmin()
    ap_data_pre_max = ap_data[:(voltage.idxmax() - time_begin)]
    ap_data_post_max = ap_data[(voltage.idxmax() - time_begin):(voltage.idxmin() - time_begin)]
    voltage_start = voltage[time_begin]
    voltage_mid = ((voltage.max() - voltage_start) * percent_up) + voltage_start
    voltage_mid_loc = (ap_data_pre_max['Voltage (V)'] - voltage_mid).abs().idxmin()

    return voltage_mid_loc


def plot_ap_vmax(ap_data, is_plotted=False):
    time = ap_data['Time (s)']
    voltage = ap_data['Voltage (V)']

    time_start_50 = time.loc[get_ap_vmax(ap_data, 0.45)]
    voltage_mid_50 = voltage.loc[get_ap_vmax(ap_data, 0.45)]
    time_start_85 = time.loc[get_ap_vmax(ap_data, 0.85)]
    voltage_mid_85 = voltage.loc[get_ap_vmax(ap_data, 0.85)]
    time_start_25 = time.loc[get_ap_vmax(ap_data, 0.25)]
    voltage_mid_25 = voltage.loc[get_ap_vmax(ap_data, 0.25)]

    if is_plotted:
        plt.plot(ap_data['Time (s)'], ap_data['Voltage (V)'])
        plt.plot([time_start_50], [voltage_mid_50], 'yo')
        plt.plot([time_start_85], [voltage_mid_85], 'yo')
        plt.plot([time_start_25], [voltage_mid_25], 'yo')


def draw_line(ap_data, is_plotted=False):
    time = ap_data['Time (s)']
    voltage = ap_data['Voltage (V)']
    time_start_50 = time.loc[get_ap_vmax(ap_data, 0.45)]
    voltage_mid_50 = voltage.loc[get_ap_vmax(ap_data, 0.45)]
    time_start_85 = time.loc[get_ap_vmax(ap_data, 0.85)]
    voltage_mid_85 = voltage.loc[get_ap_vmax(ap_data, 0.85)]
    time_start_25 = time.loc[get_ap_vmax(ap_data, 0.25)]
    voltage_mid_25 = voltage.loc[get_ap_vmax(ap_data, 0.25)]

    x_number_values = [time_start_50, time_start_85, time_start_25]
    y_number_values = [voltage_mid_50, voltage_mid_85, voltage_mid_25]

    if is_plotted:
        plt.plot(x_number_values, y_number_values, linewidth=2)
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage(V)')
        plt.show


def get_slope(ap_data):
    time = ap_data['Time (s)']
    voltage = ap_data['Voltage (V)']
    time_start_50 = time.loc[get_ap_vmax(ap_data, 0.45)]
    voltage_mid_50 = voltage.loc[get_ap_vmax(ap_data, 0.45)]
    time_start_85 = time.loc[get_ap_vmax(ap_data, 0.85)]
    voltage_mid_85 = voltage.loc[get_ap_vmax(ap_data, 0.85)]
    time_start_25 = time.loc[get_ap_vmax(ap_data, 0.25)]
    voltage_mid_25 = voltage.loc[get_ap_vmax(ap_data, 0.25)]

    m = (voltage_mid_85 - voltage_mid_50) / (time_start_85 - time_start_50)
    return m

def get_all_apds(ap_data, depolarization_percent, repolarization_percent, does_plot = False):
    cycle_lengths = get_cycle_lengths(ap_data)
    apds = []
    for x in range(len(1,cycle_lengths)+1):
        single_ap = get_single_ap(ap_data, x)
        apds.append(get_ap_duration(single_ap, depolarization_percent, repolarization_percent))

    if does_plot:
        plt.plot(apds)

    return apds

def get_all_apas(ap_data, does_plot = False):
    cycle_lengths = get_cycle_lengths(ap_data)
    apas = []
    for x in range(1,len(cycle_lengths)+1):
        single_ap = get_single_ap(ap_data, x)
        apas.append(get_ap_amplitude(single_ap))

    if does_plot:
        plt.plot(apas)

    return apas

filename = 'data/attempt_2_071519.h5'
# plot_all_aps(filename)
# trial_number=6
#
# data_h5=load_h5(filename)
#
# exp_data=get_exp_as_df(data_h5,trial_number)
# plot_V_and_I(exp_data)
# t_data=get_time_data(data_h5,trial_number)
# exp_data_sub=subsample_data(exp_data,78,85)
# tags=get_tags(data_h5,trial_number)
# label='stim'
# save_SAP_to_csv(filename,exp_data_sub,label)
#
# plt.plot(exp_data_sub['Time (s)'], exp_data_sub['Voltage (V)'])
# plt.show()
#
#
#
# file_path=f'{data_path}/sap_{max_number+1}_qq_{label}.csv'

# ch_data=extract_channel_data(data_h5,trial_number)
# voltage_channel=2
# current_channel=1
