from __future__ import print_function
import h5py
import pdb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas as pd
import os
from scipy import signal
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


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


def get_ap_amplitude(ap_data, does_plot = False):
    ap_amplitude = ap_data['Voltage (V)'].max() - ap_data['Voltage (V)'].min()

    if (does_plot):
        time_at_max = ap_data['Time (s)'].loc[ap_data['Voltage (V)'].idxmax()]
        time_at_min = ap_data['Time (s)'].loc[ap_data['Voltage (V)'].idxmin()]
        print('AP amplitude is ', ap_amplitude, ' volts')
        plot_single_ap(ap_data)
        plt.plot([time_at_max, time_at_max], [ap_data['Voltage (V)'].min(), ap_data['Voltage (V)'].max()], 'r-')
        plt.plot([time_at_min, time_at_max], [ap_data['Voltage (V)'].min(), ap_data['Voltage (V)'].max()], 'yo')

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
    voltage_90 = voltage.min() + (get_ap_amplitude(ap_data_copy) * (1 - repolarization_percent))
    time_end = time.loc[(ap_data_max_to_min['Voltage (V)'] - voltage_90).abs().idxmin()]
    ap_duration = time_end - time.loc[voltage_mid_loc]

    if does_plot:
        print('AP duration is ', ap_duration, ' seconds')
        plot_single_ap(ap_data_copy)
        plt.plot([time.loc[voltage_mid_loc], time_end], [voltage_mid, voltage_mid], 'r-')
        plt.plot([time.loc[voltage_mid_loc], time_end], [voltage_mid, voltage_90], 'yo')

    return ap_duration


def get_single_ap(ap_data, ap_number, does_plot=False):
    voltage_local_max = find_voltage_peaks(ap_data['Voltage (V)'])
    if ap_number == 0:
        ap_number = random.randint(1,(len(voltage_local_max)-1))
    cycle_start = voltage_local_max[ap_number - 1]
    single_ap_max = ap_data[cycle_start:voltage_local_max[ap_number]]
    if len(single_ap_max['Time (s)']) > 25000:
        ap_start = cycle_start - 5000
    else:
        ap_start = cycle_start - int(len(single_ap_max['Time (s)']) / 4)
    ap_data_post_max = ap_data[(ap_data['Voltage (V)'].idxmax() - ap_data['Time (s)'].idxmin()):]
    if (ap_data_post_max['Voltage (V)'].idxmin() - cycle_start) > 6000:
        ap_end = cycle_start + 7500
    else:
        ap_end = ap_start + len(single_ap_max['Time (s)'])
    single_ap = ap_data[ap_start:ap_end]

    if does_plot:
        plot_single_ap(single_ap)

    return single_ap


def get_ap_sf_points(ap_data, repolarization_percent):
    voltage = ap_data['Voltage (V)']
    time = ap_data['Time (s)']
    ap_data_post_max = ap_data[(voltage.idxmax() - time.idxmin()):(voltage.idxmin() - time.idxmin())]
    voltage_90 = voltage.min() + (get_ap_amplitude(ap_data) * (1 - repolarization_percent))
    time_end = time.loc[(ap_data_post_max['Voltage (V)'] - voltage_90).abs().idxmin()]

    return time_end, voltage_90


def get_ap_shape_factor(ap_data, does_plot=False):
    APD_30 = get_ap_duration(ap_data, 0.5, 0.3)
    APD_40 = get_ap_duration(ap_data, 0.5, 0.4)
    APD_70 = get_ap_duration(ap_data, 0.5, 0.7)
    APD_80 = get_ap_duration(ap_data, 0.5, 0.8)
    ap_shape_factor = (APD_30 - APD_40) / (APD_70 - APD_80)

    if does_plot:
        print("AP shape factor is ", ap_shape_factor)
        plot_single_ap(ap_data)
        APD_30_points = get_ap_sf_points(ap_data, .3)
        APD_40_points = get_ap_sf_points(ap_data, .4)
        APD_70_points = get_ap_sf_points(ap_data, .7)
        APD_80_points = get_ap_sf_points(ap_data, .8)
        point_voltages = [APD_30_points[1], APD_40_points[1], APD_70_points[1], APD_80_points[1]]
        point_times = [APD_30_points[0], APD_40_points[0], APD_70_points[0], APD_80_points[0]]
        plt.plot(point_times, point_voltages, 'yo')

    return ap_shape_factor


def get_cycle_lengths(ap_data, does_plot = False):
    voltage_local_max = find_voltage_peaks(ap_data['Voltage (V)'])
    cycle_lengths = []
    for x in range(len(voltage_local_max) - 1):
        cycle_time = ap_data[voltage_local_max[x]:voltage_local_max[x + 1]]['Time (s)']
        if len(cycle_time) > 25000:
            cycle_lengths.append(cycle_lengths[-1])
        else:
            cycle_lengths.append(len(cycle_time))

    if does_plot:
        cycle_lengths_copy = []
        for x in range(len(cycle_lengths)):
            cycle_lengths_copy.append(cycle_lengths[x] / 10000)
        plt.plot(cycle_lengths_copy)
        plt.xlabel('Action Potentials')
        plt.ylabel('Cycle Lengths (s)')

    return cycle_lengths


def plot_single_ap(sap_data):
    plt.plot(sap_data['Time (s)'], sap_data['Voltage (V)'])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')


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
        if random_cycle_loc == 0:
            random_cycle_loc = 1
        locs.append(random_cycle_loc)
        aps.append(get_single_ap(ap_data, random_cycle_loc))

    if does_plot:
        patches = []
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']
        for x in range(number_of_aps):
            aps_copy = zero_ap_data(aps[x].reset_index())
            plot_single_ap(aps_copy)
            patches.append(mpatches.Patch(color = colors[x], label = locs[x]))
        plt.legend(handles = patches, title = 'AP Indices')

    return aps


def get_ap_vmax(ap_data, percent_up):
    voltage = ap_data['Voltage (V)']
    time_begin = ap_data['Time (s)'].idxmin()
    ap_data_pre_max = ap_data[:(voltage.idxmax() - time_begin)]
    voltage_mid = ((voltage.max() - voltage[time_begin]) * percent_up) + voltage[time_begin]
    voltage_mid_loc = (ap_data_pre_max['Voltage (V)'] - voltage_mid).abs().idxmin()

    return voltage_mid_loc


def get_slope(ap_data, does_plot=False):
    time_start_50 = ap_data['Time (s)'].loc[get_ap_vmax(ap_data, 0.45)]
    voltage_mid_50 = ap_data['Voltage (V)'].loc[get_ap_vmax(ap_data, 0.45)]
    time_start_85 = ap_data['Time (s)'].loc[get_ap_vmax(ap_data, 0.85)]
    voltage_mid_85 = ap_data['Voltage (V)'].loc[get_ap_vmax(ap_data, 0.85)]
    time_start_25 = ap_data['Time (s)'].loc[get_ap_vmax(ap_data, 0.25)]
    voltage_mid_25 = ap_data['Voltage (V)'].loc[get_ap_vmax(ap_data, 0.25)]
    slope = (voltage_mid_85 - voltage_mid_50) / (time_start_85 - time_start_50)

    if does_plot:
        plot_single_ap(ap_data)
        x_number_values = [time_start_50, time_start_85, time_start_25]
        y_number_values = [voltage_mid_50, voltage_mid_85, voltage_mid_25]
        plt.plot(x_number_values, y_number_values, 'r-')
        plt.plot([time_start_50, time_start_85, time_start_25], [voltage_mid_50, voltage_mid_85, voltage_mid_25], 'yo')
        print('Maximum increase velocity is ', slope, ' volts per second')

    return slope


def get_all_apds(ap_data, depolarization_percent, repolarization_percent, does_plot=False):
    apds = []
    if type(ap_data) == pd.core.frame.DataFrame:
        cycle_lengths = get_cycle_lengths(ap_data)
        for x in range(1, len(cycle_lengths) + 1):
            single_ap = get_single_ap(ap_data, x)
            apds.append(get_ap_duration(single_ap, depolarization_percent, repolarization_percent))
    elif type(ap_data) == list:
        for x in range(len(ap_data)):
            apds.append(get_ap_duration(ap_data[x], depolarization_percent, repolarization_percent))

    if does_plot:
        plt.plot(apds)
        plt.xlabel('Action Potentials')
        plt.ylabel('Durations (s)')

    return apds


def get_all_apas(ap_data, does_plot = False):
    apas = []
    if type(ap_data) == pd.core.frame.DataFrame:
        cycle_lengths = get_cycle_lengths(ap_data)
        for x in range(1,len(cycle_lengths)+1):
            single_ap = get_single_ap(ap_data, x)
            apas.append(get_ap_amplitude(single_ap))
    elif type(ap_data) == list:
        for x in range(len(ap_data)):
            apas.append(get_ap_amplitude(ap_data[x]))

    if does_plot:
        plt.plot(apas)
        plt.xlabel('Action Potentials')
        plt.ylabel('Amplitudes (V)')

    return apas


def get_ap_range(ap_data, first_ap, last_ap, split, does_plot=False):
    last_ap_copy = last_ap + 1
    ap_range = get_single_ap(ap_data, first_ap)
    ap_range_singles = [get_single_ap(ap_data, first_ap)]
    for x in range(1, last_ap_copy - first_ap):
        ap_range = pd.concat([ap_range, get_single_ap(ap_data, (first_ap + x))])
        ap_range_singles.append(get_single_ap(ap_data, (first_ap + x)))

    if does_plot:
        for x in range(len(ap_range_singles)):
            plot_single_ap(ap_range_singles[x])
        patches = []
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for x in range(len(ap_range_singles)):
            current_color = x % 10
            patches.append(mpatches.Patch(color=colors[current_color], label=first_ap + x))
        plt.legend(handles = patches, loc = 'center left', bbox_to_anchor = (1.1, .5), ncol = 2, title = 'AP Indices')

    if split:
        return ap_range_singles
    else:
        return ap_range


def smooth_ap_data(ap_data, degree, does_plot=False):
    smoothed_ap_data = ap_data.copy()
    voltage = list(smoothed_ap_data['Voltage (V)'])
    window = (degree * 2) - 1
    weight = np.array([1] * window)
    weight_gauss = []
    for x in range(window):
        x = x - degree + 1
        fraction = x / float(window)
        gauss = 1 / (np.exp((4 * (fraction)) ** 2))
        weight_gauss.append(gauss)
    weight = np.array(weight_gauss) * weight
    smoothed = [0.0] * (len(voltage) - window)
    for x in range(len(smoothed)):
        smoothed[x] = sum(np.array(voltage[x : x + window]) * weight) / sum(weight)
    smoothed_length = len(smoothed)
    for x in range(window):
        smoothed.append(voltage[smoothed_length + x])
    smoothed_ap_data['Voltage (V)'] = smoothed

    if does_plot:
        plot_single_ap(smoothed_ap_data)

    return smoothed_ap_data


def compare_aps(first_ap,second_ap):
    first_ap_copy = zero_ap_data(first_ap.reset_index())
    second_ap_copy = zero_ap_data(second_ap.reset_index())
    plot_single_ap(first_ap_copy)
    plot_single_ap(second_ap_copy)
    blue_patch = mpatches.Patch(color = 'C0', label = 'First AP')
    orange_patch = mpatches.Patch(color = 'C1', label = 'Second AP')
    plt.legend(handles = [blue_patch,orange_patch])
    print('AP Durations (s):')
    print('First AP:',get_ap_duration(first_ap_copy,.5,.9),' Second AP:',get_ap_duration(second_ap_copy,.5,.9))
    print('AP Amplitudes (V):')
    print('First AP:',get_ap_amplitude(first_ap_copy),' Second AP:',get_ap_amplitude(second_ap_copy))


def no_return_sap1(ap_data, ap_number):
    get_single_ap(ap_data, ap_number, True)


def no_return_sap2(ap_data, ap_number):
    plot_single_ap(ap_data[ap_number - 1])


def plot_sap_slider(ap_data):
    if type(ap_data) == pd.core.frame.DataFrame:
        end = len(get_cycle_lengths(ap_data))
        interact(no_return_sap1, ap_data=fixed(ap_data), ap_number=(1, end))
    elif type(ap_data) == list:
        end = len(ap_data)
        interact(no_return_sap2, ap_data=fixed(ap_data), ap_number=(1, end))


def get_all_saps(ap_data):
    cycle_lengths = len(get_cycle_lengths(ap_data))
    all_aps = get_ap_range(ap_data, 1, cycle_lengths, True)

    return all_aps


def get_all_vmax(ap_data, does_plot = False):
    vmax = []
    if type(ap_data) == pd.core.frame.DataFrame:
        cycle_lengths = get_cycle_lengths(ap_data)
        for x in range(1,len(cycle_lengths)+1):
            single_ap = get_single_ap(ap_data, x)
            vmax.append(get_slope(single_ap))
    elif type(ap_data) == list:
        for x in range(len(ap_data)):
            vmax.append(get_slope(ap_data[x]))

    if does_plot:
        plt.plot(vmax)
        plt.xlabel('Action Potentials')
        plt.ylabel('Maximum Increase Velocities (V/s)')

    return vmax


def get_all_sfs(ap_data, does_plot = False):
    sfs = []
    if type(ap_data) == pd.core.frame.DataFrame:
        cycle_lengths = get_cycle_lengths(ap_data)
        for x in range(1,len(cycle_lengths)+1):
            single_ap = get_single_ap(ap_data, x)
            sfs.append(get_ap_shape_factor(single_ap))
    elif type(ap_data) == list:
        for x in range(len(ap_data)):
            sfs.append(get_ap_shape_factor(ap_data[x]))

    if does_plot:
        plt.plot(sfs)
        plt.xlabel('Action Potentials')
        plt.ylabel('Shape Factors')

    return sfs



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
