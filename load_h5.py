import h5py
import pdb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os


# 2. Read in the file

# 3. The following function is used to get the keys of an h5 object
def get_keys(f):
    return [key for key in f.keys()]

def load_h5(path):
    return h5py.File(path,'r')

def extract_channel_data(data_h5,trial_number):
    trial_str=f'Trial{trial_number}'
    data=data_h5[trial_str]['Synchronous Data']['Channel Data'][()]
    return data

def plot_V_and_I(data):
    fig, axes=plt.subplots(2,1)
    fig.tight_layout()

    axes[0].title.set_text('Voltage (V)')
    axes[0].plot(data['Time (s)'], data['Voltage (V)'])

    axes[1].title.set_text('Current (A)')
    axes[1].plot(data['Time (s)'], data['Current'])
    plt.show()

def get_time_data(data_h5, trial_number):
    total_time, period=get_time_and_period(data_h5,trial_number)
    ch_data=extract_channel_data(data_h5,trial_number)

    V=get_channel_data(ch_data,1)
    time_array=np.arange(0,len(V))*period

    return time_array
    
def get_time_and_period(data_h5, trial_number):
    start_time, end_time=start_end_time(data_h5,trial_number)
    trial_str=f'Trial{trial_number}'
    total_time=(end_time-start_time)/1E9
    period=data_h5[trial_str]['Period (ns)'][()]/1E9

    return total_time, period

def start_end_time(data_h5,trial_number):
    trial_str=f'Trial{trial_number}'
    start_time=data_h5[trial_str]['Timestamp Start (ns)'][()]
    end_time=data_h5[trial_str]['Timestamp Stop (ns)'][()]
    return start_time, end_time

def get_channel_data(ch_data, channel_V):
    return ch_data[:,channel_V-1]

def get_tags(data_h5,trial_number):
    tags=data_h5['Tags']
    start_time, end_time=start_end_time(data_h5,trial_number)
    col=['Time', 'Description']
    tag_data=pd.DataFrame(columns=col)

    for tag_n in range(1,len(tags)):
        current_tag=tags[f'Tag {tag_n}'][()][0].decode('UTF-8')
        time_of_tag=float(current_tag[0:current_tag.find(',')])
        description=current_tag[(current_tag.find(',')+1):]
        if time_of_tag > start_time and time_of_tag < end_time:
            new_tag=[(time_of_tag-start_time)/1E9,description]
            tag_data=tag_data.append(\
                    pd.Series(new_tag,index=col),ignore_index=True)

    return tag_data


    
    total_time,period=get_time_and_period(data_h5,trial_number)

def subsample_data(exp_data,start_sample,end_sample):
    ix_start=np.abs(exp_data['Time (s)']- start_sample).idxmin()
    ix_end=np.abs(exp_data['Time (s)'] - end_sample).idxmin()

    exp_data_sub=exp_data.loc[ix_start:ix_end].copy()
    exp_data_sub.loc[:,'Time (s)']=exp_data_sub.loc[:,'Time (s)']-start_sample

    return exp_data_sub

def get_exp_as_df(data_h5,trial_number):
    """I was going to save the time, voltage and current as a csv,
    but decided not to, because there can be >3million points in 
    the h5 dataset. If you want to make comparisons between trials or
    experiments, call this multiple times.
    """
    ch_data=extract_channel_data(data_h5,trial_number)
    ch_metadata=get_keys(data_h5[f'Trial{trial_number}']['Synchronous Data'])
    
    output_channel=int([s for s in ch_metadata if 'Output' in s][0][0])
    input_channel=int([s for s in ch_metadata if 'Input' in s][0][0])

    output_data=get_channel_data(ch_data,output_channel)
    input_data=get_channel_data(ch_data,input_channel)

    if np.mean(np.abs(output_data)) < 1E-6:
        current_data=output_data
        voltage_data=input_data
    else:
        current_data=input_data
        voltage_data=output_data

    t_data=get_time_data(data_h5,trial_number)
    d_as_frame=pd.DataFrame({'Time (s)':t_data, 
                             'Voltage (V)':voltage_data, 
                             'Current':current_data})
# to save:
#    data_path=h5_file_path[0:h5_file_path.find('.h5')-1]
#    file_path=f'{data_path}/trial_{trial_num}.csv'
    return d_as_frame 

def save_SAP_to_csv(h5_file_path,sap_data,is_returned,label=''):
    data_path=h5_file_path[0:h5_file_path.find('.h5')-1]
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    files_in_directory=get_files_in_directory(data_path)

    max_number=0
    for file_name in files_in_directory:
        current_number=int(file_name[file_name.find('sap_')+4: \
                                 file_name.find('_qq')])
        if current_number > max_number:
            max_number=current_number

    file_path=f'{data_path}/sap_{max_number+1}_qq_{label}.csv'
    sap_data.to_csv(file_path)
    print(f'Saved to {file_path}')
    if is_returned:
        return sap_data

def plot_all_aps(h5_file_path):
    data_path=h5_file_path[0:h5_file_path.find('.h5')-1]
    sap_files=get_files_in_directory(data_path)
    fig,axes=plt.subplots(1,1)
    fig.tight_layout()

    for sap_file in sap_files:
        ap_data=pd.read_csv(sap_file)
        ap_zeroed=zero_ap_data(ap_data)
        plt.plot(ap_zeroed['Time (s)'], ap_zeroed['Voltage (V)'])

    plt.show()

def zero_ap_data(sap_data):
    v_max_idx=sap_data['Voltage (V)'].idxmax()
    zeroed_ap=sap_data.copy()
    zeroed_ap['Time (s)']=zeroed_ap['Time (s)']-\
            zeroed_ap['Time (s)'].iloc[v_max_idx]

    return zeroed_ap
  
def get_files_in_directory(directory):
    files_in_directory=[]
    for r,d,f in os.walk(directory):
        for file in f:
            if 'sap' in file:
                files_in_directory.append(os.path.join(r,file))
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
        plt.plot([time_at_min], [voltage_at_min], marker='o', markersize=10, color='red')
        plt.plot([time_at_max], [voltage_at_max], marker='o', markersize=10, color='red')
        plt.plot([time_at_max, time_at_max], [voltage_at_min, voltage_at_max], 'g-')
        plt.xlabel('Time(s)')
        plt.ylabel('Voltage(V)')
    return ap_amplitude

def get_ap_duration(ap_data, repolarization_percent, does_plot=False):

    voltage = ap_data['Voltage (V)']
    time = ap_data['Time (s)']
    voltage_max = voltage.max()
    voltage_max_loc = voltage.idxmax()
    voltage_min = voltage.min()
    voltage_min_loc = voltage.idxmin()
    time_begin = time.idxmin()
    ap_data_pre_max = ap_data[:(voltage_max_loc-time_begin)]
    ap_data_post_max = ap_data[(voltage_max_loc-time_begin):(voltage_min_loc-time_begin)]
    voltage_start = voltage[time_begin]
    voltage_mid = ((voltage_max-voltage_start)/2)+voltage_start
    voltage_mid_loc = (ap_data_pre_max['Voltage (V)']-voltage_mid).abs().idxmin()
    amplitude = get_ap_amplitude(ap_data)
    voltage_90 = voltage_min+(amplitude*(1-repolarization_percent))
    voltage_90_loc = (ap_data_post_max['Voltage (V)']-voltage_90).abs().idxmin()
    time_start = time.loc[voltage_mid_loc]
    time_end = time.loc[voltage_90_loc]
    ap_duration = time_end-time_start

    if does_plot:
        print(ap_data)
        print('Action Potential Duration is',ap_duration)
        plt.plot(ap_data['Time (s)'],ap_data['Voltage (V)'])
        plt.plot([time_start,time_end],[voltage_mid,voltage_mid],'r-')
        plt.plot([time_start,time_end],[voltage_mid,voltage_90],'yo')


    return ap_duration




filename = 'data/attempt_2_071519.h5'
#plot_all_aps(filename)
#trial_number=6
#
#data_h5=load_h5(filename)
#
#exp_data=get_exp_as_df(data_h5,trial_number)
#plot_V_and_I(exp_data)
#t_data=get_time_data(data_h5,trial_number)
#exp_data_sub=subsample_data(exp_data,78,85)
#tags=get_tags(data_h5,trial_number)
#label='stim'
#save_SAP_to_csv(filename,exp_data_sub,label)
#
#plt.plot(exp_data_sub['Time (s)'], exp_data_sub['Voltage (V)'])
#plt.show()
#
#
#
#file_path=f'{data_path}/sap_{max_number+1}_qq_{label}.csv'

#ch_data=extract_channel_data(data_h5,trial_number)
#voltage_channel=2
#current_channel=1
