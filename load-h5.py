# 1. Import the h5py library
import h5py
import pdb
# import numpy as np
# from matplotlib import pyplot as plt
pdb.set_trace()
# 1. Import the h5 file
filename = 'data/test_rtxitoolkit_050319.h5'

# 2. Read in the file
f = h5py.File(filename, 'r')

# 3. The following function is used to get the keys of an h5 object
def get_keys(f):
    return [key for key in f.keys()]

# 4. Get the keys for the h5 file
print(get_keys(f))

# 5. Save the h5 objects in each tag
tags = f['Tags']
trial_1 = f['Trial1']
trial_2 = f['Trial2']

# 6. Print the keys for the tags and each trials
print(get_keys(tags))
print(get_keys(trial_1))
print(get_keys(trial_2))

# 7. Get the Sync data from trial 1
trial_1_data_synch = trial_1['Synchronous Data']
print(trial_1_data_synch)

# 8. Get the tags from the sync data
synch_members = get_keys(trial_1_data_synch)
print(synch_members)

# 9. Get the dataset saved to one of the asynch variables
dset = trial_1_data_synch['Channel Data']
print(dset)

# 10. print the values in the dataset
print(dset.value)


# d_to_plot = np.zeros(len(dset.value))
# i = 0

# while i < 10:
#   d_to_plot[i] = dset.value[i][0]

# plt.plt(d_to_plot)
# pdb.set_trace()







