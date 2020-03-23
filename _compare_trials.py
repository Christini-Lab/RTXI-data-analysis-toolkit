from load_h5 import load_recorded_data
import matplotlib.pyplot as plt


filename = 'attempt_2_071519.h5'
trial_number = 2
does_plot = True
no_tag = True
recorded_data = load_recorded_data(filename, trial_number, does_plot, no_tag)


does_plot = False 
no_tag = True

file_new = 'Test_7.h5'
trial_number = 6
recorded_new = load_recorded_data(file_new, trial_number, does_plot, no_tag)

file_old = 'attempt_2_071519.h5'
trial_number = 4
recorded_old = load_recorded_data(file_old, trial_number, does_plot, no_tag)



fig, axes = plt.subplots(2, 1)
fig.tight_layout()

axes[0].title.set_text('Voltage (V)')

axes[0].plot(recorded_old['Time (s)'], recorded_old['Voltage (V)'], label='July Patch')
axes[0].plot(recorded_new['Time (s)'], recorded_new['Voltage (V)'], label='October Patch')
axes[0].axhline(0, color='k')

axes[1].title.set_text('Current (A)')
axes[1].plot(recorded_old['Time (s)'], recorded_old.Current, label='July Patch')
axes[1].plot(recorded_new['Time (s)'], recorded_new.Current, label='October Patch')
axes[1].axhline(0, color='k')




plt.legend()
plt.show()

plt.plot(recorded_old['Time (s)'], recorded_old.Current, label='July Patch')
plt.plot(recorded_new['Time (s)'], recorded_new.Current, label='October Patch')
plt.legend()
plt.show()
