# RTXI-data-analysis-toolkit
A collection of tools for processing RTXI output data

#### Implementation notes:
- **load_h5.py** includes the functions for processing data
- **process_rtxi.ipynb** includes step-by-step instructions for processing and visualizing the data

#### Current Features:
- Load h5 file into python
- Plot Voltage and Current vs time and print tags for the given trial
- Automatically finds action potentials in the data
- Finds the following features for every action potential:
    - Start Time
    - End Time
    - Spontaneous vs. Non-spontaneous
    - Cycle Length
    - Diastolic Intervals
    - Duration at 30, 40, 70, 80, and 90% repolarization
    - Amplitude
    - Maximum Diastolic Potential
    - Shape Factor
    - Maximum Upstroke Velocity
- Plot this information across all action potentials

#### Where to start:

1. Install [Jupyter Notebook](https://jupyter.org/install) on your computer
2. From the terminal, clone this repo onto your computer:
```
$ git clone https://github.com/Christini-Lab/RTXI-data-analysis-toolkit.git
```
3. If you don't have an h5 file that you want to process, you can get an example file from the [Christini Box account](https://cornell.app.box.com/folder/78710760726). Copy the h5 file into the `./data/` folder in this directory.

4. From the terminal, start a jupyter notebook session:
```
$ jupyter notebook
```

5. The `jupyter notebook` command should open a new window in your web browser. Open the `rtxi_processing interface` file and follow the instructions there.
