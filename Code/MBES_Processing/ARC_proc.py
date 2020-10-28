#####################################################################################################
# Import modules
#####################################################################################################
from pandarallel import pandarallel
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")

pandarallel.initialize(nb_workers=num_cores, progress_bar=True)

#####################################################################################################
#User Input

# Todo Calculate and print stadrad deviations from binned pings
#####################################################################################################
# The script onyl work with the utput from the PROCESSING script.
from pandarallel import pandarallel
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
path = '/Volumes/Work/EMB205_stx'  # MUST end with a "/"."
filename = ''  # if empty, a rekursive search for all arc files from above folder will be done

bandwidth = 12000  #steht ach in den datenfiles, auslesen noch nicht eingebaut
only_stb_side = 'yes'
frequency = 400000
theta_across = ''  # can be left empty for nornit iwbms system
theta_along = ''   # for the Norbit, the frequency-specific array width is calculated


linear_input_values = 'yes'  #happens when field 7028 is read in mbsystem


adjust_spreading_absorption = 'yes'
"""
this is : VALUE_SET_IN_S7k - WANTED_VALUE
For example, if spreadding was set to 40 during data export, this is spreading
values given for two way travel
Values of 0 therefore have no effect on the data
"""
source_level_offset = 100  #adjust for source level
absorption = 0
spreading = 0


c = 1465  # m/s

ping_bin_number = 500  # how many pings to average
export_binned_results = 'yes'  # must be yes or no

###GSAB computation
# set bounds for A-F to speed up fitting. Roughly based on values reported by Lamarche et al (2011).
GSAB_calc = 'yes'
bounds_min = [-60, 1, -45, -50, 1]
bounds_max = [-5, 12, -10, -5, 20]
# Estimated values. Take from Lamarche 2011 for expected seafloor
x0 = np.array([-30, 5, -14, -20, 12])


###Plotting
plot_results = 'no'

#Todo: Make Region selection here?

#####################################################################################################
#Functions
#####################################################################################################
def get_array_width_norbit(freq, rad='no'):
    #Get Array width for norbit system
    along_track = 1.9 / freq * 400000  #angular resolution for norbit iwbmse
    across_track = 0.9 / freq * 400000
    if rad == 'yes':
        along_track = along_track * np.pi/180   #conver to rad
        across_track = across_track * np.pi/180   #conver to rad
    return along_track, across_track

def correct_spreading_offset_absorption(slant, spreading, absorption):
    TL = spreading * np.log10(slant) + 2 * slant * absorption * 0.001
    return TL


def gsab_func(theta, A, B, C, E, F):
    A = 10**(A/10)
    B = B * np.pi/180
    C = 10**(C/10)
    E = 10**(E/10)
    F = F * np.pi/180
    theta = theta * np.pi/180
    return 10*np.log10(A*np.exp(-theta**2/(2*B**2)) + C*np.cos(theta)**2 + E * np.exp(-theta**2 / (2*F**2)))

def smooth_ARC_Data(df, smooth_pings):
    return

def footprint_calc_BSWG(c, Pulse, Angle, theta_along, theta_across, slant, height):
    theta_across = np.deg2rad(theta_across)
    theta_along = np.deg2rad(theta_along)
    Angle = np.deg2rad(Angle)
    A1 = c * Pulse / (2 * np.sin(Angle)) * (theta_along) * \
        slant  # equation 23 from BSWG
    A2 = theta_along * theta_across * slant**2 * \
        1 / (np.cos(Angle))  # hellequin
    #A2 = 100
    #print(A1,A2)
    return np.min([A1, A2])

def footprint_calc_malik2015(slant, height, c, pulse, angle, theta_across, theta_along):
    theta_across = np.deg2rad(theta_across)
    theta_along = np.deg2rad(theta_along)
    angle = np.deg2rad(angle)
    #limit = (slant - height) - ((c * pulse) / 2) #hellequin2003
    A1 = theta_across * theta_along * slant**2 * 1 / (np.cos(angle))
    A2 = theta_along * slant * c * pulse * 1 / (2*np.sin(angle))
    return np.min([A1, A2])

def subtract_footprint(BS, Footprint):
    return BS - 10 * np.log10(Footprint)

def calculate_GSAB(index, df_bins, bounds_min, bounds_max, arc):
    arc = []
    ping_data = df_bins[df_bins['Pings_binned'] == index]  # extract one ping
    count = ping_data.Corr_BS.count()
    if count > 20:   # we want at least 20 reasonable values
        w, _ = opt.curve_fit(gsab_func, ping_data.Angle_binned, ping_data.Corr_BS,
                             p0=x0, maxfev=50000, bounds=(bounds_min, bounds_max))

        arc.append([w[0], w[1], w[2], w[3], w[4], index, count])

    cleanedArc = [x for x in arc if str(x) != 'None']
    return cleanedArc

def run_GSAB_test():
    #Test plot with parameters from Lamarche 2011 for gravel
    A = -9.1
    B = 4.9
    C = -17.9
    E = -17.5
    F = 16.2
    theta = np.arange(-80.1, 80.1, 1)
    bs = gsab_func(theta, A, B, C, E, F)
    plt.plot(theta, bs, 'b-', label='data')
    # Try to fit the function to the curv
    # bs: Backscatter values, theta: Winkel. Winkel in Grad
    # Estimated values. Take from Lamarche 2011 for expected seafloor
    x0 = np.array([-15, 7, -14, -20, 12])
    w, _ = opt.curve_fit(gsab_func, theta, bs, p0=x0)
    print("Estimated Parameters", w)
    print("Real Parameters: ",  A, B, C, E, F)

def calculate_Angle(df):
    df['Angle'] = 90 - (np.arctan(df.Height / df.Across) * 180 / np.pi) + 0.000001 # added to avoid zeros
    return


def calculate_slant(df):
    df['slant'] = np.sqrt(df.Across**2 + df.Height**2)


def convert_to_dB(df, offset=0):
    df['BS_dB'] = 20*np.log10(df.BS) - offset
    return


def calculate_effective_pulse_length(df, bandwidth):
    print("bandwidth set to ", bandwidth, "Hz")
    df['effectivePulse'] = 1 / bandwidth


def plot_results():
    print('Needs to be done')
    sys.exit(0)
    return

def binning(df, export="no", sanity_check='no', stbd_only='no', filepath=''):
    #Group data by Ping Counts and bin into 2° intervals, stacking over 30 pings
    bins_angle = np.arange(0, 70, 2)
    bins_ping = np.arange(df.PingCount.min()-ping_bin_number,
                          df.PingCount.max() + ping_bin_number, ping_bin_number)
    ## Binning Angles
    # Sort incidence angles indices into bins_angle
    bins_angle_idx = np.digitize(df.Angle, bins_angle) - 1
    # Make new dataframe columns
    data = []
    for i in range(bins_angle_idx.size):
        data.append(bins_angle[bins_angle_idx[i]])
    df['Angle_binned'] = data
    ## Binning Pings
    bins_ping_idx = np.digitize(df.PingCount, bins_ping)
    data = []
    for i in range(bins_ping_idx.size):
        data.append(bins_ping[bins_ping_idx[i]])
    df['Pings_binned'] = data
    # Make a new dataframe that includes the binned angles and pings
    df_bins = df.groupby(
        [df.Pings_binned, df.Angle_binned], as_index=False).mean()
    # Drop values with angles < 0 and > 60°
    if stbd_only == 'yes':
        df_bins.drop(df_bins[(df_bins.Angle_binned < -1) |
                             (df_bins.Angle_binned > 60)].index, inplace=True)
    # Drop unreasonable backscatter values for calibrated data
    if sanity_check == 'yes':
        df_bins.drop(df_bins[(df_bins.Corr_BS > 0) | (
            df_bins.Corr_BS < -100)].index, inplace=True)
    # Drop lines with NaN's  and infs
    df_bins.drop(df_bins[df_bins.isin(
        [np.nan, np.inf, -np.inf]).any(1)].index, inplace=True)
    # Drop uneeded columns
    del df_bins['Angle']
    del df_bins['PingCount']
    del df_bins['Height']
    del df_bins['Across']
    del df_bins['Pulse']
    del df_bins['slant']
    del df_bins['effectivePulse']

    if export == 'yes':
            print("Saving to: ", filepath + '_binned')
            df_bins.to_csv(filepath + '._binned')
    return df_bins


def getfiles(ID='', PFAD='.', rekursive='no'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    import os
    import glob2
    files = []
    if rekursive == 'no':
        for file in os.listdir(PFAD):
            if file.endswith(ID):
                files.append(str(file))
    if rekursive == 'yes':
        files = glob2.glob(PFAD + '/**/*' + ID)
    return files

#####################################################################################################
#Get Files
#####################################################################################################
if filename == '':
    print("Searching rekursively for .arc files in: ", path)
    files = getfiles(ID=".arc", PFAD=path, rekursive='yes')
else:
    files = [path + filename]

print("Working on: ", len(files), " files.")

#####################################################################################################
#Processing
#####################################################################################################
if theta_across == '':
    theta_across, theta_along = get_array_width_norbit(frequency)



error_files = []
for filepath in files:
    # Processing raw data
    # Read Files
    print("Working on: ", filepath)
    df = pd.read_csv(filepath, header=0, names=[
                     'X', 'Y', 'PingCount', 'Height', 'Across',  'Beam', 'Pulse', 'BS'], sep='\t')  # lazy script, do not change names

    if only_stb_side == 'yes':
        df = df[df.Across > 0]  # read only starboard side
    print(df.head())

    #Calculate flat bottom grazing angle
    calculate_Angle(df)

    #Calculate Slant Range
    calculate_slant(df)

    #Convert to dB
    if linear_input_values == 'yes':
        convert_to_dB(df) #
    else:
        df['BS_dB'] = df.BS

    #calculate effective pulse length
    calculate_effective_pulse_length(df, bandwidth)

    # Do footprint correction
    print("Calculate Footprint values.")
    try:
        df['Footprint'] = df.parallel_apply(lambda row: footprint_calc_malik2015(
        row['slant'], row['Height'], c, row['effectivePulse'], row['Angle'], theta_across, theta_along), axis=1)
    except:
        print("ERROR in file:", filepath)
        error_files.append([filepath, "Footprint"])


    print("Adjusting absorption and spreading gain")
    if adjust_spreading_absorption == 'yes':
        try:
            df['TL'] = df.parallel_apply(lambda row: correct_spreading_offset_absorption(row['slant'], spreading, absorption), axis=1)
        except:
            print("ERROR in file:", filepath)
            error_files.append([filepath, "SpreadAbsor"])

    # Correcting backscatter
    print("Correcting Backscatter values.")
    try:
        if adjust_spreading_absorption == 'yes':
            df['Corr_BS'] = df.parallel_apply(
            lambda row: subtract_footprint(row.BS_dB, row.Footprint), axis=1) + df['TL'] - source_level_offset
        else:
            df['Corr_BS'] = df.parallel_apply(lambda row: subtract_footprint(row.BS_dB, row.Footprint), axis=1)
    except:
        print("ERROR in file:", filepath)
        error_files.append([filepath, "Correction of BS"])

    #bin_pings
    print("Binning.")
    try:
        df_bins = binning(df, export=export_binned_results,
                      sanity_check='no', stbd_only='no', filepath=filepath)
    except:
        print("ERROR in file:", filepath)
        error_files.append([filepath, "Binning"])

    # GSAB Computation
    if GSAB_calc == 'yes':
        #try:
        print("GSAB calculation")
        ping_numbers = df_bins['Pings_binned'].unique()
        gsab = []
        gsab = Parallel(n_jobs=num_cores)(delayed(calculate_GSAB)(index, df_bins, bounds_min, bounds_max, gsab)
                                        for index in tqdm(ping_numbers))
        gsab = [item for sublist in gsab for item in sublist]  # flatten list
        gsab_df = pd.DataFrame(gsab, columns = ["A", "B", "C", "E", "F", "Ping", "Count"])
        if export_binned_results == 'yes':
            print("Saving to: ", filepath + '_binned.gsab')
            gsab_df.to_csv(filepath + '._binned.gsab')

    # plot the results
    if plot_results == 'yes':
        theta = np.arange(-60.1, 60.1, 2)
        for i in range(len(arc)):
            w = arc[i]
            bs = gsab_func(theta, w[0], w[1], w[2], w[3], w[4])
            plt.plot(theta, bs, 'b-', label='data')
            ping_data = df_bins[df_bins['Pings_binned'] == arc[i][5]]
            plt.plot(ping_data.Angle_binned, ping_data.Corr_BS,
                     'b-', label='data', color='red')
            #print(w)

print("The following files hat errors and were not processed: ")
for error in error_files:
    print(error)
