# -*- coding: utf-8 -*-
from __future__ import print_function    #python 3 print stements
"""
Created on Thu Sep 29 14:01:24 2016

@author: feldens
"""
##############################################################
# Multibeam processing script
# The calibrated s7k files need a three step approach.


##!!!! The filename of the profile lines MUST NOT END with a p !!!!
##!!! No spaces are allowed in the profile names or paths !!!

##!! A cmmon problem is that mbprocess fails becuase no .par files are present

#todo: scatter grid for dataset auf def umstellen
#todo implement high pass filter to improve stones
#This script is intended to only work on survey level datafiles.

##IOW Archive Structure
#IOWDROPBOX/Survey_Data/datalist.mb-1  -> all s7k files
#                       /year/datalist.mb-1 -> all s7k files of one year
#                            /survey/datalist.mb-1 -> all files of a particular survey
##############################################################
# LEVEL 1: IMPORT AND BASIC CORRECTIONS
##############################################################
# Control which levels are worked on
LEVEL1 = 'yes'
LEVEL2 = 'yes'
LEVEL3 = 'yes'

remove_lock_files = 'yes' #Yes tries to remove lockfiles for all files linked in the datalists via mblist
PFAD = '/Volumes/Work/KH201910/test/'   # end with /
rekursive_directory_search = 'no'
PREPROCESS = 'yes'
FORMAT = 89  # .ALL UND .S7K FILES  work
file_end = '.s7k'
SS_FORMAT = 'C'  # scbw  s snippet c calib. snippet b widebeambackscatter w calibwidebeambackscatter "auto" - no option

AREA = '12.10/12.1143/54.1844/54.18997'  # WESN. printed in datalist.info at the end of level1
GENERATE_DATALIST = 'yes'
AUTO_CLEAN_BATHY = 'yes'
ATTITUDE_LAG = ''
SELECT_SVP = ''          # mbsvpselect crashing at the moment why?? -> mbsystem bug? has to be done manually atm
SVP = ''             #this is the manual file included in all par files
CORRECT_HPR=''
ROLL_CORR = 0.00
PITCH_CORR = 0.00
CORRECT_TIDE = ''        #no: removes entries from par fileand reprocesses
TIDEFILE = ''  #Tidemode set to 2
CORRECT_DRAFT = 'yes'
DRAFT_CORR = 0.4

EXPORT_NAV = 'no'           # Export Navigation information and stores under profile file name

##############################################################
# LEVEL 2: Correct Backscatter Data
##############################################################
EXPORT_ARC_CURVES = 'no'
PROCESS_SCATTER = 'yes'  # yes is running mbbackangle
CONSIDER_SEAFLOOR_SLOPE = ''
AVERAGE_ANGLE_CORR = 'yes' # backangle correction file specific (no) or average (yes) for complete datesaet


SSS_ACROSS_CUT = 'yes'
SSS_ACROSS_CUT_MIN = 0
SSS_ACROSS_CUT_MAX = 11

SSS_CORRECTIONS = 'yes' #applies all of the follwoing settings
SSSWATHWIDTH = 50  #that is supposed to e an agnle
SSINTERPOLATE = 10
##############################################################
# LEVEL 3: Make grid data
##############################################################

SCATTER_FILTER = 'low'       #low or high - high not implemented atm works on p-files
INTERPOLATION = '-C3/1'      #up to three cells are interpolated
## Grids
WORK_ON_PER_FILE_BASIS = 'no'  # Make grids i.e. for each file individually

# Work for both on a per-survey and per file setting
GENERATE_BATHY_GRIDS = 'yes'
GENERATE_SCATTER_GRIDS = 'yes'
SCATTER_WITH_FILTER ='yes'   #Export filtered grids
EXPORT_XYI_FROM_GRID = 'no'
"""
Idea: export first pings with mblist with depth and ship speed and make an educated guess on grid size for each file
"""
BATHY_RES = '-E1/1'
SCATTER_RES = '-E0.5/0.5'

# convert Grids. ot implemented atm the results files are converted for speed reasons
UTM_Convert = 'no'
ZONE = '-JU'   # in syntax for mbsystem not implemented in mbsystem atm because of using geographical coordinates throughout

#Only for work on a per-file bases

EXPORT_BEAM_ANGLE = 'no'    #
EXPORT_XYI = 'no'           #
EXPORT_XYZ = 'no'
KFAKTOR = 50               # jeder wievielte Schuss soll exportiert werden mit EXPORT_XYI

FORCE_MBPROCESS = ''   #Force a mbprceoss run; only needed for manual changes
number_of_exceptions = 0


##############################################################
# Preparation
##############################################################
# Load modules
import multiprocessing
from joblib import Parallel, delayed
import sys
sys.path.append('./functions')
import os
import FUNCTIONS_GEOGRAPHIC as fg
from tqdm import tqdm

#Multiprocessing Setup
num_cores = multiprocessing.cpu_count() - 2
if num_cores < 2:
    print("Go buy a new PC")
    sys.exit(0)
print("Using ", num_cores, " cores. ")

#Pfade
os.chdir(PFAD)


if remove_lock_files == 'yes':
    print("Trying to remove old lock files:")
    command = "mbdatalist -F-1 -Idatalist.mb-1 -Y"  # remove prior lock files
    os.system(command)
    command = "mbdatalist -F-1 -Idatalistp.mb-1 -Y"  # remove prior lock files
    os.system(command)
    command = "mbdatalist -F-1 -Idatalistpp.mb-1 -Y"
    os.system(command)


##############################################################
# Sanity checks
##############################################################
if SS_FORMAT not in ["S", "C", "auto", "W", "B"]:
    print("Select correct SS_Format identifier - refer to man mbpreprocess")
    sys.exit(0)


##############################################################
# Functions
##############################################################

def test_for_processed_mbfiles(files):
    # Test if files already processed -> Test if inf files exists
    import pandas
    files_to_process = []
    for i in range(len(files)):
        test_file = files[i] + '.inf'
        test_case = os.path.isfile(test_file)
        if test_case == False:
            print('Bearbeite Datei: ', files[i])
            files_to_process.append(files[i])
            df = pandas.DataFrame(files_to_process)
            df.to_csv('last_import', header = False, index = False)
        else:
            print('Ignoriere Datei: ', files[i])
    return files_to_process

def export_scatter_file(SCATTERFILE):
    """Export Scatter files"""
    # Intensity Daten einlesen: Tabelle Intensity
    # X Y Intensity
    import pandas
    import sys
    DELIMITER = '\t'
    XPos = []
    YPos = []
    Intensity = []
    Depth = []
    Acrosstrack_Dist = []
    BeamAngle = []
    #FileName = []

    try:
        f = open(SCATTERFILE,'r')
    except:
        print("Fehler in import_scatter_file")
        sys.exit(1)

    for line in f:
        #jede Zeile an dem delimiter auftrennen, leerzeichen löschen
        temp=[x.strip() for x in line.split(DELIMITER)]
        XPos.append(float(temp[0]))
        YPos.append(float(temp[1]))
        Intensity.append(float(temp[2]))
        Depth.append(float(temp[3]))
        Acrosstrack_Dist.append(float(temp[4]))
        BeamAngle.append(float(temp[5]))

        #FileName.append(str(SCATTERFILE))

    # Combine into Pandas Data Frame
    dfdata = pandas.DataFrame({
                           'X' : XPos,
                           'Y' : YPos,
                           'Intensity' : Intensity,
                           'Depth_below_trans' : Depth,
                           'Acrosstrack_Dist' : Acrosstrack_Dist,
                           'BeamAngle' : BeamAngle,
                           })
    f.close()
    return dfdata

def export_grid_file(SCATTERFILE, TYPE):
    """Export grid files"""
    # Intensity Daten einlesen: Tabelle Intensity
    # X Y Intensity - das sollen doch grids werden.....
    import pandas
    import sys
    DELIMITER = '\t'
    XPos = []
    YPos = []
    Z = []

    try:
        f = open(SCATTERFILE,'r')
    except:
        print("Fehler in import_scatter_file")
        sys.exit(1)

    for line in f:
        #jede Zeile an dem delimiter auftrennen, leerzeichen löschen
        temp=[x.strip() for x in line.split(DELIMITER)]
        XPos.append(float(temp[0]))
        YPos.append(float(temp[1]))
        Z.append(float(temp[2]))

        #FileName.append(str(SCATTERFILE))

    # Combine into Pandas Data Frame
    dfdata = pandas.DataFrame({
                           'X' : XPos,
                           'Y' : YPos,
                           'Intensity' : Z,
                           })
    f.close()
    return dfdata

def process_scatter(FORMAT, mbfile, CONSIDER_SEAFLOOR_SLOPE):
    if CONSIDER_SEAFLOOR_SLOPE == 'yes':
        command = 'mbbackangle -I' + mbfile + ' -F' + \
            str(FORMAT) + ' -P2000 -G2/70/70/50 -N81/80 -R44  -Q '
    else:
        command = 'mbbackangle -I' + mbfile + \
            ' -F' + str(FORMAT) + ' -G2/70/70/50 -P2000 -R45 -N81/80 '
        os.system(command)
    command = 'mbset -I' + mbfile + ' -F' + str(FORMAT) + ' PSSCORRTYPE: 1'
    os.system(command)
    return


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


def choose_processed_unprocessed_file(file_list):
    unprocessed_files = []
    processed_files = []
    processed_processed_files = []
    temp = []
    for i in file_list:
        if "p.mb" not in i:
            unprocessed_files.append(i)
        if "p.mb" in i:
            if "pp.mb" not in i:
                processed_files.append(i)
        if "pp.mb" in i:
            processed_processed_files.append(i)
    return unprocessed_files, processed_files, processed_processed_files


def mbprocess(FORMAT, mbfile):
    command = 'mbprocess  -F' + str(FORMAT) + ' -I' + mbfile
    os.system(command)
    return


def mbfilter(FORMAT, mbfilep):
    command = 'mbfilter -S3/7/7/2 -F' + str(FORMAT) + ' -I' + mbfilep
    os.system(command)
    return


def mbpreprocess(FORMAT, mbfile, SS_FORMAT):
    if SS_FORMAT == 'auto':
        command = command = 'mbpreprocess --format=' + \
                str(FORMAT) + ' --input=' + mbfile
        os.system(command)
    else:
        command = command = 'mbpreprocess --format=' + \
            str(FORMAT) + '  --multibeam-sidescan-source=' + \
            str(SS_FORMAT) + ' --input=' + mbfile
        os.system(command)
    return

def execute_mbcommand(mbcommand):
    print("Execute: ", mbcommand)
    os.system(mbcommand)
    return

def autoclean(mbfile, FORMAT):
        command = 'mbclean -F' + str(FORMAT) + ' -I' + mbfile + ' -R3 -X5 -Q1qq -Z'
        os.system(command)
        return

def bathy_grid_file(file, BATHY_RES, FORMAT):
    command = 'mbm_grid ' + BATHY_RES + ' -A2 -C2  -G3  ' + ' -F' + \
                        str(FORMAT) + ' -I' + file + \
                            ' ' + '-O' + file + '_bath'
    os.system(command)
    command = './' + file + '_bath_mbgrid.cmd '
    os.system(command)
    return

def scatter_grid_file(file, SCATTER_WITH_FILTER, SCATTER_RES, INTERPOLATION, FORMAT):
        name_add = ''
        print('Generate Scatter Grid')
        datatype = '-A4'
        if SCATTER_WITH_FILTER =='yes':
            print('Plotting grids of low-pass-filtererd SSS data')
            datatype = '-A4F'
            name_add = '_filtered'
        # Generate first cut sidescan mosaic and plot
        command = 'mbm_grid ' + datatype + ' -M -Y1 -P0 -G3  ' + ' ' +  SCATTER_RES + ' ' + INTERPOLATION + ' -F' + str(FORMAT) + ' -I' +file + ' ' + '-O' + file +'_sss' + name_add
        os.system(command)
        if SCATTER_WITH_FILTER =='no':
            command = './' + file + '_sss_mbmosaic.cmd '
            os.system(command)
        if SCATTER_WITH_FILTER =='yes':
            command = './' + file + '_sss_filtered_mbmosaic.cmd '
            os.system(command)
        return

def export_arc(mbfile, FORMAT):
        command = "mblist -F" + str(FORMAT) + " -I" + str(mbfile) + " -NA -OXYNCd#.lb > " + mbfile + ".arc"
        os.system(command)

###############################################################################
##############################################################
# LEVEL 1: IMPORT AND BASIC CORRECTIONS
##############################################################
#Preprocessing of data
DATA_TO_PROC = 'no'
print("Level 1 processing is set to:", LEVEL1)
if LEVEL1 == 'yes':
    if PREPROCESS == 'yes':
        if rekursive_directory_search == 'no':
            print('Preprocessing')
            #Hysweeppreprocess muss jede Datei einzeln vorgesetzt bekommen sonst wirds nichts
            s7kfiles = getfiles(file_end)
        if rekursive_directory_search == 'yes':
            s7kfiles = getfiles(file_end, '.', 'yes')

        Parallel(n_jobs=num_cores)(delayed(mbpreprocess)(FORMAT, mbfile, SS_FORMAT)
                                    for mbfile in tqdm(s7kfiles))
        GENERATE_DATALIST = 'yes'

    if GENERATE_DATALIST =='yes':
        #
        if rekursive_directory_search == 'no':
            print('Generate datalists non-recursive')
            command = '/bin/ls -1 *[^p].mb' + str(FORMAT) + ' | awk \'{print $1" ' + str(FORMAT) + '"}\' > datalist.mb-1'
            print(command)
            os.system(command)

        if rekursive_directory_search == 'yes':
            print('Generate datalists recursive')
            command = 'find . -type f -name "*.mb' + str(FORMAT) + '" | awk \'{print $1" ' + str(FORMAT) + '"}\' > datalist.mb-1'
            os.system(command)

        # Generate a datalist referencing processed mb201 files named datalistp.mb-1
        command = 'mbdatalist -Z -I datalist.mb-1'
        os.system(command)

    if AUTO_CLEAN_BATHY == 'yes':
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files, _, _ = choose_processed_unprocessed_file(files)
        print("Autoclean")
        Parallel(n_jobs=num_cores)(delayed(autoclean)(mbfile, FORMAT)
                                for mbfile in tqdm(files))
        DATA_TO_PROC = 'yes'

    if ATTITUDE_LAG == 'yes':
        #---------------------------------------------------------------------------------
        # check the attitude time lag
        #---------------------------------------------------------------------------------
        print('Checking for attitude lag')
        # Look for correlation between raw roll and apparent seafloor acrosstrack slope
        command = 'mbrolltimelag -I datalist.mb-1 -T501/-0.5/0.5 -K18 -N50 -O ZInitial -V'
        os.system(command)
        command = './ZInitial_timelaghist.txt.cmd'
        os.system(command)
        command = './ZInitial_timelagmodel.txt.cmd'
        os.system(command)
        command = './ZInitial_xcorr.txt.cmd'
        os.system(command)
        command = 'convert -density 300 ZInitial_timelaghist.txt.ps -trim -quality 92 ZInitial_timelaghist.txt.jpg'
        os.system(command)
        command = 'convert -density 300 ZInitial_timelagmodel.txt.ps -trim -quality 92 ZInitial_timelagmodel.txt.jpg'
        os.system(command)
        command = 'convert -density 300 ZInitial_xcorr.txt.ps -trim -quality 92 ZInitial_xcorr.txt.jpg'
        os.system(command)

    if EXPORT_NAV == 'yes':
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files, _, _ = choose_processed_unprocessed_file(files)
        print("Exporting Navigation")
        for file in tqdm(files):
            command = 'mbnavlist -F' +str(FORMAT) + ' -I' + file + ' -D25 -G, > ' + file + '.navtemp'
            os.system(command)
            command = 'sed  "s/$/,' + str(file) + '/" ' + file + '.navtemp > ' +  file + '.nav'
            os.system(str(command))
            command = 'rm ' + file + '.navtemp'
            os.system(command)

    if CORRECT_HPR == 'yes':
        print('Correct HPR')
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files , _, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PROLLBIASMODE:1 -PROLLBIAS:' + str(ROLL_CORR) + ' -PPITCHBIASMODE:1 -PPITCHBIAS:' + str(PITCH_CORR)
            os.system(command)
        DATA_TO_PROC = 'yes'

    if CORRECT_DRAFT == 'yes':
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files , _, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PDRAFTMODE:4 -PDRAFT:' + str(DRAFT_CORR)
            os.system(command)
        DATA_TO_PROC = 'yes'

    if SELECT_SVP == 'yes':
        print('Select SVP')
        #command = 'mbsvpselect -P1 -Ssvplist.mb-1'   #P1: selection to nearest in time
        #Workaround weil das mit den svpprofilen bei format 201 nicht zu gehen scheint
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files , _, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PSVPMODE:1 -PSVPFILE:' + SVP
            os.system(command)
        DATA_TO_PROC = 'yes'

    if CORRECT_TIDE == 'yes':
        print('Correcting Tide')
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files , _, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PTIDEMODE:1 -PTIDEFILE:' + str(TIDEFILE) + ' -PTIDEFORMAT:2'
            os.system(command)
        DATA_TO_PROC = 'yes'

    if CORRECT_TIDE == 'no':
        print('Not correcting Tide/removing tidal corrections')
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files , _, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PTIDEMODE:0 -PTIDEFILE:'
            os.system(command)
        DATA_TO_PROC = 'yes'

    # Process if needed
    if FORCE_MBPROCESS == 'yes':
        DATA_TO_PROC = 'yes'

    if DATA_TO_PROC == 'yes':
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        files, _ , _= choose_processed_unprocessed_file(files)
        Parallel(n_jobs=num_cores)(delayed(mbprocess)(FORMAT, mbfile)
                                for mbfile in tqdm(files))

        print('Generate datalists for processed files')
        if rekursive_directory_search == 'no':
            command = '/bin/ls -1 *[^p]p.mb' + str(FORMAT) + ' | grep -v "p.' + str(
                FORMAT) + '" | awk \'{print $1" ' + str(FORMAT) + '"}\' > datalistp.mb-1'
            os.system(command)
        if rekursive_directory_search == 'yes':
            command = 'find . -type f -name "*[^p]p.mb' + \
                        str(FORMAT) + '" | awk \'{print $1" ' + \
                            str(FORMAT) + '"}\' > datalistp.mb-1'
            os.system(command)

        print("Writing Basic information for datalist in datalist.info")
        command = "mbinfo -F-1 -Idatalist.mb-1 > datalist.info"
        os.system(command)
    DATA_TO_PROC = 'no'

##############################################################
# LEVEL 2: Backscatter information
##############################################################
print("Level 2 processing is set to:", LEVEL2)
if LEVEL2=='yes':

    if SSS_ACROSS_CUT == 'yes':
        print('Cutting SideScan Across Track')
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        _, files, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + '-PDATACUT:2/2 -PSSCUTDISTANCE:' + \
                str(SSS_ACROSS_CUT_MIN) + '/' + str(SSS_ACROSS_CUT_MAX)
            os.system(command)
        DATA_TO_PROC = 'yes'

    if SSS_CORRECTIONS == 'yes':
        print('Applying set SSS corrections')
        ID = 'mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        _, files, _ = choose_processed_unprocessed_file(files)
        for mbfile in files:
            command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' PSSCORRMODE:1 -PSSSWATHWIDTH:' + \
            str(SSSWATHWIDTH)  + ' -PSSINTERPOLATE:' + str(SSINTERPOLATE)
            os.system(command)
        DATA_TO_PROC = 'yes'

    if EXPORT_ARC_CURVES == 'yes':
        print('Exporting ARC data')
        print("Angles cannot be correctly exported from mbsystem with the -NA side scan option as of 20.10.2020")
        print("Flat grazing angles need to be calculated in postprocessing until that time")
        ID = '.mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        _, files , _= choose_processed_unprocessed_file(files)
        # mblist command to export arc
        Parallel(n_jobs=num_cores)(delayed(export_arc)(mbfile, FORMAT)
                                for mbfile in tqdm(files))


    if PROCESS_SCATTER == 'yes':
        print('Process Scatter')
        if AVERAGE_ANGLE_CORR == 'yes':
            print('Setting BS correction to a survey level')
            process_scatter("-1", "datalistp.mb-1", CONSIDER_SEAFLOOR_SLOPE)
            ID = 'mb' + str(FORMAT)
            if rekursive_directory_search == 'no':
                files = getfiles(ID)
            if rekursive_directory_search == 'yes':
                files = getfiles(ID, '.', 'yes')
            _ , files, _ = choose_processed_unprocessed_file(files)
            for mbfile in files:
                print("updating mbset to survey level")
                command = 'mbset -F' + str(FORMAT) + ' -I' + mbfile + ' -PSSCORRFILE:datalistp.mb-1_tot.sga'
                os.system(command)
            DATA_TO_PROC = 'yes'
        else:
            print('Setting BS correction to a file level')
            ID = 'mb' + str(FORMAT)
            if rekursive_directory_search == 'no':
                files = getfiles(ID)
            if rekursive_directory_search == 'yes':
                files = getfiles(ID, '.', 'yes')
            _, files , _= choose_processed_unprocessed_file(files)
            Parallel(n_jobs=num_cores)(delayed(process_scatter)(FORMAT, mbfile, CONSIDER_SEAFLOOR_SLOPE)
                                    for mbfile in tqdm(files))
            ID = '.mb' + str(FORMAT)
            if rekursive_directory_search == 'no':
                files = getfiles(ID)
            if rekursive_directory_search == 'yes':
                files = getfiles(ID, '.', 'yes')
            _, files , _= choose_processed_unprocessed_file(files)
            for file in files:
                sga_name = file  + '.sga'
                command = 'mbset -I' + file + ' -PSSCORRFILE:' + sga_name
                os.system(command)
        # Process the data
        DATA_TO_PROC = 'yes'

    # Process if needed
    #DATA_TO_PROC = 'no'
    if FORCE_MBPROCESS == 'yes':
        DATA_TO_PROC = 'yes'

    if DATA_TO_PROC == 'yes':
        ID = 'p.mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')

        _ , files, _ = choose_processed_unprocessed_file(files)
        Parallel(n_jobs=num_cores)(delayed(mbprocess)(FORMAT, mbfile)
                                for mbfile in tqdm(files))
        #make pp-Datalist
        if rekursive_directory_search == 'no':
            print('Generate datalists for processed-processed files')
            command = '/bin/ls -1 *pp.mb' + str(FORMAT) + ' | grep -v "pp.' + str(
                FORMAT) + '" | awk \'{print $1" ' + str(FORMAT) + '"}\' > datalistpp.mb-1'
            os.system(command)

        if rekursive_directory_search == 'yes':
            command = 'find . -type f -name " *pp.mb' + \
                str(FORMAT) + '" | awk \'{print $1" ' + \
                str(FORMAT) + '"}\' > datalistpp.mb-1'
            os.system(command)

    DATA_TO_PROC = 'no'


##############################################################
# LEVEL 3: Grids
##############################################################
print("Level 3 processing is set to:", LEVEL3)
if LEVEL3 =='yes':
    if SCATTER_FILTER == 'low':
        ID = 'p.mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        _, _, files = choose_processed_unprocessed_file(files)
        print("LOW PASS FILTERING:")
        Parallel(n_jobs=num_cores)(delayed(mbfilter)(FORMAT, mbfile)
                                for mbfile in tqdm(files))

    if SCATTER_FILTER == 'high':
        print('Highpass not yet implemented')
        SCATTER_WITH_FILTER ='no'


    if WORK_ON_PER_FILE_BASIS =='no':
        print('working on a dataset basis')
        if GENERATE_BATHY_GRIDS == 'yes':
            try:
                print('Generate Bathy Grid')
                command = 'mbm_grid ' + BATHY_RES + ' -R' + AREA + ' -A2 -C2  -G3 ' + ' -F-1' + ' -Idatalistpp.mb-1' + ' ' + '-Obath_grid'
                os.system('ls')
                os.system(command)
                command = './' + 'bath_grid_mbgrid.cmd '
                os.system(command)
            except:
                print('Problem bei der Generierung der bathy-grids: ', file)

        if GENERATE_SCATTER_GRIDS =='yes':
            name_add = ''
            print('Generate Scatter Grid')
            datatype = '-A4'
            if SCATTER_WITH_FILTER =='yes':
                print('Plotting grids of low-pass-filtererd SSS data')
                datatype = '-A4F'
                name_add = '_filtered'
            # Generate sidescan mosaic and plot
            command = 'mbm_grid ' + datatype + ' -M -P0 -G3  ' + ' ' +  SCATTER_RES + ' -R' + AREA + ' ' + INTERPOLATION + ' -F-1' + ' -Idatalistpp.mb-1' + ' ' + '-Osss_grid' + name_add
            os.system(command)
            if SCATTER_WITH_FILTER =='no':
                command = './sss_grid_mbmosaic.cmd '
                os.system(command)
            if SCATTER_WITH_FILTER =='yes':
                command = './sss_grid_filtered_mbmosaic.cmd '
                os.system(command)

        if EXPORT_XYI_FROM_GRID == 'yes':
            try:
                print('Export scatter from grid for to xyi')
                command = 'gmt grd2xyz ' +  'sss_grid.grid  >  sss_grid.xyi'
                os.system(command)
            except:
                print('Problem with xyi export of grid')
                number_of_exceptions = number_of_exceptions + 1
            try:
                print('Export filtered scatter from grid for grid')
                command = 'gmt grd2xyz ' +  str(file) +'sss_grid_filtered.grd  > sss_grid_filtered.xyi'
                os.system(command)
            except:
                print('No filtered gridfile for export to xyi')


    if WORK_ON_PER_FILE_BASIS =='yes':
        print('working on a per-file-basis')
        ID = 'p.mb' + str(FORMAT)
        if rekursive_directory_search == 'no':
            files = getfiles(ID)
        if rekursive_directory_search == 'yes':
            files = getfiles(ID, '.', 'yes')
        _, _, files = choose_processed_unprocessed_file(files)
        print('Folgende Dateien werden bearbeitet:', files)
        if GENERATE_BATHY_GRIDS == 'yes':
            Parallel(n_jobs=num_cores)(delayed(bathy_grid_file)(mbfile, BATHY_RES, FORMAT)
                                for mbfile in tqdm(files))


        if GENERATE_SCATTER_GRIDS =='yes':
            print("Working on files: ", files)
            Parallel(n_jobs=num_cores)(delayed(scatter_grid_file)(mbfile, SCATTER_WITH_FILTER, SCATTER_RES, INTERPOLATION, FORMAT)
                                    for mbfile in tqdm(files))


        if EXPORT_BEAM_ANGLE == 'yes':
            for file in files:
                print('Export Beam Angles for ', file)
                command = 'mblist -F' + str(FORMAT) + ' -I ' + file + ' -OXYg -MA -K' + str(KFAKTOR) + '>  ' + file + '_ba.xya'
                os.system(command)


        if EXPORT_XYI_FROM_GRID == 'yes':
            if SCATTER_WITH_FILTER =='yes':
                try:
                    print('Export filtered scatter from grid for ', file)
                    Parallel(n_jobs=num_cores)(delayed(execute_mbcommand)('gmt grd2xyz ' +  str(file) +'_sss_filtered.grd  > ' + str(file) + '_sss_grid_filtered.xyi') for mbfile in files)
                except:
                    print('No filtered gridfile')
            else:
                try:
                    print('Export scatter from grid for ', file)
                    Parallel(n_jobs=num_cores)(delayed(execute_mbcommand)('gmt grd2xyz ' +  str(file) +'_sss.grd  > ' + str(file) + '_sss_grid.xyi') for mbfile in files)
                except:
                    print('Problem with xyi export of file', file)
                    number_of_exceptions = number_of_exceptions + 1

        if EXPORT_XYI == 'yes':
            print("Export XYI")
            Parallel(n_jobs=num_cores)(delayed(execute_mbcommand)('mblist -F' + str(FORMAT) + ' -I ' + mbfile + ' -O^X^Yb -JU -NA -K' + str(KFAKTOR) + '  >  ' + mbfile + '_sss.xyi') for mbfile in files)

        if EXPORT_XYZ == 'yes':
            print ("Export XYZ")
            Parallel(n_jobs=num_cores)(delayed(execute_mbcommand)('mblist -F' + str(FORMAT) + ' -I ' + mbfile +
                                                                    ' -O^X^YZ -JU -MA -K ' + str(KFAKTOR) + '  >  ' + mbfile + '_bath.xyz') for mbfile in files)
        print('All done')
