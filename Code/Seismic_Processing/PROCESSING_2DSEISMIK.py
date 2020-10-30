# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:14:15 2016

@author: feldens

ImportSEGY File

It is assumed that the segy files contain geogaphic coordinates

v 0.2 Okctober 2017 adding binning support
v 0.3 October 2018:
        - fixed utm Navigation CONVERSION
        - added option DROP_ZERO_NAV to remove 0 navigation values during import
v 0.4 October 2018
        - started to add argparse to control from command line: verbosity
        - add INTERPOLATE_NAV
v 0.5 November 2018
	- changed gain implementation
    - fixed plotting bug
v 0.6 July 2019
	- add alternative utm conversion with pyproj to allow conversion into a specified UTM zone

"""

import os
import sys
import pandas as pd
import numpy as np
import argparse




parser = argparse.ArgumentParser()
#Required Arguments
#parser.add_argument("echo", help="echo this string")
#parser.add_argument("SCALE_FAKTOR_X", type=float, help="the base")

#Optional Arguments: Verbosity displays all commmands
parser.add_argument("-v","--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

### Für Konvertierung
PFAD = '/Users/peter/TeamDropbox/Projects/SGD/2019_KLH_Wismar_SGD/ses'
os.chdir(PFAD)

IDENTIFIER = ".sgy" #Der Punkt vor der Endung muss dabei sein
SCALE_FAKTOR_Y = 1/(100.0*3600.0)
SCALE_FAKTOR_X = 1/(100.0*3600.0)  #Faktor für Navigation Correction ATZLER NWC = 10000
SCALE_FAKTOR_UTM = 1    # Scale Faktor for the UTM coordinates
ENDIAN = 1  #Wenn 0 nicht geht...dann ist es wohl 1
SHOTPOINTS_IN = 'tracl'   # headerword wereh shotpoints will be extracted from for NAV files

FILE_SELECT = 'allfiles'# oder newfiles allfiles  -> newfiles only consider segy files for which no su exists

## The follwoing options work only during data import
IMPORT = 'yes'
SUWIND='yes'
tmax=0.05


INTERPOLATE_NAV='yes' #Removes identical nav values and interpoltes linearly for each shot. Is embedded in the UTM conversion.

######

BINNING = 'yes'  # Bin everything with a certain distance. May have strange results if both stacking and binning are set to yes. Binning only works on UTM coordinates. Profile distance is written in the offset headerword
bin_overwrite = 'no'  #Overwrite origninal files or create new files with 'binned' added? no for no overwrite
GAIN = 'yes'
STACKING = 'no'  #Stack based on a certain number of shots
MERGE = 'no'
FILTER = 'yes'
PLOT = 'no'    #USES !DIST! to plot profile length
EXPORT='no'   #export to segy


#Butterworth
FSTOPLO = 2000  #Hz
FPASSLO = 6000
FPASSHI = 18000
FSTOPHI = 22000
#Gain
gain_string = 'tpow=2 qclip=0.99 gpow=0.4'
#Stapel
STACK = 2   #Number of traces to stack
#Bin
DIST = 1   #Distance in meters to bin



###KLUGES
SES_converted_data = 'no'   # SESconvert vertauscht versionsabhängig sx und sy keys in einer verion
ALTERNATIVE_UTM_CONVERSION = 'no' #convert all data in same UTM zone, for import in Kingdom. Uses myProj string and zone vertialbe below
utmzone = 32
ATZLER='no'  #The famous NWC format nightmare
###

#################################################################HELPER FUNCTIONS####################
def test_for_processed_files(files):
    # Test if files already processed -> Test if su file exists alraedy
    files_to_process = []
    for i in range(len(files)):
        sufile = files[i].strip(IDENTIFIER)
        test_file = sufile + '.su'
        print('Bearbeite Datei: ', files[i])
        files_to_process.append(files[i])
        df = pd.DataFrame(files_to_process)
        df.to_csv('last_import', header = False, index = False)
    return files_to_process

def get_su_file_name(segyfile, IDENTIFIER):
    sufile = segyfile.strip(IDENTIFIER)
    sufile = str(sufile) + '.su'
    sufile_utm = sufile.strip('.su')
    sufile_utm = sufile_utm + '_utm.su'
    return sufile, sufile_utm

def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    import os
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files

  #######################################################################PROGRAM START########################
if FILE_SELECT == 'allfiles':
    allfiles = getfiles(IDENTIFIER)
    files = allfiles
    print(files)
elif FILE_SELECT == 'newfiles':
    allfiles = getfiles(IDENTIFIER)
    files = test_for_processed_files(allfiles)
else:
    print('Dateiauswahl nicht korrekt')


if IMPORT == 'yes':
    print('Folgende Dateien werden bearbeitet')
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)

        # Konvertiere zu SU
        command= 'segyread ' + 'tape=' + str(element) + ' endian=' + \
        str(ENDIAN) +  ' | '  + ' segyclean'+ ' > ' + sufile
        if args.verbose:
            print(command)
        os.system(command)


        print("Dropping 0 Navigation")
        #Use sx keyword to removesuwind 0 Navigation
        command = 'suwind < ' + str(sufile) + ' key=sx min=1000 max=1000000000 > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp.su ' + sufile
        if args.verbose:
            print(command)
        os.system(command)

        #Retrieve nav
        HEADERWORDS = 'sx,sy,'+ SHOTPOINTS_IN   #Headerworte mit Geo-Informationen
        headerfile = str(sufile) + '.header'
        command = 'sugethw <' + sufile + ' key=' + HEADERWORDS + ' >' + headerfile
        if args.verbose:
            print(command)
        os.system(command)

        #Read in pandas and do stuff
        f = open(headerfile,'r')
        df = pd.read_table(f,header=None, delim_whitespace=True)
        print('Konvertiere: ', element, ' zu ', sufile)
        # Remove uneccessary part and convert to int
        df.replace('=', ' ',regex=True)
        for col in df.columns.values:
            df[col] = df[col].replace('[^0-9-]','',regex=True)
            df[col] = df[col].astype(float)
        f.close()


        df[0] = df[0] * SCALE_FAKTOR_X
        df[1] = df[1] * SCALE_FAKTOR_Y

        #%% Convert Lat Long Data to UTM
        print("Umrechnen zu UTM: ", sufile , ' zu ', sufile_utm)
        if INTERPOLATE_NAV == 'yes':
            #Difference between sx and sy
            df[3] = df[0].diff() #sx
            df[4] = df[1].diff() #sy

            df_filtered_sx = df[(df[3] != 0)]
            df_filtered_sy = df[(df[4] != 0)]

            # Interpolate sx
            offset_sx = df_filtered_sx.loc[0,2] #First shotpoint not always 0
            Index_sx = df_filtered_sx[2].unique() - offset_sx
            Index_sx[-1] = df.iloc[-1,2] - offset_sx  # Replace last entry with last shotpoint

            for index, value in enumerate(Index_sx[:-1]):  #we skip the last loop
                start = Index_sx[int(index)]
                stop = Index_sx[int(index) + 1]
                range =  Index_sx[int(index) + 1] - Index_sx[int(index)] + 1

                interp_array = np.linspace(df.loc[start,0], df.loc[stop,0], range)
                df.loc[start:stop,0] = interp_array

            # Interpolate sy
            offset_sy = df_filtered_sy.loc[0,2] #First shotpoint not always 0

            Index_sy = df_filtered_sy[2].unique() - offset_sy
            Index_sy[-1] = df.iloc[-1,2] - offset_sy  # Replace last entry with last shotpoint

            for index, value in enumerate(Index_sy[:-1]):  #we skip the last loop
                start = Index_sy[int(index)]
                stop = Index_sy[int(index) + 1]
                range =  Index_sy[int(index) + 1] - Index_sy[int(index)] + 1

                interp_array = np.linspace(df.loc[start,1], df.loc[stop,1], range)
                df.loc[start:stop,1] = interp_array
        print(df.head())

        #Conversion
        X=[]
        Y=[]
        TRACL=[]
        ZONE=[]
        Filename=[]
        LAT=[]
        LONG=[]
        if ALTERNATIVE_UTM_CONVERSION == 'no':
            import utm
            for r in zip(df[0],df[1], df[2]):
	            if SES_converted_data == 'yes':
	               utmkoord = utm.from_latlon(r[1],r[0])   # return is easting northing, zone
	            if SES_converted_data == 'no':
	               utmkoord = utm.from_latlon(r[0],r[1])
	            Y.append(int(utmkoord[1]))
	            X.append(int(utmkoord[0]))
	            ZONE.append(utmkoord[2])
	            TRACL.append(r[2])
	            Filename.append(element)
	            LAT.append(r[0])
	            LONG.append(r[1])

        if ALTERNATIVE_UTM_CONVERSION == 'yes':
            from pyproj import Proj
            myProj = Proj("+proj=utm +zone=" + utmzone + "K, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
            for r in zip(df[0],df[1], df[2]):
                if SES_converted_data == 'yes':
                    #Kluge for an ISE Bug
                    UTMx, UTMy = myProj(r[1],r[0])
                if SES_converted_data == 'no':
                    UTMx, UTMy = myProj(r[0],r[1])
                Y.append(int(UTMy))
                X.append(int(UTMx))
                ZONE.append(utmzone)
                TRACL.append(r[2])
                Filename.append(element)
                LAT.append(r[0])
                LONG.append(r[1])

        df = pd.DataFrame({'Y' : Y,
                           'X' : X,
                           'Shot' : TRACL,
                           'Zone' : ZONE,
                           'Filename' : Filename,
                           'LAT': LAT,
                           'LONG' : LONG,
                           })
        if args.verbose:
            print(df)

        #Exportieren des Navfiles:
        export_name = element + '_NAV.csv'
        df.to_csv(export_name,index=False, columns=['LAT','LONG','X','Y','Zone','Shot','Filename'], float_format='%3f')
        # Zurückschreiben der Header in SU
        export = df[['X','Y']]
        print(export.head())
        export = export.astype(int)
        export.to_csv('tempfile',sep= ' ', index=False, header=False)

        command = 'a2b < tempfile n1=2 > input.bin'
        if args.verbose:
            print(command)
        os.system(command)

        sufile_utm = sufile.strip('.su')
        sufile_utm = sufile_utm + '_utm.su'
        command = 'sushw <' + sufile + ' infile=input.bin key=sx,sy > temp2.su'
        if args.verbose:
            print(command)
        os.system(command)

        # Correct Coordinate Scaler and unit
        command = 'sushw < temp2.su ' + 'key=scalco a=1 b=0 > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = 'sushw < temp.su ' + 'key=counit a=1 b=0 >' + sufile_utm
        if args.verbose:
            print(command)
        os.system(command)

        # Copy sufile_utm to sufile
        command = 'mv ' + sufile_utm + '  ' + sufile
        if args.verbose:
            print(command)
        os.system(command)

        #Setting scalco to 1: We are limited to 1m accuracy at the moment -> this could be improved
        command = 'sushw < ' + sufile + ' key=scalco a=1 > temp '
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp ' + sufile
        if args.verbose:
            print(command)
        os.system(command)

        #Cleaning
        command = 'rm *.header'
        if args.verbose:
            print(command)
        os.system(command)





if SUWIND=='yes':
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        print('Cutte Dateilänge für ', sufile)
        command = 'suwind < ' +  str(sufile) + ' tmax=' + str(tmax)  + ' > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp.su  ' + sufile
        if args.verbose:
            print(command)
        os.system(command)

        #%% Filter
if FILTER=='yes':
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        print('Filtere: ' , sufile)
        command = ('subfilt <' + sufile + ' fstoplo='+ str(FSTOPLO) + ' fpasslo='
               +str(FPASSLO) + ' fpasshi='+str(FPASSHI) + ' fstophi='+ str(FSTOPHI)
               + ' > temp.su')
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp.su  ' + sufile
        if args.verbose:
            print(command)
        os.system(command)
else:
    print('Es wird kein filtern durchgeführt')


#%% Gain
if GAIN == 'yes':
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        print('Apply Gain: ', sufile)
        command = 'sugain ' +str(gain_string) + ' <' + sufile  + ' >  temp.su'
        #command = 'sugain gagc=1 wagc=0.005 <' + sufile +' > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp.su  ' + sufile
        if args.verbose:
            print(command)
        os.system(command)
else:
    print("Kein Gain")



if STACKING=='yes':
   for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        print('Stacke: ' , sufile)
        command = 'suwind < ' + sufile + ' key=fldr | sushw key=tracf a=1 c=1 j=' + str(STACK)  + ' | sustack key=tracf > temp.su '
        if args.verbose:
            print(command)
        os.system(command)
        command = ' mv temp.su ' + sufile
        if args.verbose:
            print(command)
        os.system(command)
else:
    print('Es wird kein stacking durchgeführt')


if MERGE=='yes':
    print( "For merging, it is assumed that the file oder by alphabetical sorting is correct")
    print( "Further, the trace numbers in the tracl headerword are rewritten continously")
    print ("Therefore, the correct navigation needs to be stored in the file")
    print ("SU MIGHT CROP VARIABLE LENGTH DATASETS")
    outfile = str(files[0]) + '_to_'+ str(files[-1]) + '_merged.su'
    print( "Merging files to outfile: " , outfile)
    for index,file in enumerate(files):
        print("Working on merging file ", index , " named " ,file)
        if index==0:
            command = 'cat ' + file + ' > ' + outfile
            if args.verbose:
                print(command)
            os.system(command)
        elif index > 0:
            command = 'cat ' + file + ' >>  ' + outfile
            if args.verbose:
                print(command)
            os.system(command)
    #write continuous trace number
    command = 'sushw < ' + outfile + ' key=tracl a=1 b=1 > temp.su'
    if args.verbose:
        print(command)
    os.system(command)
    command = 'mv temp.su ' + outfile
    if args.verbose:
        print(command)
    os.system(command)

if BINNING == 'yes':
    print("BINNING")
    print("ASSUMING HEADER COORDIBATES IN UTM")
    def binning_helper_function(xyfile, SCALE_FAKTOR=SCALE_FAKTOR_UTM):
        f = open(xyfile,'r')
        df = pd.read_table(f,header=None, delim_whitespace=True)
        f.close()

        # Remove uneccessary part and convert to int
        df.replace('=', '',regex=False)
        for col in df.columns.values:
            df[col] = df[col].replace('[^0-9-]','',regex=True)
            df[col] = df[col].astype(int)

        #Apply Scale Faktor
        df[0] = df[0] *  SCALE_FAKTOR
        return df

    for file in files:
        sufile, sufile_utm = get_su_file_name(file, IDENTIFIER)

        # Read X and Y headerwords
        command = 'sugethw < ' + sufile + ' key=sx  > temp.x'
        if args.verbose:
            print(command)
        os.system(command)

        command = 'sugethw < ' + sufile + ' key=sy  > temp.y'
        if args.verbose:
            print(command)
        os.system(command)

        #Read in pandas and export xline ans yline
        df_x = binning_helper_function('temp.x')
        df_y = binning_helper_function('temp.y')

        # Merge
        df = pd.concat([df_x, df_y], axis = 1)
        df.columns = ['X', 'Y']

        # Calculate bin numbers
        df['X_distance'] = df.X - df.X.min()
        df['Y_distance'] = df.Y - df.Y.min()

        df['X_distance_diff'] = df.X_distance.diff()
        df['Y_distance_diff'] = df.Y_distance.diff()
        df = df.fillna(0)


        df['X_distance_cumsum'] = df.X_distance_diff.cumsum()
        df['Y_distance_cumsum'] = df.Y_distance_diff.cumsum()

        df['Distance_combined'] = np.sqrt((df.X_distance_cumsum**2 + df.Y_distance_cumsum**2))


        def custom_round(x, base=5):
            return int(base * round(float(x)/base))

        # Round to nearest DIST
        df['Distance_round'] = df.Distance_combined.apply(lambda x: custom_round(x, base=DIST))
        #Make new bins
        df['bin'] = (df.Distance_round.diff() != 0).cumsum()

        df.bin.to_csv('temp.bin', index=False, header = False)

        #Write bins to CDP
        command = 'a2b < temp.bin n1=1 > input.bin'
        if args.verbose:
            print(command)
        os.system(command)

        command = 'sushw <' + sufile + ' infile=input.bin key=cdp > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = 'mv temp.su '+ sufile
        if args.verbose:
            print(command)
        os.system(command)

        #Running sustack
        command = 'sustack < ' + sufile + ' key=cdp > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        if bin_overwrite == 'no':
            sufile = sufile.strip('.su') + '_binned.su'
        if args.verbose:
            print(command)
        command = 'mv temp.su ' +  sufile
        os.system(command)

        # Setting offset headerword as profile distance
        command = 'sushw < ' + sufile + ' key=offset a=0 b=' + str(DIST) + ' > temp.su'
        if args.verbose:
            print(command)
        os.system(command)
        command = 'mv temp.su ' +  sufile
        if args.verbose:
            print(command)
        os.system(command)


#%% PLOT
if PLOT == 'yes':
    print("Plotting png overview image of each sufile")
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        psfile = sufile.strip('.su') + '.ps'
        pngfile = sufile.strip('.su') + '.png'
        svgfile = sufile.strip('.su') + '.svg'
        command = 'supsimage < ' + sufile + ' d2=' + str(DIST) + ' perc=98 title=' + sufile + '  > ' + psfile
        if args.verbose:
            print(command)
        os.system(command)
        command = 'gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -dGRAPHICSAlphaBitsa=4 -dDEVICEWIDTHPOINTS=1000 -dDEVICEHEIGHTPOINTS=1000 -sOutputFile=' + pngfile + ' ' + psfile
        os.system(command)


#%% EXPORT
if EXPORT=='yes':
    for element in files:
        sufile, sufile_utm = get_su_file_name(element, IDENTIFIER)
        exportfile = sufile
        segyexport = exportfile.strip('.su')
        try:
            segyexport = segyexport + str(ZONE[0]) +  '.sgy'
        except:
            print("Could not determine UTM Zone.")
            print("Please enter UTM Zone")
            print("Manual zone entering required that UTM number does not change across the dataset")
            ZONE = []
            ZONE.append(input())
        print('Exportiere: ' , sufile, ' zu: ', segyexport)
        command = 'segyhdrs < ' + exportfile
        if args.verbose:
            print(command)
        os.system(command)
        command = 'segywrite < ' + exportfile + ' hfile=header bfile=binary endian=0 tape=' + segyexport
        if args.verbose:
            print(command)
        os.system(command)

else:
    print('Es wird kein Export durchgeführt')
