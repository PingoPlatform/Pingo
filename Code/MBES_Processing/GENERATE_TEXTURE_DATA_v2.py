# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:01:24 2016
#Makes GLCM data from TIF files

@author: feldens


Todo: Make the monster loop run in parallel to speed up processing

Calcualtes texture parameters from a TIF image and stores as ASCII

Add gridding of output files using gmt surface

"""

#%% Variablen eingeben
import os
import sys
import pandas as pd
sys.path.insert(0, './functions')
import FUNCTIONS_GLCM as glcm
import numpy as np
import skimage
from numpy.lib.stride_tricks import as_strided as ast
from raster2xyz.raster2xyz import Raster2xyz
from tqdm import tqdm

#%%
# Pfad zu den TIFs Muss mit / enden
PFAD = '/Users/peter/git/retina_seafloor/applydata/'


input_files = ['D164_SSS_UG4_25cm_JV_20190520.tif']   # List of filenames
outname = '_nofaktor_result.csv'   #



#GLCM Parameters
greylevels=[31]   #greylevels !! 0 mitzÃ¤hlen!!
glcmangles=[0, np.pi*0.25, np.pi*0.5, np.pi*0.75]
#glcmangles=[0]
resolution = [20]  # in grid-Zellen
glcm_distances = [1]

normalize_0_255 = 'no'

# Sollen die Ergenisse als Grid weggeschrieben werden?
grid_res = 1   #das ist effektiv ein faktor, mit dem die resolution der glcm parameter multipliziert wird
filtered_grid ='no'  # create a second grid file with gaussian filter

#%%
####Programmspezifische Funktionen und Variablen###############################
parameter = ['ENTROPY', 'HOMOGENEITY', 'CORRELATION', 'ENERGY',
             'CONTRAST', 'DISSIMILARITY', 'MAXPROB', 'GLCMMEAN'
             ] # dont change, likely breaks code

def flatten( alist ):
    # flatten a nested list into a flat list
     newlist = []
     for item in alist:
         if isinstance(item, list):
             newlist = newlist + flatten(item)
         else:
             newlist.append(item)
     return newlist


## Block-View funktion durch modul von scikit-image ersetzt, die Funktion hier wird eigentlich nicht mehr benötigt.
def block_view(A, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def make_coord_grid(datafile, values, pad_y, pad_x, index='Y', columns = 'X',):
    coordgrid = datafile.pivot(index = 'Y', columns = 'X', values=values)
    coordgrid = np.asarray(coordgrid)
    coordgrid = np.flipud(coordgrid)
    #Auch padden - wird hinterher entfernt
    coordgrid =  np.lib.pad(coordgrid, [(0,pad_y),(0,pad_x)], 'constant', constant_values=(-1, -1))
    return coordgrid

def setup_dataframe():
    glcm_dataframe = pd.DataFrame({'X' : [],
                                       'Y' : [],
                                       'ENTROPY' :[],
                                       'HOMOGENEITY' : [],
                                       'CORRELATION' : [],
                                       'ENERGY' : [],
                                       'CONTRAST' : [],
                                       'DISSIMILARITY' : [],
                                       'MAXPROB': [],
                                       'GLCMMEAN' : [],
                                       'Greylevels' : [],
                                       'PatchSize' : [],
                                       'GLCM_Distance' : []
                                    })
    return glcm_dataframe

def convert_tif_to_raster(tif, out='temp.csv'):
    print('Convert to csv: ', tif)
    rtxyz = Raster2xyz()
    rtxyz.translate(tif, out)
    return out


################################################################################
#%%
os.chdir(PFAD)
for greylevel in greylevels:
    for glcm_distance in glcm_distances:
        for r in resolution:
            resolution_x = r
            resolution_y = r
            for image in input_files:
                print('Working on: ', image)
                #Ergenislisten anlegen
                glcm_results = []
                glcm_dataframe = setup_dataframe()
                #TIF konvertieren
                csv_file = convert_tif_to_raster(image)
                # TIF einlesen
                df = pd.read_csv(csv_file, sep=',', names = ['X','Y','Intensity'], skiprows=1)
                df = df.apply(pd.to_numeric)
                # Vorbereitung
                Intensity_min = df.Intensity.min()
                Intensity_max = df.Intensity.max()
                print(Intensity_min, Intensity_max)
                # scale datafile intensity to 0-greylevels
                print("Normiere Intensitätsdaten auf 0 - greylevel Interval")
                df['Intensity'] = (df['Intensity'] - Intensity_min) / (Intensity_max - Intensity_min) * greylevels
                print('Erstelle Tabelle Daten')
                # Tabelle mit UTM x und Y als Index erstellen fÃ¼r Intensity
                datagrid = df.pivot(index='Y', columns='X', values='Intensity')
                ##
                datagrid = np.asarray(datagrid)
                datagrid = np.flipud(datagrid)

                # Padden um glat durch resolution_x bzw _y glatt teilabr zu sein. Padden mit negativen Werten (-1), die werden spÃ¤ter aussortiert
                # kein einfluss auf koordinaten, da da direkt indizes abgefragt werden
                pad_y = (int(datagrid.shape[0] / resolution_y) + 1 ) * resolution_y - datagrid.shape[0]
                pad_x = (int(datagrid.shape[1] / resolution_x) + 1)  * resolution_x - datagrid.shape[1]
                datagrid =  np.lib.pad(datagrid, [(0,pad_y),(0,pad_x)], 'constant', constant_values=(-1, -1))

                #Nan als negativ maskieren
                datagrid = np.ma.array(datagrid, mask=np.isnan(datagrid), fill_value = -1)
                datagrid = np.ma.array(datagrid, mask=-1, fill_value = -1)

                print('Erstelle X,Y-Koordinaten und padde das grid analog zu den Daten')
                y_coordgrid = make_coord_grid(df, 'Y', pad_y, pad_x)
                x_coordgrid = make_coord_grid(df, 'X', pad_y, pad_x)

                # Block windows erzeugen
                print("Erstelle patches für GLCM Analyse")
                datagrid_blockview  = skimage.util.view_as_blocks(datagrid, block_shape=(resolution_y, resolution_x))

                #datagrid_blockview = block_view(datagrid, block= (resolution_y, resolution_x))
                num_y = datagrid_blockview.shape[0]
                num_x = datagrid_blockview.shape[1]

                print('Iteriere durch Patches')
                for y in tqdm(range(num_y)):
                    for x in range(num_x):

                        current_subgrid = datagrid_blockview[y][x]
                        if np.isnan(current_subgrid).any():
                            continue
                        if current_subgrid.min() < 0:
                            continue
                        y_index = int(y * resolution_y - resolution_y/2)
                        x_index = int(x * resolution_x + resolution_x/2)

                        y_coord = y_coordgrid[y_index][x_index]
                        x_coord = x_coordgrid[y_index][x_index]
                        values = []
                        #erstellt die glcm
                        glcmdata = glcm.greycomatrix(np.round(current_subgrid).astype(int), glcm_distance, glcmangles, greylevel+1)
                       #berechnet die parameter
                        for element in parameter:
                            templist =[]
                            templist = glcm.greycoparameters(glcmdata, glcmangles, element)
                            values.append(templist)

                        templist = []
                        templist.append([x_coord.item(), y_coord.item(), values])
                        flattened_list = flatten(templist)
                        glcm_results.append(flattened_list)


                #nur durchfÃ¼hren wenn glcm results nicht leer ist (leeres list sind False, volle Listen sind True)
                if glcm_results:
                    temp = np.array(glcm_results)
                    temp_df = pd.DataFrame({'X' : temp[:,0],
                                           'Y' : temp[:,1],
                                           'ENTROPY' : temp[:,2],
                                           'HOMOGENEITY' : temp[:,3],
                                           'CORRELATION' : temp[:,4],
                                           'ENERGY' : temp[:,5],
                                           'CONTRAST' : temp[:,6],
                                           'DISSIMILARITY' : temp[:,7],
                                            'MAXPROB': temp[:,8],
                                            'GLCMMEAN' : temp[:,9],
                                           })

                    #Drop negative coordinates from padding
                    temp_df = temp_df[temp_df.X > 0]
                    temp_df= temp_df[temp_df.Y > 0]


                    if normalize_0_255 == 'yes':
                        faktor = (255-1)+1
                        for element in parameter:
                            temp_df[element]  = ((temp_df[element] - temp_df[element].min()) / (temp_df[element].max() - temp_df[element].min())) * faktor

                    # Daten zusammensammeln
                    print('Appending to result dataframe')
                    glcm_dataframe = glcm_dataframe.append(temp_df)


#%%
                # Aussortieren der negativen Werte vom Padden
                #glcm_dataframe.drop([glcm_dataframe['X'] < 0], axis=0, inplace = True)

                print("Saving result dataframe into separate files")
                for column in tqdm(glcm_dataframe):
                    print("Working on : ", column)
                    x_min = int(np.floor(glcm_dataframe.X.min()))
                    x_max = int(np.ceil(glcm_dataframe.X.max()))
                    y_min = int(np.floor(glcm_dataframe.Y.min()))
                    y_max = int(np.ceil(glcm_dataframe.Y.max()))
                    region = str(str(x_min) + "/" + str(x_max) + '/' + str(y_min) + "/" + str(y_max))
                    if column in parameter:
                        file_out = image + '_' + str(greylevel) + '_' + str(r) + '_' +str(glcm_distance) + '_' + column + '_' + outname
                        grid_out = str(file_out) + ".nc"
                        glcm_dataframe.to_csv( file_out , index = False, header = False, columns = ['X', 'Y', column])

                        #gridding
                        command = "gmt surface " + file_out + " -I" + str(grid_res) +  " -R" + region +" -G" + grid_out
                        print(command)
                        os.system(command)

                        if filtered_grid == 'yes':
                            filtered_grid_out = str(file_out) + "_smooth.nc"
                            command = "gmt grdfilter " + grid_out + " -D0 -Fg2 "+ " -G"+ filtered_grid_out
                            os.system(command)
