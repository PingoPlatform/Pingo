#!/usr/bin/env python
# coding: utf8
'''
Cute image tiles centered around points that
are stored as WKT points in SQLite database and creates annotation files in csv, tfrecord or json format.
'''

import os
import sys
import argparse
import glob
from tqdm import tqdm
import gdal
import pandas as pd
import geopandas
import pyproj
import matplotlib
import numpy as np
from osgeo import gdal, ogr, osr
import sqlite3
import random
from shapely import wkt
import glob
import gdal
import numpy as np
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser()

#Required Arguments
parser.add_argument('--image_directory', type=str,help="Folder with input mosaics")
parser.add_argument('--wildcards', type=str,help="identfiy the image files")
parser.add_argument('--database_directory', type=str, help="Target directory for annotation files")
parser.add_argument('--input_databases', action='append', help='<Required> Specify input mosaic databases. order is important')
parser.add_argument('--input_classes', action='append', help='<Required> Specify which class is stored in each database. Order is important')
parser.add_argument('--out_directory', help='<Output directory')
parser.add_argument("-f", "--format", type=str,help="Output format of annotation file. Possible values are csv and tfrecord", default='csv')

appendix_for_annotation_file = 'annotation_20pix_stone_'

##NOTICE
# the classname empty is special. This is assumed to be just points, not rectangles (so it is easier to dogotize negative image examples), and will be buffered by a few pixels, with coordinates later removed as required by retina net


def get_boundaries(image):
    '''
    Bestimmen der Bildgrenzen
    '''
    src = gdal.Open(image)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    return ulx, xres, uly, yres, lrx, lry

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Strip '/'
args.image_directory.strip("/")
args.database_directory.strip("/")
args.out_directory.strip("/")

# Get Table name from file names

print(args.input_databases)
table_names = []
for tollesding in args.input_databases:
    table_name=''
    table_name = tollesding.strip("sqlite")
    table_name = table_name.strip(".")
    table_names.append(table_name)
print(table_names)


assert len(args.input_databases) == len(args.input_classes) == len(table_names)
databases = args.input_databases
class_names = args.input_classes

print("relate point to mosaic:")
data = [] #make empty result list
for i, (database, class_name, table_name) in enumerate(zip(databases, class_names, table_names)):
    print('Working on: ' , database, class_name, table_name)
    cn = sqlite3.connect(args.database_directory + '/' + database)
    sql_query = "SELECT * FROM " + table_name
    vector_df = pd.read_sql_query(sql_query, cn)

    #Create geopandas
    vector_df['Coordinates'] = vector_df['WKT_GEOMETRY'].apply(wkt.loads)
    vector_gdf = geopandas.GeoDataFrame(vector_df, geometry='Coordinates')
    # get image list
    image_list = glob.glob(args.image_directory + '/' + '*' + args.wildcards)

    results = []
    for image in image_list:
        ulx, xres, uly, yres, lrx, lry = get_boundaries(image)
        results.append([image, ulx, uly, lrx, lry, xres, yres])

    raster_df = pd.DataFrame(results, columns = ['image', 'ulx', 'uly', 'lrx', 'lry', 'xres', 'yres'])

    # add emtpy imagename columns to vector gdf and add class
    vector_gdf['image_path'] = ''
    vector_gdf['classname'] = class_name

    ##Compare the two data frames
    for row in vector_gdf.itertuples():
        #bounds return (minx, miny, maxx, maxy)
        idx = row.Index
        minx = float(row.Coordinates.bounds[0])
        miny = float(row.Coordinates.bounds[1])
        maxx = float(row.Coordinates.bounds[2])
        maxy = float(row.Coordinates.bounds[3])

        if class_name == 'empty':
            buffer_value = 4
            #this allows "rebuffer"
            mean_x = (float(minx) + float(maxx)) /2
            mean_y = (float(maxy) + float(miny))/2

            minx = mean_x - buffer_value/2
            maxx = mean_x + buffer_value/2
            miny = mean_y - buffer_value/2
            maxy = mean_y + buffer_value/2
            print("Buffer")


        image_exists = []

        image_exists = raster_df[(raster_df.ulx.values < minx) & (raster_df.uly.values > maxy) & (raster_df.lrx.values > maxx) & (raster_df.lry.values < miny )]

        if not image_exists.empty:
            for image in image_exists.iterrows():
                # get unqie list of objects/images and calculate pixels
                image_xmin = image[1].ulx
                image_ymax = image[1].uly

                image_res_y = image[1].yres
                image_res_x = image[1].xres

                pixel_ymin = np.abs(int((image_ymax - maxy) / image_res_y))
                pixel_ymax = np.abs(int((image_ymax - miny) / image_res_y))
                pixel_xmin = np.abs(int((minx - image_xmin) / image_res_x))
                pixel_xmax = np.abs(int((maxx - image_xmin) / image_res_x))

                if pixel_ymin == pixel_ymax:
                    print('Image resolution to low for picked data, skipping entry (ymin == ymax)')
                    continue
                if pixel_xmin == pixel_xmax:
                    print('Image resolution to low for picked data, skipping entry (xmin == xmax')
                    continue

                data.append(dict({'classname': row.classname, 'imagename' : image[1].image, 'minx' : minx, 'miny' : miny, 'maxx' : maxx, 'maxy' : maxy, 'pixelxmax': pixel_xmax, 'pixelxmin' : pixel_xmin, 'pixelymax' : pixel_ymax, 'pixelymin': pixel_ymin}))


print("Make annotated csv files")
df = pd.DataFrame(data)
mask = (df['classname'] == 'empty')
df.at[mask, 'minx'] = 99999
df.at[mask, 'miny'] = 99999
df.at[mask, 'maxx'] = 99999
df.at[mask, 'maxy'] = 99999


if args.format == 'tfrecord':
# Reformat for usage with Tensorflow API and no longer Keras Retinanet
    print("Export as tfrecord not currently implemented here...this code sits in one of the Google Colabs")
    sys,exit(0)



if args.format == 'csv':
# Reformat for usage with Tensorflow API and no longer Keras Retinanet

  columns = ['imagename', 'pixelxmin', 'pixelymin', 'pixelxmax', 'pixelymax', 'classname']

  # Export Test and Train sets
  train, test = train_test_split(df, test_size=0.2)
  train.to_csv(args.out_directory + '/' + appendix_for_annotation_file + 'train.csv', header = None, index = None, sep = ',', columns = columns)
  test.to_csv(args.out_directory + '/' + appendix_for_annotation_file+ 'test.csv', header = None, index = None, sep = ',',  columns = columns)

  #resolve empty classes ..only for retinannet
  columns = ['imagename', 'pixelxmin', 'pixelymin', 'pixelxmax', 'pixelymax', 'classname']

  df = pd.read_csv(args.out_directory + '/' + appendix_for_annotation_file + 'train.csv', sep=',', header=None, names = columns)
  df.pixelxmax[df.classname == 'empty'] = str()
  df.pixelymin[df.classname == 'empty'] = str()
  df.pixelymax[df.classname == 'empty'] = str()
  df.pixelxmin[df.classname == 'empty'] = str()
  df.classname[df.classname == 'empty'] = str()
  df.to_csv(args.out_directory + '/' + appendix_for_annotation_file + 'train.csv', header = None, index = None, sep = ',', columns = columns)

  df = pd.read_csv(args.out_directory + '/' + appendix_for_annotation_file + 'test.csv', sep=',', header=None, names = columns)
  df.pixelxmax[df.classname == 'empty'] = str()
  df.pixelymin[df.classname == 'empty'] = str()
  df.pixelymax[df.classname == 'empty'] = str()
  df.pixelxmin[df.classname == 'empty'] = str()
  df.classname[df.classname == 'empty'] = str()
  df.to_csv(args.out_directory + '/' + appendix_for_annotation_file + 'test.csv', header = None, index = None, sep = ',', columns = columns)
