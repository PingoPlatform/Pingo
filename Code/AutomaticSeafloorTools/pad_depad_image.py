import sys
import argparse
import os
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import gdal
from osgeo import ogr 
from osgeo import osr
from joblib import Parallel, delayed
import multiprocessing
"""
This sscript takes only the first dimension of input images(!) - so only greyscale at the moment - and pads it by zeros at each side. 

For RGB images, this likely has to be done for each band separately and then later on merged by gdal (note for later mbes related studies)
"""

#Required Arguments
parser = argparse.ArgumentParser()
parser.add_argument('source_directory', type=str,
                    help="Source with input mosaics")
parser.add_argument('target_directory', type=str,
                    help="Reference folder for image files")
parser.add_argument('pad_or_depad', type=str,
                    help="pad for padding, depad for depadding")
parser.add_argument('npad', type=int, help="how many zeros to add")
parser.add_argument('wildcards', type=str, help="identfiy the files")

num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")


def depad(source_directory, file, ext, npad, target_directory=0):
    """
    Pads a geotiff image by adding npad pixels to each edge.
    """
    todepad = gdal.Open(source_directory + '/' + file)

    if args.source_directory == args.target_directory:
        outfile = target_directory + '/' + file + '_depad' + ext
    else:
        outfile = target_directory + '/' + file + ext

    gt = todepad.GetGeoTransform()
    colortable = todepad.GetRasterBand(1).GetColorTable()
    data_type = todepad.GetRasterBand(1).DataType
    Itodepad = todepad.ReadAsArray()
    if Itodepad.ndim > 2:
        Itodepad = Itodepad[0, :, :]  # take onyl the first dimension
     # slice here
    ulx = gt[0] - gt[1] / npad
    uly = gt[3] - gt[5] / npad
#    lrx = gt[0] + gt[1] * (todepad.RasterXSize + npad)
#    lry = gt[3] + gt[5] * (todepad.RasterYSize + npad)

    # Make new geotransform
    gt_new = (ulx, gt[1], gt[2], uly, gt[4], gt[5])

    # Make padded raster (pad with zeros)
    raster = Itodepad[npad:-npad, npad:-npad]
    write_tile(raster, gt_new, todepad, outfile,
               dtype=data_type, color_table=colortable)
    return


def pad(source_directory, file, ext, npad, target_directory=0, padval=255):
    """
    Pads a geotiff image by adding npad pixels to each edge.
    """
    topad = gdal.Open(source_directory + '/' + file)
    if args.source_directory == args.target_directory:
        outfile = target_directory + '/' +  file + '_pad' + ext
    else:
        outfile = target_directory + '/' + file + ext

    gt = topad.GetGeoTransform()
    colortable = topad.GetRasterBand(1).GetColorTable()
    data_type = topad.GetRasterBand(1).DataType
    Itopad = topad.ReadAsArray()
    if Itopad.ndim > 2:
        Itopad = Itopad[0, :, :] #take onyl the first dimension
    ulx = gt[0] - gt[1] * npad
    uly = gt[3] - gt[5] * npad
#    lrx = gt[0] + gt[1] * (topad.RasterXSize + npad)
#    lry = gt[3] + gt[5] * (topad.RasterYSize + npad)

    # Make new geotransform
    gt_new = (ulx, gt[1], gt[2], uly, gt[4], gt[5])

    # Make padded raster (pad with zeros)
    raster = np.pad(Itopad, npad, mode='linear_ramp', end_values=padval)
    write_tile(raster, gt_new, topad, outfile, dtype=data_type, color_table=colortable)

    return 


def write_tile(raster, gt, data_obj, outputpath, dtype=gdal.GDT_UInt16, options=0, color_table=0, nbands=1, nodata=False):

    width = np.shape(raster)[1]
    height = np.shape(raster)[0]

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)

    # Write output raster
    if color_table != 0:
        dest.GetRasterBand(1).SetColorTable(color_table)

    dest.GetRasterBand(1).WriteArray(raster)

    if nodata is not False:
        dest.GetRasterBand(1).SetNoDataValue(nodata)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    wkt = data_obj.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())
    # Close output raster dataset 
    dest = None


def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args.source_directory.strip("/")
args.target_directory.strip("/")

filelist = getfiles(args.wildcards, args.source_directory)

if args.pad_or_depad == 'pad':
    Parallel(n_jobs=num_cores)(delayed(pad)(args.source_directory, file, ".tif", args.npad, args.target_directory, padval=128) for file in tqdm(filelist))
if args.pad_or_depad == 'depad':
    Parallel(n_jobs=num_cores)(delayed(depad)(args.source_directory, file, ".tif", args.npad, args.target_directory) for file in tqdm(filelist))
