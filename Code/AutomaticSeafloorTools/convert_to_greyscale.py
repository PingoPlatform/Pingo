#!/usr/bin/env python
# coding: utf8
'''
Convert RGB to grayscale images by copying the first band to a new file
'''
import os
import sys
import argparse
import glob
from tqdm import tqdm
import gdal
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count() -1

print("Using ", num_cores, " cores. ")
parser = argparse.ArgumentParser()

parser.add_argument('source_directory', type=str, help="Folder with input mosaics")
parser.add_argument('target_directory', type=str, help="Target folder for image files")
parser.add_argument('wildcards', type=str, help="identfiy the files. Give complete filename to work on one file only")
parser.add_argument("-t","--tag", type=str, help="Add Tag to beginning of converted file. Pass emtpy to overwrite files", default='Band_1')
parser.add_argument("-o", "--overwrite", action='store_true',
                    help="Replace Original files")


def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files

try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()
args.source_directory.strip("/")
args.target_directory.strip("/")

filelist = getfiles(args.wildcards, args.source_directory)

def make_grey(image):
    errors=[]
    try:
        src_ds = gdal.Open(args.source_directory + '/' + image)
        img_in = args.source_directory + '/' + image
        img_out = args.source_directory + '/' + args.tag + image

        if src_ds.RasterCount == 1:
            cmd = 'cp ' + img_in + ' ' + img_out
            os.system(cmd)
            #print('This image only contained one band and was copied:'), image            if args.overwrite:
            cmd ='mv ' + img_out + ' ' + img_in
            os.system(cmd)

        if src_ds.RasterCount > 1:
            cmd = 'gdal_translate -q -of png -b 1 ' +  img_in + ' ' +  img_out
            # Could theorethically also be ddone with gdal_calc for mor flexibility
            #cmd = 'gdal_calc.py -R ' + img_in + ' --R_band=1 -G ' + img_in + ' --G_band=2 -B ' + img_in + ' --B_band=3 --outfile=' + img_out + ' --calc=\"(R+G+B)/3\"'
            os.system(cmd)
            if args.overwrite:
                cmd ='mv ' + img_out + ' ' + img_in
                os.system(cmd)
    except:
        errors.append(image)
    return errors

errors = Parallel(n_jobs=num_cores)(delayed(make_grey)(image) for image in tqdm(filelist))
errors[:] = [x for x in errors if x]
print("Following files had error")
print(errors)
#for image in tqdm(filelist):
#    make_grey(image)
