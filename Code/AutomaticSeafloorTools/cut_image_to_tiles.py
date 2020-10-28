#!/usr/bin/env python
# coding: utf8
'''
This script takes an input mosaic and cuts it into user defined rectangles.

It requires that the gdal command line utilities are installed
'''

import os
import sys
import argparse
import glob
from tqdm import tqdm
import gdal

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")

parser = argparse.ArgumentParser()

#Required Arguments

parser.add_argument('source_directory', type=str, help="Folder with input mosaics")
parser.add_argument('target_directory', type=str, help="Target folder for image files")
parser.add_argument('tile_size', type=int, help="size of the squares in pixels")
parser.add_argument('wildcards', type=str, help="identfiy the files")

#Optional Arguments: Verbosity displays all commmands
parser.add_argument("-o","--overlap", type=int, help="number of overlap between pixels", default=0)


def execute_command(command):
    os.system(command)
    return
    
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
print(args.overlap)
files = getfiles(args.wildcards, args.source_directory)

print("Tiling Mosaics:")

for mosaic in files:
    print(mosaic)




print("NOTICE: ALL input images must have the same number of bands NOTICE")
print("Split the dataset if required or convert all to greysscale")
skipped_files = []
for mosaic in tqdm(files):
    try:
        if args.overlap == 0:
            print("No overlap")
            cmd = 'gdal_retile.py -ps ' + str(args.tile_size) + ' ' + str(args.tile_size) + ' -targetDir ' + \
            '"' + args.target_directory + '"' + ' ' + '"' + \
            args.source_directory + '/' + mosaic+'"' 
            #os.system(cmd)
            Parallel(n_jobs=num_cores)(delayed(execute_command)(str(cmd))
                                       for file in tqdm(files))
        if args.overlap > 0:
            print("Cutting with Overlap")
            cmd = 'gdal_retile.py -ps ' + str(args.tile_size) + ' ' + str(args.tile_size) + ' -overlap ' + str(args.overlap) + ' -targetDir ' + '"' + args.target_directory + '"' + ' ' + '"' +  args.source_directory + '/' + mosaic+'"' 
            #os.system(cmd)
            Parallel(n_jobs=num_cores)(delayed(execute_command)(str(cmd))
                                       for file in tqdm(files))

    except:
        skipped_files.append(mosaic)

#print list of files that did not work for further investigation
if skipped_files:
    print('Skipped Files:', skipped_files)
    print('Most likely these mosaics include more than 1 band. Convert to true grayscale images.')
