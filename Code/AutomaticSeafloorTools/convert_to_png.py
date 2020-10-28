#!/usr/bin/env python
# coding: utf8
'''
Create tf datasets from the training and validation image folders
'''


import argparse
from tqdm import tqdm
import os
import sys
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")

parser = argparse.ArgumentParser()
#Required Arguments
parser.add_argument('directory', type=str, help="Folder with data")

try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()
args.directory.strip("/")
print("TIF warnings are supressed")

def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files


filelist = getfiles('.tif', args.directory)

def make_png(directory, image):
    cmd = 'mogrify -quiet -format png ' + '"' + directory + '/' + image + '"'
    os.system(cmd)
    return


Parallel(n_jobs=num_cores)(delayed(make_png)(args.directory, image) for image in tqdm(filelist))
