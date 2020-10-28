#!/usr/bin/env python
# coding: utf8
'''
Take a grayscale image and make a fake RGB image with each channel having the grayscale value
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
parser.add_argument("-t","--tag", type=str, help="Add Tag to beginning of converted file. Pass emtpy to overwrite files", default='')


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

print("Converting files grom grayscale to RGB:")

def make_rgb(image):
    img_name = args.source_directory + '/' + image
    img_out_name = args.target_directory + '/' + args.tag + image
    cmd = 'gdal_translate -quiet -of PNG -b 1 -b 1 -b 1 '  + img_name + '  ' + args.target_directory + '/temp.png'
    os.system(cmd)
    cmd = 'mv ' + args.target_directory +  '/temp.png ' + img_out_name
    os.system(cmd)

Parallel(n_jobs=num_cores)(delayed(make_rgb)(image) for image in tqdm(filelist))

