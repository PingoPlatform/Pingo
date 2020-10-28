#!/usr/bin/env python
# coding: utf8
'''
Check folder for correct tile size of images and delete those that dont fit
'''



import argparse
from tqdm import tqdm
import os
import sys
from osgeo import gdal

parser = argparse.ArgumentParser()
#Required Arguments
parser.add_argument('directory', type=str, help="Folder with data")
parser.add_argument('tiles', type=int, help="tile size of images")
parser.add_argument('wildcards', type=str, help="identifier for Files")

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


filelist = getfiles(args.wildcards, args.directory)

counter=0
for image in tqdm(filelist):
    img = args.directory + '/' + image
    rds = gdal.Open(img)
    img_width,img_height=rds.RasterXSize,rds.RasterYSize
    if not img_width==args.tiles:
        cmd = 'rm ' + img
        os.system(cmd)
        counter = counter + 1
    if not img_height==args.tiles:
        cmd = 'rm ' + img
        os.system(cmd)
        counter = counter +1
print("A total of ", counter, " files have been deleted ( ", counter/len(filelist)* 100, " Percent.")
