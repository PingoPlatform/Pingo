#!/usr/bin/env python
# coding: utf8
'''
upsample images to create input data for superresolustion training data
'''

import os
import sys
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")

parser = argparse.ArgumentParser()

#Required Arguments

parser.add_argument('source_directory', type=str, help="Folder with input mosaics")
parser.add_argument('target_directory', type=str, help="Target folder for image files")
parser.add_argument('factor', type=int, help="upsampling factor")
parser.add_argument('wildcards', type=str, help="identfiy the files")

def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files


def upsample_image(folder, image_name, factor, target_folder):
    from skimage.transform import rescale
    from skimage.io import imsave
    from skimage.io import imread
    from skimage import img_as_ubyte

    image = folder + '/' + image_name
    img = imread(image)

    img_down = rescale(img, factor)
    img_down = img_as_ubyte(img_down)
    imsave(target_folder + '/' + image_name, img_down, check_contrast=False)
    return


try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


args = parser.parse_args()
args.source_directory.strip("/")
args.target_directory.strip("/")

files = getfiles(args.wildcards, args.source_directory)

print("Working on ", len(files), "images")

Parallel(n_jobs=num_cores)(delayed(upsample_image)(args.source_directory, file , args.factor, args.target_directory) for file in tqdm(files))
