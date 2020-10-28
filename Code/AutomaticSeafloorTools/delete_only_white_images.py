

#!/usr/bin/env python
# coding: utf8

#check for images that are only white and delete these images
# white is defined for everything more than >250 on average in brightness

import os
import sys
import argparse
from tqdm import tqdm
import skimage
import numpy as  np

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count() - 1
print("Using ", num_cores, " cores. ")

PARSER = argparse.ArgumentParser()

#Required Arguments
PARSER.add_argument('source_directory', type=str, help="Folder with input mosaics")
PARSER.add_argument('wildcards', type=str, help="identfiy the files")
PARSER.add_argument('threshold', type=float, help="Threshold, above is considered white. Range 0-1", default = 255)
def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for match in os.listdir(PFAD):
        if match.endswith(ID):
            files.append(str(match))
    return files



try:
    options = PARSER.parse_args()
except:
    PARSER.print_help()
    sys.exit(0)


args = PARSER.parse_args()
args.source_directory.strip("/")

files = getfiles(args.wildcards, args.source_directory)
print('Working on ', len(files) , ' files.')
def delete_white(file):
    img = skimage.io.imread(args.source_directory + '/' +  file, as_gray = True)
    if np.mean(img) > args.threshold:
        cmd = 'rm' +  '  ' +  args.source_directory + '/' + file
        os.system(cmd)

Parallel(n_jobs=num_cores)(delayed(delete_white)(file) for file in tqdm(files))

