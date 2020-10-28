import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from skimage import data, img_as_float
from skimage import exposure
from skimage import io 

import argparse 


"""
Example use 
hist_stretch.py /Path/to/folder tif recursive
histogram stretches all tif files in folder and subfolders 
"""

PARSER = argparse.ArgumentParser()

# Required Arguments
PARSER.add_argument('source_directory', type=str, help="Folder with input mosaics")
PARSER.add_argument('wildcards', type=str, help="identfiy the files")
PARSER.add_argument('recursive', type=str, help="identfiy the files", default="no")

try:
    options = PARSER.parse_args()
except:
    PARSER.print_help()
    sys.exit(0)

args = PARSER.parse_args()
args.source_directory.strip("/")


def getfiles(ID='', PFAD='.', rekursive='no'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    import os
    import glob2
    files = []
    if rekursive == 'no':
        for file in os.listdir(PFAD):
            if file.endswith(ID):
                files.append(str(file))
    if rekursive == 'yes':
        files = glob2.glob(PFAD + '/**/*' + ID)
    return files


clip_limit = 0.1
print("Clip limit for adaptive equalization set to:", clip_limit)

clip_stretch = (2,98)
print("Percentiles for contrast stretching set to:", clip_stretch)


if args.recursive == "recursive":
    print("Searching for files in subfolders")
    files = getfiles(args.wildcards, args.source_directory, rekursive='yes')
else:
    print("Searching for files in folder:", args.source_directory)
    files = getfiles(args.wildcards, args.source_directory)

# Convert
for file in tqdm(files):
    img = io.imread(file)
    #rescaled image
    p2, p98 = np.percentile(img, clip_stretch)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2,p98))
    io.imsave(file + "_stretched.png", img_rescale, check_contrast=False)

    #hist-eq image
    img_eq = exposure.equalize_hist(img)
    io.imsave(file + "_equalization.png", img_eq, check_contrast=False)

    #adapative-histe
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=clip_limit)
    io.imsave(file + "_adaptive.png", img_adapteq, check_contrast=False)
