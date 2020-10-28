import sys
import argparse
import os
import numpy as np
import math
import pandas as pd
from skimage.measure import compare_ssim as ssim
from skimage.io import imread
from tqdm import tqdm
parser = argparse.ArgumentParser()

#Required Arguments

parser.add_argument('source_directory', type=str, help="Source with input mosaics")
parser.add_argument('target_directory', type=str, help="Reference folder for image files")
parser.add_argument('out_directory', type=str, help="Target folder for result file")
parser.add_argument('name', type=str,help="name of output file")
parser.add_argument('wildcards', type=str, help="identfiy the files")
parser.add_argument('-t', '--target_prefix', type=str, help="Prefix added to target files", default = '')
parser.add_argument('-s', '--histo_stretch', type=int, help="Histogram stretch reference images", default = '0')

# define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):

    # assume gray image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    return err

def histo_stretch(image):
    equ_image = cv2.equalizeHist(image)
    return equ_image

# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel =True))
    return scores

def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args.source_directory.strip("/")
args.target_directory.strip("/")
args.out_directory.strip("/")

files_source = getfiles(args.wildcards, args.source_directory)
files_target = getfiles(args.wildcards, args.target_directory)

#get files which are in both lists
print("Number of source files: ", len(files_source))
print("Number of target files: ", len(files_target))
file_names = intersection(files_source, files_target)
print("Number of common files: ", len(file_names))

results = []
for file in tqdm(file_names):
    # open target and reference images
    try:
        image  = imread(args.source_directory + '/' + file, as_gray = True)
        reference = imread(args.target_directory + '/'  + args.target_prefix + file, as_gray = True)
    except:
        print("No matching file for: ", file)
        continue
    #stretch histogram of reference image
    if args.histo_stretch == 1:
        image = histo_stretch(image)
        reference = histo_stretch(reference)

    # calculate score
    scores = compare_images(image, reference)

    # print all three scores with new line characters (\n)
#    print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(file, scores[0], scores[1], scores[2]))
    results.append([scores[0], scores[1], scores[2], file])

df = pd.DataFrame(results, columns = ['PSNR', 'MSE', 'SSIM', 'Filename'])
print(df.head)
df.to_csv(args.out_directory + '/' + args.name + '.csv')
print(df.mean())
print(df.std())
