#!/usr/bin/env python
# coding: utf8
'''
Split files in a folder randomly on training and validation datasets
'''

import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

#Required Arguments

parser.add_argument('source_directory', type=str, help="Folder with all training data")
parser.add_argument('target_directory', type=str, help="A percentage of files is moved in this folder")
parser.add_argument('valid_percentage', type=float, help="Percentage of validation dataset")
parser.add_argument('wildcards', type=str, help="identfiy the files")


def random_distribution(source_folder, target_folder, valid_percentage = 0.2):
    import os
    import random
    import shutil

    filelist = os.listdir(source_folder)

    number_of_files = len(filelist)
    print(number_of_files)

    index_of_files =  range(0,number_of_files-1)
    print(index_of_files)

    number_of_valid_files = int(number_of_files * valid_percentage)

    random_files_index = random.sample(index_of_files, number_of_valid_files)
    for i in tqdm(random_files_index):
        shutil.move(source_folder + '/'+ filelist[i], target_folder +'/'+ filelist[i])
    return


try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()

args.source_directory.strip("/")
args.target_directory.strip("/")

print("Splitting all files in folder:", args.source_directory)

random_distribution(args.source_directory, args.target_directory, args.valid_percentage)
