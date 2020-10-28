#!/usr/bin/env python
# coding: utf8
'''
Add suffix to filename
'''



import argparse
from tqdm import tqdm
import os
import sys

parser = argparse.ArgumentParser()
#Required Arguments
parser.add_argument('directory', type=str, help="Folder with data")
parser.add_argument('wildcards', type=str, help="identifier for Files")
parser.add_argument('suffix', type=str, help="add text before wildcard")

try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()
args.directory.strip("/")

def getfiles(ID='', PFAD='.'):
    # Gibt eine Liste mit Dateien in PFAD und der Endung IDENTIFIER aus.
    files = []
    for file in os.listdir(PFAD):
        if file.endswith(ID):
            files.append(str(file))
    return files


filelist = getfiles(args.wildcards, args.directory)

for file in tqdm(filelist):
    file_base = file.strip(args.wildcards)
    file_new = file_base + args.suffix + args.wildcards
    cmd = 'cp ' + args.directory + '/' + file + ' ' + args.directory + '/' + file_new
    os.system(cmd)
