#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 11:01:28 2018

@author: peter
"""

# Input File:
# X Y Line Trace Time Amplitude
# !!!!! In the input file, make sure there is always ONE SPACE between columns
# this has to be changed in an editor at the moment

import os
os.chdir('/Users/peter/GoogleDrive')
import pandas as pd
import numpy as np

#Load the datafile
df = pd.read_csv('test_horizon.dat', sep=' ',index_col = False,  names=['X', 'Y', 'Line', 'Trace', 'Time', 'Amplitude'])


# Get deltax and deltay and detlat
delta_x = df['X'].max() - df['X'].min()
delta_y = df['Y'].max() - df['Y'].min()
distance = np.sqrt(delta_x**2 + delta_y**2)

delta_t = df['Time'].max() - df['Time'].min()
# convert time to depth
vel = 1587 # m/s
delta_depth = delta_t * vel/2

#calculate the cosine of the angle
cosalpha = delta_depth / distance

# Print the angle
angle = np.arccos(cosalpha)
print angle
