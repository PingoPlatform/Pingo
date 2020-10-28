#!/usr/bin/env python
# coding: utf8

'''
After training the model, run it over the list of file, extract the boxes and scores and labels, convert back to real coordinates, and save as csv
'''
# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from tqdm import tqdm

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from shapely.geometry import Point
import pandas as pd
import glob
import os, gdal


def get_boundaries(image):
    '''
    Bestimmen der Bildgrenzen
    '''
    src = gdal.Open(image)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    return ulx, xres, uly, yres, lrx, lry


def convert_pixel_to_real(image, box):
    # box : A list of 4 elements (x1, y1, x2, y2).
    ulx, xres, uly, yres, lrx, lry = get_boundaries(image)
    box_ulx = box[0]
    box_uly = box[1]
    box_lrx = box[2]
    box_lry = box[3]

    assert yres < 0   #otherwise, upper left is not the origin of the image

    # Convert pixel to real
    box_ulx_coord = ulx + (box_ulx * xres)
    box_uly_coord = uly + (box_uly * yres) #yres ist negativv
    box_lrx_coord = ulx + (box_lrx * xres)
    box_lry_coord = uly + (box_lry * yres)
    box_mean_x = (box_ulx_coord + box_lrx_coord) /2
    box_mean_y = (box_uly_coord + box_lry_coord) /2

    return box_ulx_coord, box_uly_coord, box_lrx_coord, box_lry_coord, box_mean_x, box_mean_y


#############START

image_folder = '/Volumes/Work/valid_orig_20pixels'
image_type = '.tif'
output = '/Volumes/Work/out_valid_original.csv'
labels_to_names = {0: 'stone'}
model_path = os.path.join('/Users/peter/IOW Marine Geophysik Dropbox/KI_Training_Sets/Weights/Object_Detection/resnet50_csv_14_runSR14_63mAP.h5')
convert_model = 'yes'
detection_threshold = 0.3  #include detections with accuracy above
min_side=300
boundary_threshold = 0.4  #Stones smaller together will be merged

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
if convert_model == 'yes':
    model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'stone'}

# get image list
image_list = glob.glob(image_folder + '*' + image_type)

#iterate over list
results = []
print("Working on Folder ", image_folder)
for img in tqdm(image_list):
    #start = time.time()
    image = read_image_bgr(img)
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side = min_side)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    #get coordinates
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < detection_threshold:
            break
        box_ulx_coord, box_uly_coord, box_lrx_coord, box_lry_coord, box_mean_x, box_mean_y  = convert_pixel_to_real(img, box)

        results.append(dict({'image' : img, 'class' : label, 'score' : score, 'ulx': box_ulx_coord, 'uly' : box_uly_coord , 'lrx' : box_lrx_coord, 'lry' : box_lry_coord, 'x': box_mean_x, 'y' : box_mean_y , 'WKT': str('POLYGON ((' + str(box_ulx_coord) + ' ' + str(box_uly_coord) + ',' +  str(box_ulx_coord) + ' ' + str(box_lry_coord) + ',' + str(box_lrx_coord) + ' ' + str(box_lry_coord) + ',' + str(box_lrx_coord) + ' ' + str(box_uly_coord) + ',' + str(box_ulx_coord) + ' ' + str(box_uly_coord) + '))') }))

#wegschreiben
df = pd.DataFrame(results)

print('Entferne Koordinaten mit Entfernunge < als: ', boundary_threshold)
merge_list = []
for row in df.itertuples():
    #bounds return (minx, miny, maxx, maxy)
    idx = row.Index
    x = row.x
    y = row.y
    near_points = df[(np.abs(df.x.values - x) < boundary_threshold ) & (np.abs(df.y.values - y) < boundary_threshold )].index
    merge_list.append(near_points)

    # df.loc[merge_list[element]].agg({'class':'mean', 'lrx':'mean'}) # so koennte man die Mittelwerte ausrechnen
# quick and dirtz: delete all antries except first
for element in merge_list:
    to_del = element[1:]
    try:
        df.drop(labels = to_del, inplace = True)
    except:
        continue

print("Berechne bounding box Fläche unter der Annahme vom projizierten Koordinates")
df['Area_bounding_box'] = np.abs((df.uly - df.lry)) * np.abs((df.ulx - df.lrx))

print(df.head)
df.to_csv(output, index = None)
