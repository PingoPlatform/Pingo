
# Use Neural Networks to identify boulders in backscatter mosaics

This tutorial is even more to do then the others, given we still look for the best way. A description of backgrounds can be found here: https://www.mdpi.com/2076-3263/9/4/159 and here: https://www.mdpi.com/2072-4292/12/14/2284

The first publication also includes a supplement of test and training backscatter mosaics and manualy identified objects, that can be used for testing implementations of automatic detection methods. 

The object detection makes use of the fantastic repository hosting RetinaNet
https://github.com/fizyr/keras-retinanet
which is basically a one-stop solution for our purpose. The works that remains is related to preparing input and training data, and preparing the output. The main adaptions have to made because we want to preserve geographic coordinates throughout the process. Assuming that Retina-Net is installed. The mentioned tools are located in the "AutomaticSeaflorTools" folder in Code repository of PINGO-PLATFORM. There is very little in terms of documentation. The required command line arguments are at the moment listed in the program files. 

1. Prepare training data: Pick boundary boxes of boulders (and optionally negative examples) from backatter mosaics using e.g. the open source QGIS. The bounding boxes have to be exported as sqlite databases with the coordinates written in WKT format. Coordinates of mosaics and database should be UTM WGS84. Examples are stored provided in the articles linked above. 
2. Create small tiles (eg. 100x100 pixels) by using cutout_of_training_daty.py. This program can cut large mosaics in small tiles with overlap. This can create a lot of files. 
3. Relate database of QGIS and tiles: Also in cutout_pf_training_data.py there is an option to generate csv files from the sqlite database exported from Qgis. The program can also buffer points, if only point data exists. It further adds a class name and splits the results in an training and validation .csv file. At the moment, the program has to be run for each class separately (we had only 1 class, the boulders), and the resulting .csv files merged with cat csv1 csv2 > csvmerge.

These files can then be used to train a model with RetinaNet. Once a model exists,
it can be applied to backscatter mosaics. Cut the backscatter mosaics in tiles as above, then run the program apply_object_detection.py from the AutomaticSeafloorTools folder. The script includes several variables that have to be set, including the path to the files, a detection_threshold (detections with lower scores are discarded), the image minimum size in pixels tha the input is upscaled to, and a boundary threshold that represents the minimum distance between to detections.