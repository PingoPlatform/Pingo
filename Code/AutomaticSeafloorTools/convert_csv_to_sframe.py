# turi create uses sfrage. our qgis workflow results in
# csv files that need to be converted

# this script - from the turicrate doc page, modified to work with the qgis export files
# convert the csv into an sframe
# our csv_output is:
# the image files neet to be present as .png - so convert from .tif before

# note: sframe here do not handle empty coordinates, which were supported by retinanet
# if this in an annotion file, empty line are deleted with sed '/pattern/d' file > file , where
# pattern is something like ,,

import turicreate as tc
import os
import argparse


parser = argparse.ArgumentParser()
#Required Arguments
parser.add_argument('directory', type=str, help="Folder with image data")
parser.add_argument('csv', type=str, help="csv file with image description")
parser.add_argument('sframe_out', type=str, help="name of sframe_out file")

try:
    options = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

args = parser.parse_args()
args.directory.strip("/")



IMAGES_DIR = args.directory # Change if applicable
csv_path = args.csv  # assumes CSV column as above
csv_sf = tc.SFrame.read_csv(csv_path, header = False, na_values = '')


def row_to_bbox_coordinates(row):
    """
    Takes a row and returns a dictionary representing bounding
    box coordinates:  (center_x, center_y, width, height)  e.g. {'x': 100, 'y': 120, 'width': 80, 'height': 120}
    """
    return {'x': row['X2'] + (row['X4'] - row['X2'])/2,
            'width': (row['X4'] - row['X2']),
            'y': row['X3'] + (row['X5'] - row['X3'])/2,
            'height': (row['X5'] - row['X3'])}


csv_sf['coordinates'] = csv_sf.apply(row_to_bbox_coordinates)
# delete no longer needed columns
del csv_sf['X2'], csv_sf['X4'], csv_sf['X3'], csv_sf['X5']
# rename columns
csv_sf = csv_sf.rename({'X6': 'label', 'X1': 'name'})


# Load all images in random order
sf_images = tc.image_analysis.load_images(IMAGES_DIR, recursive=True,
                                          random_order=True)

# Split path to get filename
info = sf_images['path'].apply(
    lambda path: os.path.basename(path).split('/')[:1])

# Rename columns to 'name'
info = info.unpack().rename({'X.0': 'name'})

# Add to our main SFrame
sf_images = sf_images.add_columns(info)

# Original path no longer needed
del sf_images['path']

# Combine label and coordinates into a bounding box dictionary
csv_sf = csv_sf.pack_columns(
    ['label', 'coordinates'], new_column_name='bbox', dtype=dict)

# Combine bounding boxes of the same 'name' into lists
sf_annotations = csv_sf.groupby('name',
                                {'annotations': tc.aggregate.CONCAT('bbox')})

#workaround, weil in "name" bei annotations immer noch der ganze pfad steht und dann der vergleich nicht geht
info = sf_annotations['name'].apply(
    lambda path: os.path.basename(path).split('/')[:1])
info = info.unpack().rename({'X.0': 'name'})
del sf_annotations['name']
sf_annotations = sf_annotations.add_columns(info)

print(sf_images)
print(sf_annotations)
# Join annotations with the images. Note, some images do not have annotations,
# but we still want to keep them in the dataset. This is why it is important to
# a LEFT join (how="left"). I changed that to only include images with annotation
sf = sf_images.join(sf_annotations, on='name')

# The LEFT join fills missing matches with None, so we replace these with empty
# lists instead using fillna.
sf['annotations'] = sf['annotations'].fillna([])

print(sf)

# Save SFrame
sf.save(args.sframe_out)
