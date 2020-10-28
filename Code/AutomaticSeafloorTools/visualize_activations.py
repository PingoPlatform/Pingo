#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:04:28 2017

@author: peter
"""

import keras.backend as K
from keras import models
from scipy.misc import imread
from scipy.misc import imshow
import matplotlib.pyplot as plt


def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()
   

modelname = '/home/peter/databox/NeuralSeafloor/model_pairs/ANN_stone_det_75perc'


model_input = '/home/peter/databox/NeuralSeafloor/Data/Stone_Training_Data_Franz/steine_etc/Decompose/test/stone/stone_6.png'
img1 = imread(model_input,flatten = True)
#img_red = fa.image_stretch(img, greylevels)
img_shaped1 = img1.reshape((1, 20, 20, 1))
# Load Model
model = models.load_model(modelname)     
activation_maps=get_activations(model, img_shaped1)
display_activations(activation_maps)
plt.imshow(img1, cmap='Greys')
plt.show()

model_input = '/home/peter/databox/NeuralSeafloor/Data/Stone_Training_Data_Franz/steine_etc/Decompose/test/nostone/nostone_163.png'
img1 = imread(model_input,flatten = True)
#img_red = fa.image_stretch(img, greylevels)
img_shaped1 = img1.reshape((1, 20, 20, 1))
# Load Model
model = models.load_model(modelname)     
activation_maps=get_activations(model, img_shaped1)
display_activations(activation_maps)
plt.imshow(img1, cmap='Greys')
plt.show()