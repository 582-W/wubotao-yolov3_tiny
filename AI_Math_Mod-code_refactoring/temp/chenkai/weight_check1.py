import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np 
import os
import matplotlib.pyplot as plt
import json


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import * 
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input,decode_predictions



m = load_model('/home/chenkai/model zoo/vgg16.h5')
model = onnx.load('/home/chenkai/model zoo/vgg16h5.onnx')
node_num = len(model.graph.node)

weights = {}
dim = {}
for i in range (node_num):


    # obtain node's name form model graph
    weight_name = model.graph.node[i].name
    # check whether current node contains weight(this include weight and bias)
    # if node's op_type is 'Constant', the node contains weight. 
    if model.graph.node[i].op_type == "Constant":

        # create a multiple dict step by step.
        weights[weight_name] = {}
        dim[weight_name] = {}            
        
        # obtain the weight 
        weights[weight_name] = model.graph.node[i].attribute[0].t.float_data
        # obtain the origin shape of current weight
        dim[weight_name] = model.graph.node[i].attribute[0].t.dims 
        # reshape the 1 dim weight to 4 dims weight if it should be. For those bias, just keep 1 dim.      
        weights[weight_name] = np.reshape(weights[weight_name], dim[weight_name])
        # reshape weights to keras'h .h file style. For those bias, do nothing.
        if weights[weight_name].ndim == 4:
            weights[weight_name] = weights[weight_name].transpose(2, 3, 1, 0)
            keras_weight_name = weight_name[:-7]
            keras_weights = m.get_layer(keras_weight_name).get_weights()[0]
        elif weights[weight_name].ndim == 2:
            keras_weight_name = weight_name[:-7]
            keras_weights = m.get_layer(keras_weight_name).get_weights()[0]
        else:
            keras_weight_name = weight_name[:-5]
            keras_weights = m.get_layer(keras_weight_name).get_weights()[1]
        
        if ((keras_weights == weights[weight_name]).all()):
            print('The onnx file weight value in layer' + ' ' + weight_name + ' is same as keras file')
        else:
            print(False)


        
        
        

        



