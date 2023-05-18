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

import numpy as np
import h5py
import json
import math


###load image and preprocess the image with keras.
#image path on your PC.
img_path = '/home/chenkai/AIP/succ/test_img/3689162471_5f9ffb5aa0.png'
#load the image and resize it to the right scale that VGG16 needs.
img = image.load_img(img_path, target_size=(224, 224))
#convert the image pixels to a numpy array
img = image.img_to_array(img)
#add one more dimension
img = np.expand_dims(img, axis=0)
#preprocess the image
img = preprocess_input(img)


###Implements the forward propagation for a convolution function
def conv_vgg(input_img, W, b):
 
    # Retrieve dimensions from input_img's shape
    (m, Height_prev, Width_prev, C_prev) = np.shape(input_img)
    #Retrieve dimensions from W's shape
    (f, f, C_prev, C) = np.shape(W)
    # Retrieve information of stride and pad based on keras' vgg16
    stride = 1
    pad = 1
    # Compute the dimensions of the CONV output volume
    Height = int((Height_prev - f + 2 * pad) / stride) + 1
    Width = int((Width_prev - f + 2 * pad) / stride) + 1
    # nitialize the output volume Z with zeros.
    conv_out = np.zeros((m, Height, Width, C))
    #reshape b to make sure that b's size is consistent with W
    b = b.reshape(1, 1, 1, W.shape[-1])
    # pad the input data(input_img) with zeros
    img_padded = np.pad(input_img, ((0, 0),(pad, pad),(pad, pad),(0, 0)), 'constant', constant_values=0)
    
    #convolution
    for i in range(m):                           # loop over the batch of examples
        image_in = img_padded[i]                 # Select ith  example's padded activation
        for h in range(Height):                  # loop over vertical axis of the output volume
            for w in range(Width):               # loop over horizontal axis of the output volume
                for c in range(C):               # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    h_start = h * stride
                    h_end = h * stride + f
                    w_start = w * stride
                    w_end = w * stride + f
                    # Use the corners to define the (3D) slice of image_in
                    conv_box = image_in[h_start:h_end, w_start:w_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    conv_out[i, h, w, c] = np.sum(conv_box * W[:, :, :, c]) + b[0, 0, 0, c]
   
    # check whether the output shape is correct
    assert(conv_out.shape == (m, Height, Width, C))
#    conv_out = relu(conv_out)
    return conv_out

def fc(inputMap, w, b):
    b = np.reshape(b, (1,) + b.shape)
    outputMap = np.dot(inputMap, w) + b
    return outputMap


def pooling(inputMap, poolSize=2, poolStride=2):

    # inputMap sizes
    a, in_row, in_col, in_dep = np.shape(inputMap)
    
    # outputMap sizes
    out_row, out_col = int(np.floor(in_row/poolStride)), int(np.floor(in_col/poolStride))
    out_dep = in_dep
    row_remainder, col_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride)
    if row_remainder != 0:
        out_row += 1
    if col_remainder != 0:
        out_col += 1
    outputMap = np.zeros((out_row, out_col, out_dep))
    
    # padding
    temp_map = np.lib.pad(inputMap, ((0, 0), (0, poolSize-row_remainder), (0, poolSize-col_remainder), (0, 0)), 'edge')
    
    # max pooling
    for d_idx in range(0, out_dep):
        for r_idx in range(0, out_row):
            for c_idx in range(0, out_col):
                startX = c_idx * poolStride
                startY = r_idx * poolStride
                poolField = temp_map[0, startY:startY + poolSize, startX:startX + poolSize, d_idx]
                poolOut = np.max(poolField)
                outputMap[r_idx, c_idx, d_idx] = poolOut
    
    outputMap = np.reshape(outputMap, (1,) + outputMap.shape)
    
    # return outputMap
    return outputMap


### define the relu function.
def relu(x):
    s = np.where(x < 0, 0., x)
    return s

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
     # print ("x_sum = ", x_sum)
    s = x_exp / x_sum
    return s

###define inference function for vgg16 onnx file 
def vgg16_onnx_inference(model_path, img):
    #load onnx model
    model = onnx.load(model_path)
    #compute node number of onnx model
    node_num = len(model.graph.node)
    # initialize node_name list
    node_name = [None] * node_num

    #create an empty dict to store each layer's output
    out = {}
    # the first element of out dict is input image
    out['input_1_o0'] = img
    # loop over node_num to infer
    for i in range (node_num):
        # find ith node's name. Note that the name is same as keras's .h file
        node_name[i] = model.graph.node[i].name
        # check if the current node just contains weight and bias
        if model.graph.node[i].op_type == "Constant":
            pass
        # check whether current node is a conv 
        if model.graph.node[i].op_type == "Conv":
            # obtain weight, bias, and their dims from i-2 and i-1 node.
            # reshpe and transpose data to make it consistent with keras 
            w  = model.graph.node[i - 2].attribute[0].t.float_data
            dim_w = model.graph.node[i - 2].attribute[0].t.dims
            w = np.reshape(w, dim_w)
            w = w.transpose(2, 3, 1, 0)
            b = model.graph.node[i - 1].attribute[0].t.float_data
            dim_b = model.graph.node[i - 1].attribute[0].t.dims
            b = np.reshape(b, dim_b)
            #obtain input from last layer
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            #compute output of currnt node
            out_name = model.graph.node[i].output[0]
            out[out_name] = conv_vgg(input, w, b)

        #check whether current node is a relu node 
        if model.graph.node[i].op_type == "Relu":
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            out_name = model.graph.node[i].output[0]
            out[out_name] = relu(input)

        #check whether current node is a maxpool node 
        if model.graph.node[i].op_type == "MaxPool":
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            out_name = model.graph.node[i].output[0]
            out[out_name] = pooling(input, poolSize=2, poolStride=2)

        # check whether current node is a flatten node 
        if model.graph.node[i].op_type == "Flatten":
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            out_name = model.graph.node[i].output[0]
            out[out_name] = input.flatten()

        # check whether current node is a fc node
        if model.graph.node[i].op_type == "Gemm":
            w  = model.graph.node[i - 2].attribute[0].t.float_data
            dim_w = model.graph.node[i - 2].attribute[0].t.dims
            w = np.reshape(w, dim_w)
            b = model.graph.node[i - 1].attribute[0].t.float_data
            dim_b = model.graph.node[i - 1].attribute[0].t.dims
            b = np.reshape(b, dim_b)
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            out_name = model.graph.node[i].output[0]
            out[out_name] = fc(input, w, b)

        # check whether current node is a softmax node 
        if model.graph.node[i].op_type == "Softmax":
            last_layer = model.graph.node[i].input[0]
            input = out[last_layer]
            out_name = model.graph.node[i].output[0]
            out[out_name] = softmax(input)


    return out
