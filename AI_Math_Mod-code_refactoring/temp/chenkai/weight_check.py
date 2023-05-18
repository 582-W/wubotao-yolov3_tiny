import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np 
import os
import matplotlib.pyplot as plt
import json


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
        else:
            pass

print(weights["block1_conv1_weight"].shape)