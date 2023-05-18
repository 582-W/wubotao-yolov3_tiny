
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np 
import os
import matplotlib.pyplot as plt
import json



### the following function is to analysis weights
def WeightAnalyser(onnx_file_path, output_folder_path, analysis_mode = 'per_layer', conv_percent = 1.0):

    # load onnx model
    model = onnx.load(onnx_file_path)

    # get node number from onnx model
    node_num = len(model.graph.node)

    # check analysis mode 
    if analysis_mode == 'per_channel':
        pass
    else:
        analysis_mode = 'per_layer'

        # create an empty dic to store outputs. note that we use  a multiple dict.
        out_dict = {}
        # loop over node_num to analysis weights
        for i in range (node_num):

            weights = {}
            # obtain node's name form model graph
            weight_name = model.graph.node[i].name
            # check whether current node contains weight(this include weight and bias)
            # if node's op_type is 'Constant', the node contains weight. 
            if model.graph.node[i].op_type == "Constant":

                # create a multiple dict step by step.
                out_dict[weight_name] = {}
                out_dict[weight_name][analysis_mode] = {}                
                
                # obtain the weight 
                weights[weight_name] = model.graph.node[i].attribute[0].t.float_data
        #        print(np.shape(weights[weight_name]))
                w = weights[weight_name]

                ## the following is to delete abnormal values
                # sort weight value 
                w = np.sort(w, axis = None)
                w_l = len(w)
                # calculate  weight values to be kept
                start = int( w_l * (1 - conv_percent) / 2 )
                end = int(w_l * (1 + conv_percent) / 2) 
                w = w[start:end]

                # find max, min, mean, std of current node's weight
                out_dict[weight_name][analysis_mode]['max'] = np.max(w)
                out_dict[weight_name][analysis_mode]['min'] = np.min(w)
                out_dict[weight_name][analysis_mode]['mean'] = np.mean(w)
                out_dict[weight_name][analysis_mode]['std'] = np.std(w)

        # to write the out_dict to output_folder as a .json file 
        json_str = json.dumps(out_dict, indent=4)
        with open(output_folder_path + '/' + 'weight_analyser.json', 'w') as json_file:
            json_file.write(json_str)
    return out_dict


onnx_file_path = '/home/chenkai/model zoo/vgg16h5.onnx'
output_folder_path = '/home/chenkai/AIP/succ/onnx_analysis'
WeightAnalyser(onnx_file_path, output_folder_path, analysis_mode = 'per_layer', conv_percent = 1.0)




    