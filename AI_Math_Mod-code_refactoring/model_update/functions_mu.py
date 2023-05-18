import json
import math
import numpy as np
import onnx
import os
import copy
import sys,getopt
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnxparser.editor import OnnxEditor
from onnxparser.parser import OnnxParser
from inference.functions.functions_fl import mish as floatmish
from inference.functions.functions_fl import silu as floatsilu
def add_size_info(onnx_file_path: str, updated_onnx_path: str) -> onnx:
    """This function helps to add size information, which includes output_size and constant_node_size, to updated_model.
    Note that this will also add size information for new BN layer. 
    Args:
        onnx_file_path (str): the input onnx model for model_update
        updated_onnx_path (str): the onnx model after updating
    Returns:
        onnx: updated onnx model with extra size value_info.
    """
    original_model = onnx.load(onnx_file_path)
    umodel = onnx.load(updated_onnx_path)
    model_size_list = original_model.graph.value_info
    umodel_size_list = umodel.graph.value_info

    # get bn information 
    bn_output_dict = {}
    bn_set = set()
    for i in range(len(umodel.graph.node)):
        if umodel.graph.node[i].op_type == "BatchNormalization":
            bn_name = umodel.graph.node[i].name
            bn_out = str(umodel.graph.node[i].output[0])
            bn_set.add(bn_out)
            bn_output_dict[bn_name] = bn_out


    #copy size from original onnx except for new BN layer.
    update_output_set = set() 
    for i in range(len(umodel.graph.node)):
        update_output_name = umodel.graph.node[i].output[0]
        update_output_set.add(update_output_name)

    # copy size info.
    for index in model_size_list:
        if index.name in update_output_set:
            umodel_size_list.append(index)

    # add output size information for BN layer that model update adds  
    output_layer_name = set()
    for k in umodel.graph.output:
        output_layer_name.add(k.name)

    conv_next = get_conv_next_layer_ori(updated_onnx_path)
    helper = copy.deepcopy(model_size_list)

    for i in range(len(umodel.graph.node)):
        if umodel.graph.node[i].op_type == "Conv" and umodel.graph.node[i].output[0] not in output_layer_name:
            conv_node_name = umodel.graph.node[i].name
            conv_out_name = str(umodel.graph.node[i].output[0])
            conv_next_layers = conv_next[conv_node_name]
            for conv_next_layer in conv_next_layers:
                if conv_next_layer in bn_output_dict:
                    bn_out = bn_output_dict[conv_next_layer]
                    for k in helper:
                        if k.name + "special_o0" == conv_out_name:
                            temp = k
                            temp.name = conv_out_name
                            umodel_size_list.append(temp) 
    onnx.save(umodel, updated_onnx_path)
    return umodel

def get_conv_next_layer_ori(onnx_file_path: str)->dict:
    """To find the next layer of current conv.
    Args:
        onnx_file_path (str): onnx model file path
    Returns:
        dict: key--current conv name, value--next layer name
    """
    model = onnx.load(onnx_file_path)
    next_layer_dict = {}

    output_layer_name = []
    for k in model.graph.output:
        output_layer_name.append(k.name) 

    conv_out_dict = {}
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type == "Conv":
            conv_name = model.graph.node[i].name
            conv_out_dict[conv_name] = str(model.graph.node[i].output[0])

    name_dict = {}
    for i in range(len(model.graph.node)):
        node_name = model.graph.node[i].name
        name_dict[node_name] = i

    input_layer_name_list = []
    output_dict = {}# key: output_name, val: node_name
    input_dict = {} #key: input layer name, val: [list of layers having this input] 
    for i in range(len(model.graph.node)):

        if model.graph.node[i].op_type != "Constant":
            node_name = model.graph.node[i].name
            inputs = model.graph.node[i].input

            outputs = model.graph.node[i].output
            output_dict[outputs[0]] = node_name
            for ele in inputs:
                if "input" in ele.lower():
                    input_layer_name_list.append(ele)

            for ele in inputs:
                if not ele in name_dict.keys():
                    if not ele in input_dict.keys():
                        input_dict[ele] = []
                    input_dict[ele].append(node_name)
                elif model.graph.node[name_dict[ele]].op_type != "Constant": 
                    if not ele in input_dict.keys():
                        input_dict[ele] = []
                    input_dict[ele].append(node_name)  

    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type == "Conv" and model.graph.node[i].output[-1] not in output_layer_name:
            conv_name = model.graph.node[i].name
            next_layer_dict[conv_name] = input_dict[conv_out_dict[conv_name]]

    return next_layer_dict

def change_add_input_order(onnx_file_path: str, datapath_dict: dict, config_dict: dict, output_dict: dict) -> onnx:
    """This function helps to exchange add layer inputs based on radix calculated from datapath.
    Args:
        onnx_file (str): onnx file path
        datapath_dict (dict): Statistical information obtained from datpath
        config_dict (dict): hardware configuration dict, read from hardware.json file.
    Returns:
        onnx: onnx file after changing add layer inputs.
    """
    model = onnx.load(onnx_file_path)
    datapath_bit_width = config_dict['datapath_bitwidth']
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type == "Add":
            add_name = model.graph.node[i].name 
            input_layer_list = list(model.graph.node[i].input)
            assert (len(input_layer_list) == 2), "Add Layer Inputs Number Error"
            input_A = model.graph.node[i].input.pop()
            input_B = model.graph.node[i].input.pop()
            A_layer, B_layer = output_dict[input_A], output_dict[input_B]
            A_layer_max = max(abs(datapath_dict[A_layer]["layer_max"]), abs(datapath_dict[A_layer]["layer_min"]))
            B_layer_max = max(abs(datapath_dict[B_layer]["layer_max"]), abs(datapath_dict[B_layer]["layer_min"]))
            A_layer_radix = getradix(A_layer_max, datapath_bit_width)
            B_layer_radix = getradix(B_layer_max, datapath_bit_width)

            if A_layer_radix < B_layer_radix:
                model.graph.node[i].input.append(input_A)
                model.graph.node[i].input.append(input_B)
                datapath_dict[add_name]["inputs"][0] = input_A
                datapath_dict[add_name]["inputs"][1] = input_B
            else:
                model.graph.node[i].input.append(input_B)
                model.graph.node[i].input.append(input_A)
                datapath_dict[add_name]["inputs"][0] = input_B
                datapath_dict[add_name]["inputs"][1] = input_A

    return model, datapath_dict

def get_shift_bit(analysis_mode: str, config_dict: dict, updated_onnx_path: str, datapath_dict_update: str, \
                    weight_analy_dict_update: str, extra_info: dict, output_dict: dict, model_graph_input) -> dict:
    """Help to get shift bits of bn and conv layers based on analysis_mode.

    Args:
        analysis_mode (str): per_layer or per_channel_weight.
        config_dict (dict): hardware configuration dict, read from hardware.json file.
        updated_onnx_path (str): the path where updated onnx model is.
        datapath_dict_update (str): get this from datapath.
        weight_analy_dict_update (str): get this from weight analyser.
        extra_info (dict): dict to save results.
        output_dict (dict): key: output of layer, value: name of layer
        model_graph_input (str): input of whole model

    Returns:
        dict: containing  conv shift bits and bn shift bits(if the model has bn layers).
    """
          
    ### Save shift bit results as a json file.
    scale_info = extra_info['scale_info']
    conv_shift_dict = {}
    bn_shift_dict = {}
    datapath_bit_width = config_dict['datapath_bitwidth']
    kernel_bit_width = config_dict['conv_bitwidth']['kernel']
    bn_bit_width = config_dict["bn_bitwidth"]
    updated_model = onnx.load(updated_onnx_path)   

    if analysis_mode == 'per_layer':

        for i in range(len(updated_model.graph.node)):
            node_name = updated_model.graph.node[i].name
            if updated_model.graph.node[i].op_type == "Conv":
                input_name_list = updated_model.graph.node[i].input
                last_layer = input_name_list[0]
                
                # check whether current layer is a input layer
                if last_layer in model_graph_input:
                    input_max = datapath_dict_update['input']['layer_max']
                    input_radix = getradix(input_max, datapath_bit_width)
                if last_layer not in model_graph_input:
                    input_layer = output_dict[last_layer]
                    # input_max = datapath_dict_update[input_layer]['layer_max']
                    # input_min = datapath_dict_update[input_layer]['layer_min']
                    # input_abs_max = max(abs(input_max), abs(input_min))
                    # input_radix = getradix(input_abs_max, datapath_bit_width)     
                    input_radix = scale_info[input_layer]['y_radix'][0] 
                # out_max = datapath_dict_update[node_name]['layer_max']
                # out_min = datapath_dict_update[node_name]['layer_min']
                # out_abs_max = max(abs(out_max), abs(out_min))
                # out_radix = getradix(out_abs_max, datapath_bit_width)
                out_radix = scale_info[node_name]['y_radix'][0]

                w_max = weight_analy_dict_update[node_name]["kernel_max"]
                w_min = weight_analy_dict_update[node_name]["kernel_min"]
                w_abs_max = max(abs(w_max), abs(w_min))
                w_radix = getradix(w_abs_max, kernel_bit_width)
                conv_shift_dict[node_name] = out_radix - w_radix - input_radix
            
            if updated_model.graph.node[i].op_type == "BatchNormalization":
                input_name_list = updated_model.graph.node[i].input
                last_layer = input_name_list[0]

                if last_layer not in model_graph_input:
                    input_layer = output_dict[last_layer]
                    # input_max = datapath_dict_update[input_layer]['layer_max']
                    # input_min = datapath_dict_update[input_layer]['layer_min']
                    # input_abs_max = max(abs(input_max), abs(input_min))
                    # input_radix = getradix(input_abs_max, datapath_bit_width)
                    input_radix = scale_info[input_layer]['y_radix'][0]             
                # out_max = datapath_dict_update[node_name]['layer_max']
                # out_min = datapath_dict_update[node_name]['layer_min']
                # out_abs_max = max(abs(out_max), abs(out_min))
                # out_radix = getradix(out_abs_max, datapath_bit_width)
                out_radix = scale_info[node_name]['y_radix'][0]

                a_max = weight_analy_dict_update[node_name]["a_max"]
                a_min = weight_analy_dict_update[node_name]["a_min"]
                a_abs_max = max(abs(a_max), abs(a_min))
                a_radix = getradix(a_abs_max, bn_bit_width)
                bn_shift_dict[node_name] = out_radix - a_radix - input_radix


        extra_info['conv_shift_bit_info'] = conv_shift_dict
        extra_info['bn_shift_info'] = bn_shift_dict


    if analysis_mode == 'per_channel_weight':
        for i in range(len(updated_model.graph.node)):
            node_name = updated_model.graph.node[i].name
            if updated_model.graph.node[i].op_type == "Conv":
                input_name_list = updated_model.graph.node[i].input
                last_layer = input_name_list[0]

                # check whether current layer is a input layer
                if last_layer in model_graph_input:
                    input_max = datapath_dict_update['input']['layer_max']
                    input_radix = getradix(input_max, datapath_bit_width)
                if last_layer not in model_graph_input:
                    input_layer = output_dict[last_layer]
                    # input_max = datapath_dict_update[input_layer]['layer_max']
                    # input_min = datapath_dict_update[input_layer]['layer_min']
                    # input_abs_max = max(abs(input_max), abs(input_min))
                    # input_radix = getradix(input_abs_max, datapath_bit_width) 
                    input_radix = scale_info[input_layer]['y_radix'][0]           
                # out_max = datapath_dict_update[node_name]['layer_max']
                # out_min = datapath_dict_update[node_name]['layer_min']
                # out_abs_max = max(abs(out_max), abs(out_min))
                # out_radix = getradix(out_abs_max, datapath_bit_width)
                out_radix = scale_info[node_name]['y_radix'][0]

                w_max_list = weight_analy_dict_update[node_name]['group_max_list']
                w_min_list = weight_analy_dict_update[node_name]['group_min_list']
                w_radix_list = getradixlist(w_max_list, w_min_list, kernel_bit_width)
                conv_shift_dict[node_name] = [(out_radix - input_radix - w_radix_list[i]) for i in range(len(w_radix_list))]

            if updated_model.graph.node[i].op_type == "BatchNormalization":
                input_name_list = updated_model.graph.node[i].input
                last_layer = input_name_list[0]

                if last_layer not in model_graph_input:
                    input_layer = output_dict[last_layer]
                    # input_max = datapath_dict_update[input_layer]['layer_max']
                    # input_min = datapath_dict_update[input_layer]['layer_min']
                    # input_abs_max = max(abs(input_max), abs(input_min))
                    # input_radix = getradix(input_abs_max, datapath_bit_width) 
                    input_radix = scale_info[input_layer]['y_radix'][0]           
                # out_max = datapath_dict_update[node_name]['layer_max']
                # out_min = datapath_dict_update[node_name]['layer_min']
                # out_abs_max = max(abs(out_max), abs(out_min))
                # out_radix = getradix(out_abs_max, datapath_bit_width)
                out_radix = scale_info[node_name]['y_radix'][0]

                a_max_list = weight_analy_dict_update[node_name]['a_max_list']
                a_min_list = weight_analy_dict_update[node_name]['a_min_list']
                a_radix_list = getradixlist(a_max_list, a_min_list, bn_bit_width)
                bn_shift_dict[node_name] = [(out_radix - input_radix - a_radix_list[i]) for i in range(len(a_radix_list))]

        extra_info['conv_shift_bit_info'] = conv_shift_dict
        extra_info['bn_shift_info'] = bn_shift_dict

    return extra_info

def getradix(xmax, bitwidth):
    if xmax == 0:
        return 0
    radix = bitwidth - 1 - (math.floor(math.log2(xmax) + 1))
    return radix



def getradixlist(channel_max_list, channel_min_list, bitwidth):

    bitwidth -= 1
    assert len(channel_max_list) == len(channel_min_list)
    SFT_Bit_List = []
    for i in range(len(channel_max_list)):
        channel_max = max(abs(channel_max_list[i]), abs(channel_min_list[i]))
        SFT_Bit = bitwidth - (math.floor(math.log2(channel_max)+1))
        SFT_Bit_List.append(SFT_Bit)
    return SFT_Bit_List

def add_bn_after_conv_initializer_(model, node_conv, scale, channel_num, bn_count):
    """Add BN layer after Conv

    Args:
        model (onnx model): model to add node
        node_conv (onnx node): conv node which need to change
        scale (float): scaling factor for model update
        channel_num (int): BN channel numbers
        bn_count (int): BN layer number index
    """
    node = node_conv
    assert len(node.output)==1, 'there are multi-outputs in {}'.format(node.name)
    bn_name = 'batch_normalization_update' + str(bn_count)
    orig_node_output_name = node.output.pop()
    node_output_name = orig_node_output_name + "special_o0"
    node.output.append(node_output_name)
    input_list_name = [node_output_name]

    if type(scale) == float:
        momentum = 0.99
        epsilon =  10**-12
        gamma_np = [1 / scale] * channel_num
        beta_np = [0] * channel_num
        mean_np = [0] * channel_num
        var_np = [1] * channel_num

    elif type(scale) == list:
        momentum = 0.99
        epsilon =  10**-12
        gamma_np = [1 / scale[i] for i in range(channel_num)]
        beta_np = [0] * channel_num
        mean_np = [0] * channel_num
        var_np = [1] * channel_num
                        
    name_gamma = bn_name + "_gamma"
    input_list_name.append(name_gamma)
    OnnxEditor.add_initializer(model,name_gamma,gamma_np,(channel_num,))

    name_beta = bn_name + "_beta"
    input_list_name.append(name_beta)
    OnnxEditor.add_initializer(model,name_beta,beta_np,(channel_num,))           

    name_mean = bn_name + "_mean"
    input_list_name.append(name_mean)                
    OnnxEditor.add_initializer(model,name_mean,mean_np,(channel_num,))

    name_var = bn_name + "_var"
    input_list_name.append(name_var)
    OnnxEditor.add_initializer(model,name_var,var_np,(channel_num,))

    attribute = {
        "epsilon": epsilon,
        "momentum": momentum}
    
    node_add = OnnxEditor.make_node(op_tpye='BatchNormalization',name=bn_name,inputs=input_list_name,outputs=[orig_node_output_name],attribute=attribute)
    model.graph.node.append(node_add)
    

        
def is_update_conv_weight(weight_max, target_bitwidth = 8, threshold = 0.8):
    scale_flag = 0
    radix = getradix(weight_max, target_bitwidth)
    num_min = (2 **(target_bitwidth - 1)) * 2 ** (-radix - 1)   
    num_max = (2 ** (target_bitwidth - 1)) * 2 ** (-radix)
    assert num_min <= weight_max <= num_max, "the radix caculat wrong"
    ### set the scale

    if (num_min <= weight_max <= num_min + (num_max - num_min) * threshold):
        scale = num_max * 2 ** (-1) *0.95 / weight_max
        scale_flag = 1
        
    else:
        scale = 1
        scale_flag = 0

    return scale_flag, scale

def is_update_conv_weight_channel(w_kernel_max_list, w_kernel_min_list, target_bitwidth = 8, threshold = 0.8):
    channel_num = len(w_kernel_max_list)
    radix_list = getradixlist(w_kernel_max_list, w_kernel_min_list, target_bitwidth)
    num_min_list = [((2 **(target_bitwidth - 1)) * 2 ** (-radix_list[i] - 1)) for i in range(channel_num)]
    num_max_list = [((2 ** (target_bitwidth - 1)) * 2 ** (-radix_list[i])) for i in range(channel_num)]

    weight_max_list = [max(abs(w_kernel_max_list[i]), abs(w_kernel_min_list[i])) for i in range(channel_num)]
    for i in range(channel_num):
        assert num_min_list[i] <= weight_max_list[i] <= num_max_list[i], "the radix caculat wrong"

    ### set the scale
    scale_list = [1] * channel_num
    scale_flag_list = [0] * channel_num
    for i in range(channel_num):
        if (num_min_list[i] <= weight_max_list[i] <= num_min_list[i] + (num_max_list[i] - num_min_list[i]) * threshold):
            scale_list[i] = num_max_list[i] * 2 ** (-1) *0.95 / weight_max_list[i]
            scale_flag_list[i] = 1
        else:
            scale_list[i] = 1
            scale_flag_list[i] = 0
    
    return scale_flag_list, scale_list


def get_scale_info(datapath_dict,weight_analy_dict,config_dict,analysis_mode,limit_shift,shift_upper_limit,shift_lower_limit,node_list,former_layer_dict,next_layer_dict,last_node_list,model_graph_input):
    kernel_bitwidth = config_dict['conv_bitwidth']['kernel']
    datapath_bitwidth = config_dict['datapath_bitwidth']
    leaky_bitwidth = config_dict['leaky_relu_alpha']
    bn_bitwidth = config_dict["bn_bitwidth"]
    scale_dict = {}
    if analysis_mode == 'per_channel_weight':
        for i in range(len(node_list)):
            if node_list[i].op_type == 'Constant':
                continue
            layer = node_list[i].name

            '''
            set lut bitwidth.for yolov5s,silu's input node are all conv
            '''
            datapath_bitwidth = config_dict['datapath_bitwidth']
            if layer in last_node_list :
                pass
            else :
                next_layer_list = next_layer_dict[layer]
                for n in next_layer_list:
                    if datapath_dict[n]['op_type'] == "Silu" or datapath_dict[n]['op_type'] == "Mish":
                        datapath_bitwidth = config_dict['lut_bitwidth']
                        break



            layer_dict = {}
            layer_dict['output_bitwidth'] = datapath_bitwidth

            if model_graph_input[0] in node_list[i].input: # this layer is first layer, input is image
                iutput_max = max(abs(datapath_dict['input']['layer_max']), abs(datapath_dict['input']['layer_min']))
                x_radix = getradix(iutput_max, config_dict['datapath_bitwidth'])
                x_radix_list = [x_radix for _ in range(3)] # image has RGB channels
                layer_dict['x_radix'] = x_radix_list
                layer_dict['input_bitwidth'] = config_dict['datapath_bitwidth']
            else:
                former_layer_list = former_layer_dict[layer]
                if len(former_layer_list) == 1:
                    before_layer = former_layer_list[0]
                    layer_dict['x_radix'] = scale_dict[before_layer]['y_radix']
                else:
                    ## add, concat layer
                    for i in range(len(former_layer_list)):
                        layer_dict[f'x{i}_radix'] = scale_dict[former_layer_list[i]]['y_radix']
                layer_dict['input_bitwidth'] = scale_dict[former_layer_list[0]]['output_bitwidth']
            
            if datapath_dict[layer]['op_type'] == 'Conv':
                if layer in config_dict:
                    kernel_bitwidth = config_dict[layer]["kernel"]
                    print("!!!layer {}: specific weight kernel bitwidth config: {}, bias bitwidth: {}".format(layer, config_dict[layer]["kernel"], config_dict[layer]["bias"]))
                weight_max_list = list(map(abs, weight_analy_dict[layer]['group_max_list']))
                weight_min_list = list(map(abs, weight_analy_dict[layer]['group_min_list']))
                w_kernel_radix_list = []
                psum_radix_list = []
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(weight_max_list))]
                for j in range(len(weight_max_list)):
                    if weight_max_list[j] > weight_min_list[j]:
                        w_kernel_radix = getradix(weight_max_list[j], kernel_bitwidth)
                    else:
                        w_kernel_radix = getradix(weight_min_list[j], kernel_bitwidth)
                    #w_kernel_radix_list.append(w_kernel_radix)
                    if limit_shift == True:
                        if layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] > int(shift_upper_limit):
                            w_kernel_radix = int(shift_upper_limit) + y_radix_list[j] - layer_dict['x_radix'][0]
                        assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] <= int(shift_upper_limit),"datapath shift out of range"
                        assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] >= int(shift_lower_limit),"datapath shift out of range"
                    w_kernel_radix_list.append(w_kernel_radix)
                    if layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] > 15:
                        coarse_shift = 4*math.floor((layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] - 16)/4)+4
                    elif layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] < 0:
                        coarse_shift = -4
                    else:
                        coarse_shift = 0
                    psum_radix = layer_dict['x_radix'][0] + w_kernel_radix_list[j] - coarse_shift
                    psum_radix_list.append(psum_radix)
                layer_dict['kernel_weight_radix'] = w_kernel_radix_list
                layer_dict['bias_radix'] = y_radix_list
                layer_dict['y_radix'] = y_radix_list
                layer_dict['psum_radix'] = psum_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'BatchNormalization':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                a_max_list = list(map(abs, weight_analy_dict[layer]['a_max_list']))
                a_min_list = list(map(abs, weight_analy_dict[layer]['a_min_list']))
                a_radix_list = []
                for j in range(len(layer_dict['x_radix'])):
                    if a_max_list[j] > a_min_list[j]:
                        a_radix = getradix(a_max_list[j], bn_bitwidth)
                    else:
                        a_radix = getradix(a_min_list[j], bn_bitwidth)
                    bn_shift = y_radix - layer_dict['x_radix'][0] - a_radix + 8
                    assert (bn_shift <= 12),  "bn_shift out of range"
                    if bn_shift not in [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]:
                        if bn_shift < -10:
                            a_radix -= (-10 - bn_shift)
                        else:
                            a_radix -= 1
                    a_radix_list.append(a_radix)
                layer_dict['y_radix'] = y_radix_list
                layer_dict['bn_a_radix'] = a_radix_list
                layer_dict['bn_b_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'LeakyRelu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                alpha = weight_analy_dict[layer]['alpha']
                w_radix = 18
                # w_radix = getradix(alpha, leaky_bitwidth)
                w_radix_list = [w_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                layer_dict['w_radix'] = w_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Add':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Upsample':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Concat':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                length = 0
                for i in range(len(former_layer_list)):
                    length = length + len(scale_dict[former_layer_list[i]]['y_radix'])
                y_radix_list = [y_radix for _ in range(length)]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'MaxPool':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'L2Normalization':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Relu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Mish':
                input_bitwidth = scale_dict[before_layer]['output_bitwidth']
                input_radix = scale_dict[before_layer]['y_radix'][0]
                xmax = 2 ** (input_bitwidth - 1 - input_radix)
                ymax = floatmish(xmax)
                ymin = floatmish(-xmax)
                lookuptable_max = max(abs(ymax),abs(ymin))
                lenth_lut=datapath_bitwidth
                y_radix = getradix(lookuptable_max, lenth_lut)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            
        
            elif datapath_dict[layer]['op_type'] == 'Clip':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'PRelu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'AveragePool':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Flatten':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'upsample_yolov4':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Softplus':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Sigmoid':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Tanh':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Silu':
                input_bitwidth = scale_dict[before_layer]['output_bitwidth']
                input_radix = scale_dict[before_layer]['y_radix'][0]
                xmax = 2 ** (input_bitwidth - 1 - input_radix)
                ymax = floatsilu(xmax)
                ymin = floatsilu(-xmax)
                lookuptable_max = max(abs(ymax),abs(ymin))
                lenth_lut=datapath_bitwidth
                y_radix = getradix(lookuptable_max, lenth_lut)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Mul':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Reshape':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Expand':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Resize':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict
            else:
                print("datapath_dict op_type: ", datapath_dict[layer]['op_type'])
                assert False, 'There is an unsupported layer'
    
    
    if analysis_mode == 'per_layer':
        for i in range(len(node_list)):
            if node_list[i].op_type == 'Constant':
                continue
            layer = node_list[i].name
            '''
            set lut bitwidth.for yolov5s,silu's input node are all conv
            '''
            datapath_bitwidth = config_dict['datapath_bitwidth']
            if layer in last_node_list :
                pass
            else :
                next_layer_list = next_layer_dict[layer]
                for n in next_layer_list:
                    if datapath_dict[n]['op_type'] == "Silu" or datapath_dict[n]['op_type'] == "Mish":
                        datapath_bitwidth = config_dict['lut_bitwidth']
                        break

            layer_dict = {}
            layer_dict['output_bitwidth'] = datapath_bitwidth

            if model_graph_input[0] in node_list[i].input: ## this layer is first layer, input is image
                x_radix_list = []
                for j in range(3):## input image has RGB channels
                    iutput_max = max(abs(datapath_dict['input']['layer_max']), abs(datapath_dict['input']['layer_min']))
                    x_radix = getradix(iutput_max, datapath_bitwidth)
                    x_radix_list.append(x_radix)
                layer_dict['x_radix'] = x_radix_list
                layer_dict['input_bitwidth'] = config_dict['datapath_bitwidth']
            else:
                former_layer_list = former_layer_dict[layer]
                if len(former_layer_list) == 1:
                    before_layer = former_layer_list[0]
                    layer_dict['x_radix'] = scale_dict[before_layer]['y_radix']
                else:
                    ## add, concat layer
                    for i in range(len(former_layer_list)):
                        layer_dict[f'x{i}_radix'] = scale_dict[former_layer_list[i]]['y_radix']
                layer_dict['input_bitwidth'] = scale_dict[former_layer_list[0]]['output_bitwidth']


            if datapath_dict[layer]['op_type'] == 'Conv':
                kernel_num = int(weight_analy_dict[layer]['kernel_num'])
                kernel_max = weight_analy_dict[layer]['kernel_max']
                kernel_min = weight_analy_dict[layer]['kernel_min']
                weight_max = max(abs(kernel_max), abs(kernel_min))
                w_kernel_radix = getradix(weight_max, kernel_bitwidth)
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                
                if limit_shift == True:
                    if layer_dict['x_radix'][0] + w_kernel_radix - y_radix > int(shift_upper_limit):
                        w_kernel_radix = int(shift_upper_limit) + y_radix - layer_dict['x_radix'][0]
                    assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix <= int(shift_upper_limit),"datapath shift out of range"
                    assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix >= int(shift_lower_limit),"datapath shift out of range"
                if layer_dict['x_radix'][0] + w_kernel_radix - y_radix > 15:
                    coarse_shift = 4*math.floor((layer_dict['x_radix'][0] + w_kernel_radix - y_radix- 16)/4)+4
                elif layer_dict['x_radix'][0] + w_kernel_radix - y_radix < 0:
                    coarse_shift = -4
                else:
                    coarse_shift = 0
                psum_radix = layer_dict['x_radix'][0] + w_kernel_radix - coarse_shift

                y_radix_list = [y_radix for _ in range(kernel_num)]
                w_kernel_radix_list = [w_kernel_radix for _ in range(kernel_num)]
                psum_radix_list = [psum_radix for _ in range(kernel_num)]

                layer_dict['kernel_weight_radix'] = w_kernel_radix_list
                layer_dict['bias_radix'] = y_radix_list
                layer_dict['y_radix'] = y_radix_list
                layer_dict['psum_radix'] = psum_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'LeakyRelu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                alpha = weight_analy_dict[layer]['alpha']
                # w_radix = getradix(alpha, leaky_bitwidth)
                w_radix = 18
                w_radix_list = [w_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = layer_dict['x_radix']
                layer_dict['w_radix'] = w_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Add':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Upsample':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Concat':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                length = 0
                for i in range(len(former_layer_list)):
                    length = length + len(scale_dict[former_layer_list[i]]['y_radix'])
                y_radix_list = [y_radix for _ in range(length)]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'MaxPool':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'L2Normalization':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'BatchNormalization':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                a_max = weight_analy_dict[layer]['a_max']
                a_min = weight_analy_dict[layer]['a_min']
                a_abs_max = max(abs(a_max), abs(a_min))
                a_radix = getradix(a_abs_max, bn_bitwidth)
                bn_shift = y_radix - layer_dict['x_radix'][0] - a_radix + 8
                assert (bn_shift <= 12),  "bn_shift out of range"
                if bn_shift not in [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]:
                    if bn_shift < -10:
                        a_radix -= (-10 - bn_shift)
                    else:
                        a_radix -= 1
                a_radix_list = [a_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                layer_dict['bn_a_radix'] = a_radix_list
                layer_dict['bn_b_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Relu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Mish':
                input_bitwidth = scale_dict[before_layer]['output_bitwidth']
                input_radix = scale_dict[before_layer]['y_radix'][0]
                xmax = 2 ** (input_bitwidth - 1 - input_radix)
                ymax = floatmish(xmax)
                ymin = floatmish(-xmax)
                lookuptable_max = max(abs(ymax),abs(ymin))
                lenth_lut=datapath_bitwidth
                y_radix = getradix(lookuptable_max, lenth_lut)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict


            elif datapath_dict[layer]['op_type'] == 'Clip':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'PRelu':
                # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                # y_radix = getradix(output_max, datapath_bitwidth)
                # y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                # layer_dict['y_radix'] = y_radix_list
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'AveragePool':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            elif datapath_dict[layer]['op_type'] == 'Flatten':
                layer_dict['y_radix'] = scale_dict[before_layer]['y_radix']
                scale_dict[layer] = layer_dict
            
            elif datapath_dict[layer]['op_type'] == 'upsample_yolov4':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            
            elif datapath_dict[layer]['op_type'] == 'Mystery':
                break

            elif datapath_dict[layer]['op_type'] == 'Softplus':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Sigmoid':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Tanh':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            
            elif datapath_dict[layer]['op_type'] == 'Silu':
                input_bitwidth = scale_dict[before_layer]['output_bitwidth']
                input_radix = scale_dict[before_layer]['y_radix'][0]
                xmax = 2 ** (input_bitwidth - 1 - input_radix)
                ymax = floatsilu(xmax)
                ymin = floatsilu(-xmax)
                lookuptable_max = max(abs(ymax),abs(ymin))
                lenth_lut=datapath_bitwidth
                y_radix = getradix(lookuptable_max, lenth_lut)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict
            
            elif datapath_dict[layer]['op_type'] == 'Mul':
                output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
                y_radix = getradix(output_max, datapath_bitwidth)
                y_radix_list = [y_radix for _ in range(len(layer_dict['x0_radix']))]
                layer_dict['y_radix'] = y_radix_list
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Reshape':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Expand':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            elif datapath_dict[layer]['op_type'] == 'Resize':
                layer_dict['y_radix'] = layer_dict['x_radix']
                scale_dict[layer] = layer_dict

            else:
                print("datapath_dict op_type: ", datapath_dict[layer]['op_type'])
                assert False, 'There is an unsupported layer'
    return scale_dict

def get_io_shift(extra_info,datapath_dict,config_dict,node_list,former_layer_dict):
    scale_info = extra_info['scale_info']
    datapath_bitwidth = config_dict['datapath_bitwidth']
    io_shift_dict = {}
    y_dict = {}
    for i in range(len(node_list)):
        if node_list[i].op_type == 'Constant':
            continue
        layer = node_list[i].name
        layer_dict = {}
        # output_max = max(abs(datapath_dict[layer]['layer_max']), abs(datapath_dict[layer]['layer_min']))
        # y_radix = getradix(output_max, datapath_bitwidth)
        y_radix = scale_info[layer]['y_radix'][0] 
        layer_dict['y_radix'] = y_radix
        y_dict[layer] = layer_dict['y_radix']
        if datapath_dict[layer]['op_type'] == 'Add':
            former_layer_list = former_layer_dict[layer]
            x0 = former_layer_list[0]
            x1 = former_layer_list[1]
            layer_dict['x0_radix'] = y_dict[x0]
            layer_dict['x1_radix'] = y_dict[x1]
            input_shift = abs(layer_dict['x0_radix']-layer_dict['x1_radix'])
            output_shift = y_radix-min(layer_dict['x0_radix'], layer_dict['x1_radix'])
            layer_dict['input_shift'] = input_shift
            layer_dict['output_shift'] = output_shift
            io_shift_dict[layer] = layer_dict
        elif datapath_dict[layer]['op_type'] == 'Concat':
            former_layer_list = former_layer_dict[layer]
            x0 = former_layer_list[0]
            x1 = former_layer_list[1]
            layer_dict['x0_radix'] = y_dict[x0]
            layer_dict['x1_radix'] = y_dict[x1]
            input_shift = abs(layer_dict['x0_radix']-layer_dict['x1_radix'])
            layer_dict['input_shift'] = input_shift
            io_shift_dict[layer] = layer_dict
    return io_shift_dict



def update_conv_weight_initializer(model, weight, weight_name, scale):
    '''update weight value of conv node, including kernel and bias
    '''
    if len(weight.shape) == 4:
        if isinstance(scale,list):
            channel_num = weight.shape[0]
            assert (channel_num == len(scale)) , "check whether the channel number is right"
            weight_up = [weight[c, :, :, :] * scale[c] for c in range(channel_num)]
            weight_up = np.array(weight_up).flatten()
        else:
            weight_up = [ddd for aaa in weight for bbb in aaa for ccc in bbb for ddd in ccc]
            weight_up = [w * scale for w in weight_up]

    else:
        assert len(weight.shape) == 1, 'error shape of weight'
        if isinstance(scale,list):
            assert len(weight) == len(scale), "check whether the channel number is right"
            weight_up = [weight[i] * scale[i] for i in range(len(weight))]
        else:
            weight_up = [bias * scale for bias in weight]
    
    OnnxEditor.rm_initializer(model, weight_name)
    OnnxEditor.add_initializer(model, weight_name, weight_up, weight.shape)
    

def update_bn_weight_initializer(model, bn_node, scale, channel_num):
    _,gamma_name,_,mean_name,_ = bn_node.input
    for ini in model.graph.initializer[:]:
        if ini.name == gamma_name:
            weight = onnx.numpy_helper.to_array(ini)
            shape = weight.shape
            weight = weight.flatten().tolist()
            if type(scale) == float:
                weight_up = [item / scale for item in weight]
            elif type(scale) == list:
                weight_up = [weight[i] / scale[i] for i in range(channel_num)]
            
            OnnxEditor.rm_initializer(model, ini.name)
            OnnxEditor.add_initializer(model, ini.name, weight_up, shape)
        
        elif ini.name == mean_name:
            weight = onnx.numpy_helper.to_array(ini)
            shape = weight.shape
            weight = weight.flatten().tolist()
            if type(scale) == float:
                weight_up = [item * scale for item in weight]
            elif type(scale) == list:
                weight_up = [weight[i] * scale[i] for i in range(channel_num)]
            
            OnnxEditor.rm_initializer(model, ini.name)
            OnnxEditor.add_initializer(model, ini.name, weight_up, shape)

def concat_relu_change_radix(datapath_dict_update,scale_dict,next_layer_dict,node_list,former_layer_dict,last_node_list):
    '''
    The purpose of this function is to adjust the relu(or leakyrelu) output-result's radix
     and make the concat node output-result's radix be same with relu(or leakyrelu) output-result's radix
    '''
    for i in range(len(node_list)):
        if node_list[i].op_type in ["LeakyRelu","Relu"]:       
            this_layer_name = node_list[i].name
            if this_layer_name in last_node_list :
                continue
            former_layer_list = former_layer_dict[this_layer_name]
            next_layer_list = next_layer_dict[this_layer_name]
            for next_layer_name in next_layer_list:
                if datapath_dict_update[next_layer_name]["op_type"] == "Concat":
                    next_layer_radix = scale_dict[next_layer_name]["y_radix"][0]
                    this_layer_radix = scale_dict[this_layer_name]["y_radix"][0]
                    if  this_layer_radix != next_layer_radix:
                        former_layer_name = former_layer_list[0]
                        former_layer_radix = scale_dict[former_layer_name]["y_radix"][0]
                        assert former_layer_radix > next_layer_radix,"concat_relu_change_radix error"
                        update_y_radix_list = [next_layer_radix for _ in range(len(scale_dict[former_layer_name]['x_radix']))]
                        scale_dict[former_layer_name]['y_radix'] = update_y_radix_list
                        scale_dict[this_layer_name]['y_radix'] = update_y_radix_list
                        scale_dict[this_layer_name]['x_radix'] = update_y_radix_list
                        former_layer_list_ = former_layer_dict[next_layer_name]
                        scale_dict[next_layer_name]['x0_radix'] = scale_dict[former_layer_list_[0]]["y_radix"]
                        scale_dict[next_layer_name]['x1_radix'] = scale_dict[former_layer_list_[1]]["y_radix"]


    return scale_dict


def add_change_radix(datapath_dict_update,scale_dict,next_layer_dict,node_list,former_layer_dict,last_node_list):
    '''
    The purpose of this function is to adjust the add's input radix to abs(x1_radix - x0_radix) <= 2
    '''
    for i in range(len(node_list)):
        if node_list[i].op_type == "Add":       
            this_layer_name = node_list[i].name
            x = former_layer_dict[this_layer_name]
            x0 = x[0]
            x0_radix = scale_dict[x0]["y_radix"][0]
            x1 = x[1]
            x1_radix = scale_dict[x1]["y_radix"][0]
            
            if np.abs(x1_radix - x0_radix) > 2 :
                if datapath_dict_update[x0]["op_type"] != "Add" and datapath_dict_update[x1]["op_type"] != "Add" :
                    if scale_dict[x0]["y_radix"][0] > scale_dict[x1]["y_radix"][0] :
                        scale_dict[x0]["y_radix"][0] = scale_dict[x1]["y_radix"][0] + 2
                        update_y_radix_list = [scale_dict[x0]["y_radix"][0] for _ in range(len(scale_dict[x0]['x_radix']))]
                        scale_dict[x0]['y_radix'] = update_y_radix_list
                    else :
                        scale_dict[x1]["y_radix"][0] = scale_dict[x0]["y_radix"][0] + 2
                        update_y_radix_list = [scale_dict[x1]["y_radix"][0] for _ in range(len(scale_dict[x1]['x_radix']))]
                        scale_dict[x1]['y_radix'] = update_y_radix_list
                else :
                    if datapath_dict_update[x0]["op_type"] != "Add" :
                        if scale_dict[x0]["y_radix"][0] > scale_dict[x1]["y_radix"][0] :
                            scale_dict[x0]["y_radix"][0] = scale_dict[x1]["y_radix"][0] + 2
                        else :
                            scale_dict[x0]["y_radix"][0] = scale_dict[x1]["y_radix"][0] - 2
                        update_y_radix_list = [scale_dict[x0]["y_radix"][0] for _ in range(len(scale_dict[x0]['x_radix']))]
                        scale_dict[x0]['y_radix'] = update_y_radix_list  
                    else :
                        if scale_dict[x0]["y_radix"][0] > scale_dict[x1]["y_radix"][0] :
                            scale_dict[x1]["y_radix"][0] = scale_dict[x0]["y_radix"][0] - 2
                        else :
                            scale_dict[x1]["y_radix"][0] = scale_dict[x0]["y_radix"][0] + 2
                        update_y_radix_list = [scale_dict[x1]["y_radix"][0] for _ in range(len(scale_dict[x1]['x_radix']))]
                        scale_dict[x1]['y_radix'] = update_y_radix_list 
                scale_dict[this_layer_name]["x0_radix"]=scale_dict[x0]['y_radix']
                scale_dict[this_layer_name]["x1_radix"]=scale_dict[x1]['y_radix']    
            else :
                pass


    return scale_dict


def concat_lut_change_radix(datapath_dict_update,scale_dict,next_layer_dict,node_list,former_layer_dict,last_node_list):
    '''
    The purpose of this function is to adjust the concat's input and output radix considering lut's radix
    ,when concat input change is true
    '''
    for i in range(len(node_list)):
        if node_list[i].op_type in ["Silu","Mish"]:       
            this_layer_name = node_list[i].name
            if this_layer_name in last_node_list :
                continue
            
            next_layer_list = next_layer_dict[this_layer_name]
            for next_layer_name in next_layer_list:
                if datapath_dict_update[next_layer_name]["op_type"] == "Concat":
                    former_layer_list = former_layer_dict[next_layer_name]
                    lenth = len(former_layer_list)
                    if lenth != 2:
                        continue
                    radix_list = []

                    if scale_dict[former_layer_list[0]]["y_radix"][0]>scale_dict[former_layer_list[1]]["y_radix"][0]:
                        new_radix = scale_dict[former_layer_list[1]]["y_radix"][0]
                        update_y_radix_list = [new_radix for _ in range(len(scale_dict[former_layer_list[0]]['y_radix']))]
                        scale_dict[former_layer_list[0]]["y_radix"] = update_y_radix_list
                        scale_dict[next_layer_name]['x0_radix'] = scale_dict[former_layer_list[0]]["y_radix"]
                        scale_dict[next_layer_name]['x1_radix'] = scale_dict[former_layer_list[1]]["y_radix"]
                        
                        y_radix_list = [new_radix for _ in range(len(scale_dict[next_layer_name]['x0_radix'])+len(scale_dict[next_layer_name]['x1_radix']))]
                        scale_dict[next_layer_name]['y_radix'] = y_radix_list
                        next_next_layer_list = next_layer_dict[next_layer_name]
                        for i in range(len(next_next_layer_list)):
                            next_next_layer_name = next_next_layer_list[i]
                            scale_dict[next_next_layer_name]['x_radix'] = scale_dict[next_layer_name]['y_radix']
                    else :
                        new_radix = scale_dict[former_layer_list[0]]["y_radix"][0]
                        update_y_radix_list = [new_radix for _ in range(len(scale_dict[former_layer_list[1]]['y_radix']))]
                        scale_dict[former_layer_list[1]]["y_radix"] = update_y_radix_list
                        scale_dict[next_layer_name]['x0_radix'] = scale_dict[former_layer_list[0]]["y_radix"]
                        scale_dict[next_layer_name]['x1_radix'] = scale_dict[former_layer_list[1]]["y_radix"]  

                        y_radix_list = [new_radix for _ in range(len(scale_dict[next_layer_name]['x0_radix'])+len(scale_dict[next_layer_name]['x1_radix']))]
                        scale_dict[next_layer_name]['y_radix'] = y_radix_list
                        next_next_layer_list = next_layer_dict[next_layer_name]
                        for i in range(len(next_next_layer_list)):
                            next_next_layer_name = next_next_layer_list[i]
                            scale_dict[next_next_layer_name]['x_radix'] = scale_dict[next_layer_name]['y_radix']               

    return scale_dict




def re_compute_conv_radix(analysis_mode,datapath_dict,scale_dict,next_layer_dict,node_list,former_layer_dict,last_node_list,weight_analy_dict,config_dict,limit_shift,shift_upper_limit,shift_lower_limit):
    '''
    The purpose of this function is to recompute the conv's kernel psum bias radix 
    becase the y radix is changed in other function
    '''
    for i in range(len(node_list)):
        layer = node_list[i].name
        if node_list[i].op_type == 'Constant':
                continue
        if datapath_dict[layer]["op_type"] == "Conv":
            if analysis_mode == 'per_channel_weight':
                y_radix_list = scale_dict[layer]['y_radix']
                layer_dict = scale_dict[layer]
                weight_max_list = list(map(abs, weight_analy_dict[layer]['group_max_list']))
                weight_min_list = list(map(abs, weight_analy_dict[layer]['group_min_list']))
                w_kernel_radix_list = []
                psum_radix_list = []
                kernel_bitwidth = config_dict['conv_bitwidth']['kernel']
                if layer in config_dict:
                    kernel_bitwidth = config_dict[layer]["kernel"]
                    print("!!!layer {}: specific weight kernel bitwidth config: {}, bias bitwidth: {}".format(layer, config_dict[layer]["kernel"], config_dict[layer]["bias"]))

                for j in range(len(weight_max_list)):
                    if weight_max_list[j] > weight_min_list[j]:
                        w_kernel_radix = getradix(weight_max_list[j], kernel_bitwidth)
                    else:
                        w_kernel_radix = getradix(weight_min_list[j], kernel_bitwidth)
                    #w_kernel_radix_list.append(w_kernel_radix)
                    if limit_shift == True:
                        if layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] > int(shift_upper_limit):
                            w_kernel_radix = int(shift_upper_limit) + y_radix_list[j] - layer_dict['x_radix'][0]
                        assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] <= int(shift_upper_limit),"datapath shift out of range"
                        assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix_list[j] >= int(shift_lower_limit),"datapath shift out of range"
                    w_kernel_radix_list.append(w_kernel_radix)
                    if layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] > 15:
                        coarse_shift = 4*math.floor((layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] - 16)/4)+4
                    elif layer_dict['x_radix'][0] + w_kernel_radix_list[j] - y_radix_list[j] < 0:
                        coarse_shift = -4
                    else:
                        coarse_shift = 0
                    psum_radix = layer_dict['x_radix'][0] + w_kernel_radix_list[j] - coarse_shift
                    psum_radix_list.append(psum_radix)
                layer_dict['kernel_weight_radix'] = w_kernel_radix_list
                layer_dict['bias_radix'] = y_radix_list
                layer_dict['psum_radix'] = psum_radix_list
                scale_dict[layer] = layer_dict
            else :
                layer_dict = scale_dict[layer]
                kernel_num = int(weight_analy_dict[layer]['kernel_num'])
                kernel_max = weight_analy_dict[layer]['kernel_max']
                kernel_min = weight_analy_dict[layer]['kernel_min']
                weight_max = max(abs(kernel_max), abs(kernel_min))
                kernel_bitwidth = config_dict['conv_bitwidth']['kernel']
                if layer in config_dict:
                    kernel_bitwidth = config_dict[layer]["kernel"]
                    print("!!!layer {}: specific weight kernel bitwidth config: {}, bias bitwidth: {}".format(layer, config_dict[layer]["kernel"], config_dict[layer]["bias"]))

                w_kernel_radix = getradix(weight_max, kernel_bitwidth)
                y_radix_list = scale_dict[layer]['y_radix']
                y_radix = y_radix_list[0]
                
                if limit_shift == True:
                    if layer_dict['x_radix'][0] + w_kernel_radix - y_radix > int(shift_upper_limit):
                        w_kernel_radix = int(shift_upper_limit) + y_radix - layer_dict['x_radix'][0]
                    assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix <= int(shift_upper_limit),"datapath shift out of range"
                    assert layer_dict['x_radix'][0] + w_kernel_radix - y_radix >= int(shift_lower_limit),"datapath shift out of range"
                if layer_dict['x_radix'][0] + w_kernel_radix - y_radix > 15:
                    coarse_shift = 4*math.floor((layer_dict['x_radix'][0] + w_kernel_radix - y_radix- 16)/4)+4
                elif layer_dict['x_radix'][0] + w_kernel_radix - y_radix < 0:
                    coarse_shift = -4
                else:
                    coarse_shift = 0
                psum_radix = layer_dict['x_radix'][0] + w_kernel_radix - coarse_shift

                w_kernel_radix_list = [w_kernel_radix for _ in range(kernel_num)]
                psum_radix_list = [psum_radix for _ in range(kernel_num)]

                layer_dict['kernel_weight_radix'] = w_kernel_radix_list
                layer_dict['bias_radix'] = y_radix_list
                layer_dict['psum_radix'] = psum_radix_list
                scale_dict[layer] = layer_dict

        
    return scale_dict
