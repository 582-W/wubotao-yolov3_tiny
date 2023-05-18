import os
import numpy as np 
import math 
import onnx
import argparse
import logging 
import json
import shutil
import logging
from tqdm import tqdm


def evaluate(float_folder_path, dfp_folder_path, output_result_path, onnx_file_path, log_level = logging.INFO, log_file=""):
    """
    Args:
        float_folder_path (string): absolute path of the float_inference result folder.
        dfp_folder_path(string): absolute path of the dfp_inference result folder.
        output_result_path(string): absolute path of the all results folder.
    Returns:
        No output.
    """
    
    if log_file != "":
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(format='%(message)s', level=log_level, filename=log_file)
    else:
        logging.basicConfig(format='%(message)s', level=log_level)
   
    layer_name_list = []

    #Get image_name_list
    image_name_list = [] 
    for filename in os.listdir(float_folder_path):

        if not os.path.exists(float_folder_path):
            logging.error("{} not exists, it might be there's no float_inference result".format(float_folder_path))
            continue
        layer_name = filename.split("layer_")[-1].split("_img_")[0]
        index1 = filename.index('_img_')
        index2 = filename.index('.npy')
        image_name = filename[index1 + 1 : index2]
        logging.info("image_name : {}".format(image_name))
        if image_name not in image_name_list:
            image_name_list.append(image_name) 
        if layer_name not in layer_name_list:
            layer_name_list.append(layer_name)  

    logging.info(image_name_list) 
    #Caculate each layer's psnr 
    evaluate_out = {}
    for i in tqdm(range(len(layer_name_list))):

        _layer_name = layer_name_list[i]
        float_result = []
        dfp_result = []
        evaluate_out[_layer_name] = {}

        for j in range(len(image_name_list)):
            file_name = 'layer_' + _layer_name + '_' + image_name_list [j] + '.npy'
            if not os.path.exists(dfp_folder_path):
                logging.error("{} not exists, it might be there's no dfp_inference float result".format(float_folder_path))
                continue
         
            float_tem = np.load(os.path.join(float_folder_path, file_name)).flatten()
            dfp_tem = np.load(os.path.join(dfp_folder_path, file_name)).flatten()
            float_result = np.hstack((float_result, float_tem))
            dfp_result = np.hstack((dfp_result, dfp_tem))

        difference = dfp_result - float_result
        psnr = 20 * math.log10(np.max(np.abs(float_result)) / np.std(np.abs(difference)))
        logging.info('PSNR of current layer ' + _layer_name + ' is : {}'.format(psnr))
        snr = 20 * math.log10(np.mean(np.abs(float_result)) / np.std(np.abs(difference)))
        logging.info('SNR of current layer ' + _layer_name + ' is : {}'.format(snr))

        evaluate_out[_layer_name]['PSNR'] = psnr
        evaluate_out[_layer_name]['SNR'] = snr

    json_str = json.dumps(evaluate_out, sort_keys=True, indent=4)

    
    if not os.path.exists(output_result_path):
        os.makedirs(output_result_path)

    with open(os.path.join(output_result_path, 'evaluation.json'), 'w+') as json_file:
        json_file.write(json_str)

    ### compare results of float inference and fp inference
    results_compare = {}
    model = onnx.load(onnx_file_path)
    nodes = model.graph.node
    dict_node = {}
    dict_input = {}
    dict_output = {}
    for i in range(len(nodes)):
        dict_node[nodes[i].name] = i
    for i in range(len(nodes)):
        for j in range(len(nodes[i].input)):
            dict_input[nodes[i].input[j]] = i
    for i in range(len(nodes)):
        assert len(nodes[i].output)==1, 'there are multi-outputs in {}'.format(nodes[i].name)
        dict_output[nodes[i].output[0]] = i
    
    for item in os.listdir(float_folder_path):
        ## get node info of original model
        name_node = item.split('_img_')[0].replace('layer_','')
        name_img = item.split('_img_')[1].replace('.npy','')
        assert name_node in dict_node.keys(), '{} not found in model node'.format(name_node)
        idx_node = dict_node[name_node]
        output_ = nodes[idx_node].output
        ## look for corresponding node in updated model
        if output_[0] in dict_input.keys():
            idx_node = dict_input[output_[0]]
            if nodes[idx_node].op_type == 'BatchNormalization' and \
                ('layer_' + nodes[idx_node].name + '_img_' + name_img + '.npy') not in os.listdir(float_folder_path):
                item_dfp = 'layer_' + nodes[idx_node].name + '_img_' + name_img + '.npy'
            else:
                item_dfp = item
        else:
            item_dfp = item

        res_float = np.load(os.path.join(float_folder_path,item))            
        res_dfp = np.load(os.path.join(dfp_folder_path,item_dfp))
        diff_max = (abs(res_float-res_dfp)).max()
        diff_mean = (abs(res_float-res_dfp)).mean()
        diff_max_float = float(res_float[abs(res_float-res_dfp)==diff_max][0])                 
        name_layer = item.split('.')[0]
        results_compare[name_layer] = {}
        results_compare[name_layer]['diff_max'] = str(diff_max) + '/' + str(diff_max_float)
        results_compare[name_layer]['diff_mean'] = diff_mean

    json_str = json.dumps(results_compare, sort_keys=True, indent=4)
    with open(os.path.join(output_result_path, 'results_compare.json'), 'w+') as json_file:
        json_file.write(json_str)
