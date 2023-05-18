from onnxparser.parser import OnnxParser,AttributeExtractor
from .common import DataProcessor
from .functions import functions_fp as fpFunctions
from .functions import functions_fl as floatFunctions
from .functions import custom_fl
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import os
import json
#import keras
#from keras.models import *
import math
import shutil
import time
import logging
from tqdm import tqdm
from utils import utils
#from preprocessing.img_to_txt import ImgPreprocessor
from utils.utils import convert2float64

class FixedPointInferencer(OnnxParser):
    """FixedPointInferener class
    """

    def __init__(self, model_path, img_folder_path, input_case_folder, list_of_customized_name=None, config_file_path="../config/hardware.json"):
        self.model_path = model_path
        onnxmodel = onnx.load_model(model_path)
        super().__init__(onnxmodel)
        self.extractor_a = AttributeExtractor()
        self.processor = DataProcessor()
        self.img_folder_path = img_folder_path
        self.input_case_folder = input_case_folder
        self.list_of_customized_name = list_of_customized_name
        f = open(config_file_path, encoding = 'utf-8')
        self.config_dict = json.load(f)
        self.kernel_bitwidth = self.config_dict['conv_bitwidth']['kernel']
        self.bias_bitwidth = self.config_dict['conv_bitwidth']['bias']
        self.working_bitwidth = self.config_dict['working_bitwidth']
        self.datapath_bitwidth = self.config_dict['datapath_bitwidth']
        self.leaky_relu_alpha_bitwidth = self.config_dict['leaky_relu_alpha']
        self.bn_bitwidth = self.config_dict["bn_bitwidth"]
        self.average_pool_radix = self.config_dict["average_pool_radix"]
        self.SHIFT_LIST = [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]

    def _get_dfp_input(self, img_preprocess_method, image_npy_folder, float_folder_path, int_folder_path): 
        
        """get the dfp input function
        Args:
            img_preprocess_method (string) : the method for img_preprocess
            image_npy_folder (string): the path of image_npy_folder
            float_folder_path (string): the path of float result folder
            int_folder_path (string): the path of int result folder
            
        Returns:
            img_name_list (list): the list of img name
        """
        
        img_path = fpFunctions.read_directory(image_npy_folder)
        logging.info('total image number:{}'.format(len(img_path)))
        img_name_list = []
        input_int_floder = '{}/input_int'.format(image_npy_folder)
        os.mkdir(input_int_floder)
        for img in img_path:
            img_name = os.path.splitext(os.path.split(img)[1])[0]
            img_name_list.append(img_name)
            shutil.copy(img, '{}/layer_{}_img_{}.npy'.format(float_folder_path, 'input', img_name))
            temp_res = np.load(img)
            #data_max = ImgPreprocessor().get_datapath_input(img_preprocess_method)
            data_max_path = os.path.join(self.input_case_folder, 'data_max.txt')
            data_max_file = open(data_max_path, 'r')
            data_max = data_max_file.readline().replace('\n','')
            data_max = float(data_max)
            temp_res_radix = fpFunctions.getradix(data_max, self.datapath_bitwidth)
            temp_res = np.round(temp_res * 2 ** temp_res_radix)
            np.save('{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, 'input', img_name, temp_res_radix), temp_res)
            np.save('{}/{}.npy'.format(input_int_floder, img_name), temp_res)
        
        return img_name_list

    def _fix_point_inference_per_layer(self, img_name_list, output_folder_path, float_folder_path, int_folder_path, update_json_dict, output_per_layer, 
                log_level = logging.INFO):
        """fix_point_inference_per_layer function
        Args:
            img_name_list (list): the list of img_name
            output_folder_path (string): the path of output folder
            float_folder_path (string): the path of float result folder
            int_folder_path (string): the path of int result folder
            update_json_dict (dict): the update_json_dict
            output_per_layer (bool): options for output_per_layer
            log_level : log_level for log info 
            
        Returns:
            
        """
        
        dict_overflow = {}

        for i in tqdm(range(len(self.node_list))):
        
            cur_node = self.node_list[i]
            node_name = cur_node.name

            if cur_node.op_type == "Constant": 
                logging.warning("=================================================")
                logging.warning("node id: {}/{} layer_name: {} op_type: {}".format(i, len(self.model.graph.node), node_name, cur_node.op_type))
                continue
            else:
                logging.warning("=================================================")
                logging.warning("node id: {}/{} layer_name: {} op_type: {}".format(i, len(self.model.graph.node), node_name, cur_node.op_type))
                
                dict_overflow[node_name] = {}

                for img_name in img_name_list:
                    res_path = ""
                    float_res_path = ""
                    temp_res_list = []
                    temp_res_radix_list = []

                    if self.node_list[i].name in self.list_of_customized_name:
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]

                        temp_res = temp_res * 2 ** (-temp_res_radix)
                        temp_res = custom_fl.custom_node_fl(temp_res, self.node_list[i].name)   ### the custom function call
                        res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)

                        float_res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, node_name, img_name)
                        np.save(float_res_path, temp_res)
                        int_temp_res = np.round(temp_res * 2 ** (temp_res_radix))
                        res_path = '{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, node_name, img_name, temp_res_radix)
                        np.save(res_path, int_temp_res)

                    elif cur_node.op_type == "Conv":
                        dilation, group, pad, stride = self.extractor_a.get_info_conv(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list) == 3:
                            temp_res, w_kernel, w_bias = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                            [dim_bias] = w_bias.shape
                            w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))
                        elif len(temp_res_list) == 2:
                            temp_res, w_kernel = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                            w_bias = []
                        else:
                            assert False, 'input number of this node is wrong!'
                        if w_bias != []:
                            w_bias = w_bias.astype(np.float64)
                        w_kernel = w_kernel.astype(np.float64)
                        kernel_overflow_time = 0
                        kernel_overflow = 0
                        kernel_overflow_info = {}
                        if node_name in self.config_dict: ## you can set bitwidth for specific conv node in config file
                            kernel_overflow = 2 ** (self.config_dict[node_name]["kernel"] - 1) - 1
                        else:
                            kernel_overflow = 2 ** (self.kernel_bitwidth - 1) - 1

                        # get radix from update_json_dict
                        x_radix = update_json_dict['scale_info'][node_name]['x_radix'][0]
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        w_kernel_radix = update_json_dict['scale_info'][node_name]['kernel_weight_radix'][0]
                        psum_radix = update_json_dict['scale_info'][node_name]['psum_radix'][0]
                        if node_name in self.config_dict:
                            weight_max = max(abs(update_json_dict['weight_analysis_info'][node_name]['kernel_max']), 
                                        abs(update_json_dict['weight_analysis_info'][node_name]['kernel_min']))
                            w_kernel_radix = fpFunctions.getradix(weight_max, self.config_dict[node_name]["kernel"])
                            logging.critical("!!!layer {}: specific weight kernel bitwidth config: {}, bias bitwidth: {}".format(node_name, self.config_dict[node_name]["kernel"], self.config_dict[node_name]["bias"]))
                        
                        w_bias_radix = y_radix
                        w_kernel = np.round(w_kernel * 2 ** w_kernel_radix)
                        w_bias = np.round(w_bias * 2 ** w_bias_radix)
                        
                        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]

                        if kernel_overflow_time > 0:
                            (C, C_prev, f, f) = w_kernel.shape
                            logging.warning('kernel overflow: {}/{}'.format(kernel_overflow_time, C * C_prev * f * f ))
                            kernel_overflow_info["overflow_time"] = kernel_overflow_time
                            kernel_overflow_info["data_cnt"] = C * C_prev * f * f
                            kernel_overflow_info["overflow_th"] = kernel_overflow
                            kernel_overflow_info["overflow_rate"] = kernel_overflow_time / (C * C_prev * f * f)
                            kernel_overflow_info["data_max"] = abs(w_kernel).max()

                        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
                        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1

                        w_kernel = floatFunctions.update_w_kernel_for_dilation_conv(w_kernel, dilation)
                        output_bitwidth = update_json_dict['scale_info'][node_name]['output_bitwidth']
                        if group != 1:
                            ### depthwise conv
                            temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.depthwise_conv(temp_res, temp_res_radix, w_kernel, w_kernel_radix, w_bias, group, y_radix, self.working_bitwidth, output_bitwidth, self.bias_bitwidth, stride, pad, dilation)
                        else:
                            ### conv 
                            if node_name in self.config_dict:
                                temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.conv(temp_res, temp_res_radix, w_kernel, w_kernel_radix, w_bias, y_radix, self.working_bitwidth, output_bitwidth, self.config_dict[node_name]["bias"], stride, pad, dilation)
                            else:
                                temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.conv(temp_res, temp_res_radix, w_kernel, w_kernel_radix, w_bias, y_radix, self.working_bitwidth, output_bitwidth, self.bias_bitwidth, stride, pad, dilation)
                        
                        dict_overflow[node_name]["kernel_overflow"] = kernel_overflow_info
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                        if temp_res_radix != y_radix:
                            logging.error('conv radix error!')
                        assert temp_res_radix == y_radix, 'conv radix error!'

                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "BatchNormalization":
                        epsilon, momentum = self.extractor_a.get_info_bn(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 5, 'input number of this node is wrong!'
                        temp_res, gamma, beta, mean, var = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        gamma = gamma.astype(np.float64)                       
                        beta = beta.astype(np.float64)
                        mean = mean.astype(np.float64)
                        var = var.astype(np.float64)    
                        
                        A_list, B_list = [], []
                        for bn_channel in range(len(gamma)):
                            A_list.append(gamma[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
                            B_list.append(beta[bn_channel] - gamma[bn_channel]*mean[bn_channel] / np.sqrt(var[bn_channel] + epsilon))

                        a_radix = update_json_dict['scale_info'][node_name]['bn_a_radix'][0]
                        x_radix = update_json_dict['scale_info'][node_name]['x_radix'][0]
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        b_radix = y_radix                   
                        
                        A_list = np.round(np.array(A_list) * 2 ** a_radix)
                        B_list = np.round(np.array(B_list) * 2 ** y_radix)

                        A_overflow = 2 ** (self.bn_bitwidth - 1) - 1
                        B_overflow = 2 ** (self.datapath_bitwidth - 1) -1
                        A_list = np.array(A_list)
                        B_list = np.array(B_list)
                        A_list[np.where(A_list > A_overflow)] = A_overflow
                        A_list[np.where(A_list < -A_overflow - 1)] = -A_overflow - 1
                        B_list[np.where(B_list > B_overflow)] = B_overflow
                        B_list[np.where(B_list < -B_overflow - 1)] = -B_overflow - 1
                        output_bitwidth = update_json_dict['scale_info'][node_name]['output_bitwidth']
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.BatchNormalization(temp_res, temp_res_radix, A_list, a_radix, B_list, y_radix, self.working_bitwidth, output_bitwidth)

                        if temp_res_radix != y_radix:
                            logging.error('bn radix error!')
                        assert temp_res_radix == y_radix, 'bn radix error!'
                        
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "L2Normalization":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, gamma_list = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]

                        temp_res = temp_res * 2 ** (-temp_res_radix)
                        float_temp_res = floatFunctions.l2_norm(temp_res, gamma_list)
                        float_res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, node_name, img_name)
                        np.save(float_res_path, float_temp_res)
                        ouput_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                        abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))

                        y_radix = fpFunctions.getradix(ouput_max, self.datapath_bitwidth)
                        temp_res_radix = y_radix
                        int_temp_res = np.round(float_temp_res * 2 ** (temp_res_radix))
                        res_path = '{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, node_name, img_name, temp_res_radix)
                        np.save(res_path, int_temp_res)
                    
                    elif cur_node.op_type == "PRelu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, slope = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]

                        slope_max = max(abs(update_json_dict['weight_analysis_info'][node_name]['slope_max']), abs(update_json_dict['weight_analysis_info'][node_name]['slope_min']))
                        slope_radix = fpFunctions.getradix(slope_max, self.leaky_relu_alpha_bitwidth)
                        assert(slope_radix == 18), "slope_radix must be 18"
                        slope = np.round(slope * 2 ** slope_radix)
                        
                        y_radix = temp_res_radix
                        temp_res, calc_overflow_info, datapath_overflow_info = fpFunctions.PRelu(temp_res, temp_res_radix, slope, slope_radix, y_radix, self.working_bitwidth, self.datapath_bitwidth)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "MaxPool":
                        kernel_size, pad, stride = self.extractor_a.get_info_maxpool(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.maxpool(temp_res, kernel_size, pad, stride)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "AveragePool":
                        kernel_size, pad, stride = self.extractor_a.get_info_averagepool(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.averagepool(temp_res, self.average_pool_radix, kernel_size, pad, stride)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Clip":
                        # clip_max, clip_min = self.extractor_a.get_info_clip(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list)==3 :
                            clip_min = temp_res_list[1]
                            clip_max = temp_res_list[2]
                        else :
                            clip_max, clip_min = self.extractor_a.get_info_clip(self.node_list[i]) 
                        # assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        clip_max, clip_min = np.round(clip_max * (2 ** temp_res_radix)), np.round(clip_min * (2 ** temp_res_radix))
                        temp_res = fpFunctions.clip(temp_res, clip_min, clip_max)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Relu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.relu(temp_res)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "Flatten":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.flatten(temp_res)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "LeakyRelu":
                        alpha = self.extractor_a.get_info_leakyrelu(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]

                        alpha_radix = fpFunctions.getradix(alpha, self.leaky_relu_alpha_bitwidth)
                        assert(alpha_radix == 18), "alpha_radix must be 18"
                        alpha = round(alpha * 2 ** alpha_radix)
                        y_radix= update_json_dict['scale_info'][node_name]['y_radix'][0]
                        x_radix= update_json_dict['scale_info'][node_name]['x_radix'][0]
                        assert x_radix==y_radix
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.LeakyRelu(temp_res, temp_res_radix, alpha, alpha_radix, y_radix, self.working_bitwidth, self.datapath_bitwidth)                        
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "Upsample":
                        mode,scales = self.extractor_a.get_info_upsample(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list) == 2:
                            temp_res, scales = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                        elif len(temp_res_list) == 1:
                            temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        else:
                            assert False, 'input number of this node is wrong!'
                        input_radix = temp_res_radix
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                        abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        temp_res_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        
                        temp_res = fpFunctions.upsample(temp_res, scales, mode)
                        temp_res = np.floor(temp_res * 2 ** (temp_res_radix - input_radix))
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "Add":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                        abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)                            
                        temp_res, temp_res_radix, datapath_overflow_info = fpFunctions.add(temp_res_list, temp_res_radix_list, y_radix, self.datapath_bitwidth)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "Concat":
                        ##TODO: constant node input need to be supported, multi inputs(three or more) need to be suppoerted
                        axis = self.extractor_a.get_info_cancat(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                        abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix, datapath_overflow_info = fpFunctions.concatenate(temp_res_list, temp_res_radix_list, y_radix, axis, self.datapath_bitwidth)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)                            
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                    

                    elif cur_node.op_type == "Mish":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        
                        input_bitwidth = update_json_dict['scale_info'][node_name]['input_bitwidth']
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.mish(temp_res_list[0], temp_res_radix_list[0],input_bitwidth,
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
                   
                    elif self.node_list[i].op_type == "upsample_yolov4":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res,temp_res_radix = fpFunctions.upsample_yolov4(temp_res_list,temp_res_radix_list)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "Reshape":
                        allowzero = self.extractor_a.get_info_reshape(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, shape = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        if len(temp_res_radix_list) == 2:
                            shape_radix = temp_res_radix_list[1]
                        elif len(temp_res_radix_list) == 1:
                            shape_radix = 0
                        else:
                            assert False, "Unsupport reshape node type"
                        temp_res = fpFunctions.reshape(temp_res, shape, shape_radix, allowzero)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,
                                                            img_name, temp_res_radix)

                    elif cur_node.op_type == "Expand":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, shape = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        temp_res = fpFunctions.expand(temp_res, shape)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Silu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'

                        input_bitwidth = update_json_dict['scale_info'][node_name]['input_bitwidth']
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.silu(temp_res_list[0], temp_res_radix_list[0],input_bitwidth,
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
                   
                    elif cur_node.op_type == "Softplus":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix ,datapath_overflow_info = fpFunctions.softplus(temp_res_list[0], temp_res_radix_list[0],y_radix, self.datapath_bitwidth)
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Tanh":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix,datapath_overflow_info = fpFunctions.tanh(temp_res_list[0], temp_res_radix_list[0], y_radix,
                                                                    self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Sigmoid":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.sigmoid(temp_res_list[0], temp_res_radix_list[0],
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Mul":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.mul(temp_res_list, temp_res_radix_list, y_radix,self.working_bitwidth, self.datapath_bitwidth)

                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Resize":
                        coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode = \
                            self.extractor_a.get_info_resize(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 3, 'input number of this node is wrong!'

                        temp_res, roi, scales = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        sizes = None
                        temp_res = fpFunctions.resize(temp_res, roi, scales, sizes, \
                            coordinate_transformation_mode, cubic_coeff_a, exclude_outside,extrapolation_value, mode, nearest_mode)

                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    else:
                        logging.error("Unsupport op type: {}".format(self.node_list[i].op_type))
                        assert False, "Unsupport op type: {}".format(self.node_list[i].op_type)

            convert2float64(dict_overflow) ## we do this cause json file doesn't support float32 data type
            dict_overflow_json = json.dumps(dict_overflow, sort_keys=True, indent=4)
            dict_overflow_json_path = os.path.join(output_folder_path, 'overflow.json')
            with open(dict_overflow_json_path, 'w+') as json_file:
                json_file.write(dict_overflow_json)

    def _fix_point_inference_per_channel_weight(self, img_name_list, output_folder_path, float_folder_path, int_folder_path, update_json_dict, output_per_layer, 
                log_level = logging.INFO):
        
        """fix_point_inference_per_channel_weight function
        Args:
            img_name_list (list): the list of img_name
            output_folder_path (string): the path of output folder
            float_folder_path (string): the path of float result folder
            int_folder_path (string): the path of int result folder
            update_json_dict (dict): the update_json_dict
            output_per_layer (bool): options for output_per_layer
            log_level : log_level for log info 
            
        Returns:
            
        """

        dict_overflow = {}
        start_total_time = time.time()
        for i in tqdm(range(len(self.node_list))):
            node_name = self.node_list[i].name
            cur_node = self.node_list[i]
            if self.node_list[i].op_type == "Constant":
                logging.warning("=================================================")
                logging.warning("node id: {}/{} layer_name: {} op_type: {}".format(i, len(self.node_list), node_name, self.node_list[i].op_type))
                continue
            else:
                logging.warning("=================================================")
                logging.warning("node id: {}/{} layer_name: {} op_type: {}".format(i, len(self.node_list), node_name, self.node_list[i].op_type))
                dict_overflow[node_name] = {}
                for img_name in img_name_list:
                    start_time = time.time()
                    res_path = ""
                    float_res_path = ""
                    temp_res_list = []
                    temp_res_radix_list = []

                    if self.node_list[i].name in self.list_of_customized_name:
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]

                        temp_res = temp_res * 2 ** (-temp_res_radix)
                        temp_res = custom_fl.custom_node_fl(temp_res, self.node_list[i].name)   ### the custom function call
                        res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)

                        float_res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, node_name, img_name)
                        np.save(float_res_path, temp_res)
                        int_temp_res = np.round(temp_res * 2 ** (temp_res_radix))
                        res_path = '{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, node_name, img_name, temp_res_radix)
                        np.save(res_path, int_temp_res)

                    elif cur_node.op_type == "Conv":
                        dilation, group, pad, stride = self.extractor_a.get_info_conv(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list) == 3:
                            temp_res, w_kernel, w_bias = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                            [dim_bias] = w_bias.shape
                            w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))
                        elif len(temp_res_list) == 2:
                            temp_res, w_kernel = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                            (C,C_prev,f, f) = w_kernel.shape
                            w_bias = np.zeros((1,1,1,C))
                            # w_bias = []
                        else:
                            assert False, 'input number of this node is wrong!'
                        if w_bias != []:
                            w_bias = w_bias.astype(np.float64)
                        w_kernel = w_kernel.astype(np.float64)
                        (C,C_prev,f, f) = w_kernel.shape
                        kernel_overflow_time = 0
                        kernel_overflow = 0
                        if node_name in self.config_dict:
                            kernel_overflow = 2 ** (self.config_dict[node_name]["kernel"] - 1) - 1
                        else:
                            kernel_overflow = 2 ** (self.kernel_bitwidth - 1) - 1
                        
                        # get radix from update_json_dict
                        x_radix = update_json_dict['scale_info'][node_name]['x_radix'][0]
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        w_kernel_radix_list = update_json_dict['scale_info'][node_name]['kernel_weight_radix']
                        psum_radix_list = update_json_dict['scale_info'][node_name]['psum_radix']      
                        w_bias_radix = y_radix
                        for c in range(w_kernel.shape[0]):
                            w_kernel[c, :, :, :] = np.round(w_kernel[c, :, :, :] * 2.0 ** w_kernel_radix_list[c])

                        w_kernel_ori = w_kernel
                        w_kernel = floatFunctions.update_w_kernel_for_dilation_conv(w_kernel, dilation)
                        w_bias = np.round(w_bias * 2 ** w_bias_radix)
                        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]
                       
                        kernel_overflow_info = {}

                        if kernel_overflow_time > 0:
                            logging.warning('kernel overflow time: {}/{}'.format(kernel_overflow_time, f * f * C_prev * C))
                            kernel_overflow_info["overflow_time"] = kernel_overflow_time
                            kernel_overflow_info["data_cnt"] = f * f * C_prev * C
                            kernel_overflow_info["overflow_th"] = kernel_overflow
                            kernel_overflow_info["overflow_rate"] = kernel_overflow_time / (f * f * C_prev * C)
                            kernel_overflow_info["data_max"] = abs(w_kernel).max()

                        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
                        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1
                        output_bitwidth = update_json_dict['scale_info'][node_name]['output_bitwidth']
                        if group != 1:
                            temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.depth_wise_conv_by_channel(temp_res, temp_res_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, self.working_bitwidth, output_bitwidth, self.bias_bitwidth, group, stride, pad, dilation)
                        else:
                            temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.conv_by_channel(temp_res, temp_res_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, self.working_bitwidth, output_bitwidth, self.bias_bitwidth, stride, pad, dilation)
                            # for iee check, select here
                            # iee_version = 'n900'
                            # temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.conv_by_channel_iee(temp_res, temp_res_radix, w_kernel_ori, w_kernel_radix_list, w_bias, y_radix, self.working_bitwidth, self.datapath_bitwidth, self.bias_bitwidth, stride, pad, dilation=dilation, node_name=node_name, iee_version=iee_version)
                        
                        dict_overflow[node_name]["kernel_overflow"] = kernel_overflow_info
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        
                        if temp_res_radix != y_radix:
                            logging.error('conv radix error!')
                        assert temp_res_radix == y_radix, 'conv radix error!'
                        
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "BatchNormalization":
                        epsilon, momentum = self.extractor_a.get_info_bn(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 5, 'input number of this node is wrong!'
                        temp_res, gamma, beta, mean, var = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        gamma = gamma.astype(np.float64)                       
                        beta = beta.astype(np.float64)
                        mean = mean.astype(np.float64)
                        var = var.astype(np.float64)    
                        
                        A_list = []
                        B_list = []
                        for bn_channel in range(len(gamma)):
                            A_list.append(gamma[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
                            B_list.append(beta[bn_channel] - gamma[bn_channel]*mean[bn_channel] / np.sqrt(var[bn_channel] + epsilon))

                        a_radix_list = update_json_dict['scale_info'][node_name]['bn_a_radix']
                        x_radix = update_json_dict['scale_info'][node_name]['x_radix'][0]
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        b_radix = y_radix
                        for k in range(len(A_list)):
                            A_list[k] = np.round(A_list[k] * 2 ** a_radix_list[k])

                        B_list = np.round(np.array(B_list) * 2 ** y_radix)
                        A_overflow = 2 ** (self.bn_bitwidth - 1) - 1
                        B_overflow = 2 ** (self.datapath_bitwidth - 1) -1
                        A_list = np.array(A_list)
                        B_list = np.array(B_list)
                        A_list[np.where(A_list > A_overflow)] = A_overflow
                        A_list[np.where(A_list < -A_overflow - 1)] = -A_overflow - 1
                        B_list[np.where(B_list > B_overflow)] = B_overflow
                        B_list[np.where(B_list < -B_overflow - 1)] = -B_overflow - 1
                        output_bitwidth = update_json_dict['scale_info'][node_name]['output_bitwidth']
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.BN_per_channel(temp_res, temp_res_radix, A_list, a_radix_list, B_list, y_radix, self.working_bitwidth, output_bitwidth)
                        
                        if temp_res_radix != y_radix:
                            logging.error('bn radix error!')

                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "L2Normalization":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, gamma_list = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        temp_res = temp_res * 2 ** (-temp_res_radix)

                        float_temp_res = floatFunctions.l2_norm(temp_res, gamma_list)
                        float_res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, node_name, img_name)
                        np.save(float_res_path, float_temp_res)
                        ouput_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                        abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))

                        y_radix = fpFunctions.getradix(ouput_max, self.datapath_bitwidth)
                        temp_res_radix = y_radix
                        int_temp_res = np.round(float_temp_res * 2 ** (temp_res_radix))
                        res_path = '{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, node_name, img_name, temp_res_radix)
                        np.save(res_path, int_temp_res)

                    elif cur_node.op_type == "PRelu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, slope = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]

                        slope_max = max(abs(update_json_dict['weight_analysis_info'][node_name]['slope_max']), abs(update_json_dict['weight_analysis_info'][node_name]['slope_min']))
                        slope_radix = fpFunctions.getradix(slope_max, self.leaky_relu_alpha_bitwidth)
                        slope = np.round(slope * 2 ** slope_radix)
                        assert(slope_radix == 18), "alpha_radix must be 18"
                        y_radix = temp_res_radix
                        temp_res, calc_overflow_info, datapath_overflow_info = fpFunctions.PRelu(temp_res, temp_res_radix, slope, slope_radix, y_radix, self.working_bitwidth, self.datapath_bitwidth)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
 
                    elif cur_node.op_type == "MaxPool":
                        kernel_size, pad, stride = self.extractor_a.get_info_maxpool(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.maxpool(temp_res, kernel_size, pad, stride)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Relu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.relu(temp_res)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Flatten":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.flatten(temp_res)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "LeakyRelu":
                        alpha = self.extractor_a.get_info_leakyrelu(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        alpha_radix = fpFunctions.getradix(alpha, self.leaky_relu_alpha_bitwidth)
                        if alpha_radix!=18:
                            alpha_radix = 18
                            print("alpha_radix must be 18")
                        alpha = round(alpha * 2 ** alpha_radix)
                        x_radix = update_json_dict['scale_info'][node_name]['x_radix'][0]
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        assert x_radix==y_radix
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.LeakyRelu(temp_res, temp_res_radix, alpha, alpha_radix, y_radix, self.working_bitwidth, self.datapath_bitwidth)

                            
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "Upsample":
                        mode,scales = self.extractor_a.get_info_upsample(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list) == 2:
                            temp_res, scales = temp_res_list
                            temp_res_radix = temp_res_radix_list[0]
                        elif len(temp_res_list) == 1:
                            temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        else:
                            assert False, 'input number of this node is wrong!'
                        input_radix = temp_res_radix
                        
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                            abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        temp_res_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        
                        temp_res = fpFunctions.upsample(temp_res, scales, mode)
                        temp_res = np.floor(temp_res * 2 ** (temp_res_radix - input_radix))
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Add":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                            abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        
                        temp_res, temp_res_radix, datapath_overflow_info = fpFunctions.add(temp_res_list, temp_res_radix_list, y_radix, self.datapath_bitwidth)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
        
                    elif cur_node.op_type == "Concat":
                        ##TODO: constant node input need to be supported, multi inputs(three or more) need to be suppoerted
                        axis = self.extractor_a.get_info_cancat(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']), 
                                            abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix, datapath_overflow_info = fpFunctions.concatenate(temp_res_list, temp_res_radix_list, y_radix, axis, self.datapath_bitwidth)
                        
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info

                    elif cur_node.op_type == "Clip":
                        # clip_max, clip_min = self.extractor_a.get_info_clip(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        if len(temp_res_list)==3 :
                            clip_min = temp_res_list[1]
                            clip_max = temp_res_list[2]
                        else :
                            clip_max, clip_min = self.extractor_a.get_info_clip(cur_node)
                        # assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]

                        clip_max, clip_min = np.round(clip_max * (2 ** temp_res_radix)), np.round(clip_min * (2 ** temp_res_radix))
                        temp_res = fpFunctions.clip(temp_res, clip_min, clip_max)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)                    

                    elif cur_node.op_type == "AveragePool":
                        kernel_size, pad, stride = self.extractor_a.get_info_averagepool(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res, temp_res_radix = temp_res_list[0], temp_res_radix_list[0]
                        temp_res = fpFunctions.averagepool(temp_res, self.average_pool_radix, kernel_size, pad, stride)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "Mish":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        
                        input_bitwidth = update_json_dict['scale_info'][node_name]['input_bitwidth']
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.mish(temp_res_list[0], temp_res_radix_list[0],input_bitwidth,
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
                    
                    elif self.node_list[i].op_type == "upsample_yolov4":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res,temp_res_radix = fpFunctions.upsample_yolov4(temp_res_list,temp_res_radix_list)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix)

                    elif cur_node.op_type == "Reshape":
                        allowzero = self.extractor_a.get_info_reshape(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, shape = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        if len(temp_res_radix_list) == 2:
                            shape_radix = temp_res_radix_list[1]
                        elif len(temp_res_radix_list) == 1:
                            shape_radix = 0
                        else:
                            assert False, "Unsupport reshape node type"
                        temp_res = fpFunctions.reshape(temp_res, shape, shape_radix, allowzero)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,
                                                            img_name, temp_res_radix)

                    elif cur_node.op_type == "Expand":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, shape = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        temp_res = fpFunctions.expand(temp_res, shape)
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
                    
                    elif cur_node.op_type == "Silu":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        
                        input_bitwidth = update_json_dict['scale_info'][node_name]['input_bitwidth']
                        y_radix = update_json_dict['scale_info'][node_name]['y_radix'][0]
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.silu(temp_res_list[0], temp_res_radix_list[0],input_bitwidth,
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
 
                    elif cur_node.op_type == "Softplus":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix ,datapath_overflow_info = fpFunctions.softplus(temp_res_list[0], temp_res_radix_list[0],y_radix, self.datapath_bitwidth)
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Tanh":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix,datapath_overflow_info = fpFunctions.tanh(temp_res_list[0], temp_res_radix_list[0], y_radix,
                                                                    self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Sigmoid":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix ,datapath_overflow_info= fpFunctions.sigmoid(temp_res_list[0], temp_res_radix_list[0],
                                                                       y_radix, self.datapath_bitwidth)

                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Mul":
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        output_max = max(abs(update_json_dict['datapath_analysis_info'][node_name]['layer_max']),
                                         abs(update_json_dict['datapath_analysis_info'][node_name]['layer_min']))
                        y_radix = fpFunctions.getradix(output_max, self.datapath_bitwidth)
                        temp_res, temp_res_radix, calc_overflow_info, datapath_overflow_info = fpFunctions.mul(temp_res_list, temp_res_radix_list, y_radix,self.working_bitwidth, self.datapath_bitwidth)

                        dict_overflow[node_name]["working_overflow"] = calc_overflow_info
                        dict_overflow[node_name]["datapath_overflow"] = datapath_overflow_info
                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)

                    elif cur_node.op_type == "Resize":
                        coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode = \
                            self.extractor_a.get_info_resize(cur_node)
                        weight_dict = self.get_node_weights(cur_node)
                        temp_res_list, temp_res_radix_list = self.processor.get_input_tensor_fp(cur_node, \
                            weight_dict,self.input_dict,self.output_dict,int_folder_path,float_folder_path,img_name,output_per_layer)
                        assert len(temp_res_list) == 3, 'input number of this node is wrong!'

                        temp_res, roi, scales = temp_res_list
                        temp_res_radix = temp_res_radix_list[0]
                        sizes = None
                        temp_res = fpFunctions.resize(temp_res, roi, scales, sizes, \
                            coordinate_transformation_mode, cubic_coeff_a, exclude_outside,extrapolation_value, mode, nearest_mode)

                        self.processor.save_compute_results(temp_res, int_folder_path, float_folder_path, node_name,img_name, temp_res_radix)
    
                    else:
                        logging.error("Unsupport op type: {}".format(self.node_list[i].op_type))
                        assert False, "Unsupport op type: {}".format(self.node_list[i].op_type)

                end_time = time.time()
                end_total_time = time.time()
                running_time = round((end_time - start_time)/60, 1)
                total_time = round((end_total_time - start_total_time)/60, 1)
                #logging.debug('time:{} min'.format(running_time), end = '  ')
                #logging.debug('total time:{} min'.format(total_time))
                logging.info('{:.2f}/{:.1f} min running_time/total_time'.format(running_time, total_time))
            
            convert2float64(dict_overflow) ## we do this cause json file doesn't support float32 data type
            dict_overflow_json = json.dumps(dict_overflow, sort_keys=True, indent=4)
            dict_overflow_json_path = os.path.join(output_folder_path, 'overflow.json')
            with open(dict_overflow_json_path, 'w+') as json_file:
                json_file.write(dict_overflow_json)
        
    def fix_point_inference(self, image_npy_folder, output_folder_path, update_json_file_path, 
          log_level = logging.INFO, analysis_mode = 'per_layer', 
          output_per_layer = False, is_hardware = False, accelerate_option="img2col", 
          log_file="", img_preprocess_method="yolo"):
        
        """fix_point_inference function, (per_layer, per_channel_weight)
        Args:
            image_npy_folder (string): the path of image_npy_folder
            output_folder_path (string): the path of output folder
            update_json_file_path (string): the path of update json file
            log_level : log_level for log info 
            analysis_mode (string): analysis mode
            output_per_layer (bool): options for output_per_layer
            is_hardware (bool): options for is_hardware, default to False 
            accelerate_option (string): options for conv accelerate, default to img2col 
            log_file (string): the path for file saving log info 
            img_preprocess_method (string) : the method for img_preprocess

        """

        logging.critical("analysis_mode: {}".format(analysis_mode))
        #set output information 
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)

        int_folder_path = os.path.join(output_folder_path, "int_res")
        utils.check_folder(int_folder_path)
    
        float_folder_path = os.path.join(output_folder_path, "float_res")
        utils.check_folder(float_folder_path)
                 
        if log_file != "":
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(format='%(message)s', level=log_level, filename=log_file)
        else:
            logging.basicConfig(format='%(message)s', level=log_level)

        f = open(update_json_file_path, encoding = 'utf-8')
        update_json_dict = json.load(f)

        img_name_list = self._get_dfp_input(img_preprocess_method, image_npy_folder, float_folder_path, int_folder_path) 

        if analysis_mode == 'per_channel':
            pass
            # scaled_weight_inference(output_folder_path, update_json_dict, output_per_layer, log_level = logging.INFO, group_size = 1, img_preprocess_method = "yolo")
        elif analysis_mode == 'per_channel_weight':
            self._fix_point_inference_per_channel_weight(img_name_list, output_folder_path, float_folder_path, int_folder_path, update_json_dict, output_per_layer, 
                log_level = logging.INFO)
        elif analysis_mode == 'per_layer':
            self._fix_point_inference_per_layer(img_name_list, output_folder_path, float_folder_path, int_folder_path, update_json_dict, output_per_layer, 
                log_level = logging.INFO)
        else:
            assert False, 'analysis_mode is unsupported'
