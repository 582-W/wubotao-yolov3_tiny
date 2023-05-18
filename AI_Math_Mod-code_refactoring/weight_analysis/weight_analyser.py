import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np 
import os
import json
import shutil
import logging
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pdb
from onnxparser.parser import OnnxParser,AttributeExtractor
from utils.utils import convert2float64


class WeightAnalyser(OnnxParser):

    """Analyzing the parameter information of convolutional layer and batch normalization layers.
    """

    def __init__(self, model_file_path: str, result_json_path: str, analysis_mode: str):
        if not model_file_path or not os.path.exists(model_file_path):
            raise ValueError("Invalid onnx file path '{}'".format(model_file_path))

        if not result_json_path:
            raise ValueError("Invalid onnx file path '{}'".format(result_json_path))

        if analysis_mode not in {'per_layer', 'per_channel_weight'}:
            raise ValueError("Unsupported analysis mode '{}'".format(analysis_mode))

        if not os.path.exists(result_json_path):
            os.mkdir(result_json_path)

        self.model_path = model_file_path
        onnxmodel = onnx.load_model(model_file_path)
        super().__init__(onnxmodel)
        self.extractor_a = AttributeExtractor()
        self.result_json_path = result_json_path
        self.analysis_mode = analysis_mode
        self.CONV_PERCENT = 1.0
        self.BN_PERCENT = 1.0
        self.GROUP_SIZE = 1


    def parse_conv_weight(self, conv_node: onnx) -> list:
        """The corresponding weights and bias are extracted according to the convolution layer name.

        Args:
            conv_node (onnx): onnx node that represents a conv layer

        Returns:
            list: three lists that contains weight, bias, and the combination of the two, respectively.
        """
        w, w_kernel, w_bias = [], [], []
        weight_dict = self.get_node_weights(conv_node)
        if len(weight_dict)  == 1:
            [w_kernel] = [v for v in weight_dict.values()]
        elif len(weight_dict) == 2:
            [w_kernel, w_bias] = [v for v in weight_dict.values()]
        else:
            raise ValueError('weight number of this node is wrong!')

        for ele in w_kernel.flatten():
            w.append(ele)
        if w_bias!=[]:
            for ele in w_bias.flatten():
                w.append(ele)

        return w, w_kernel, w_bias


    def parse_bn_parameters(self, bn_node: onnx) -> list: 
        """Key parameters are extracted from BN and simplified

        Args:
            bn_node (onnx): onnx node that represents a BN layer

        Returns:
            list: two lists: A_list, B_list
        """
        epsilon, _ = self.extractor_a.get_info_bn(bn_node)
        weight_dict = self.get_node_weights(bn_node)
        assert len(weight_dict) == 4, 'weight number of this node is wrong!'
        [gamma, beta, mean, var] = [v for v in weight_dict.values()]

        A_list, B_list = [], []
        for bn_channel in range(len(gamma)):
            A_list.append(gamma[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
            B_list.append(beta[bn_channel] - gamma[bn_channel] * mean[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
        
        return A_list, B_list


    def _remove_outliers(self, data: list, keep_percent: float) -> list:

        data = np.array(data)
        data = data.flatten()
        data = np.sort(data, kind = 'heapsort')
        start, end = int( len(data) * (1 - keep_percent) / 2), int(len(data) * (1 + keep_percent) / 2)

        return data[start: end]


    def _analyse_data(self, data: list, out_dict: dict, key_word: str) -> dict:
        """Make statistics of max,min,mean and std of the current data.

        Args:
            data (list): input data
            out_dict (dict): a dict used to store statistical data
            key_word (str): the string name that represents the current statistics 

        Returns:
            dict: the dict containing statistical data
        """
        out_dict[key_word + '_max'] = np.max(data)
        out_dict[key_word + '_min'] = np.min(data)
        out_dict[key_word + '_mean'] = np.mean(data)
        out_dict[key_word + '_std'] = np.std(data)
        
        return out_dict


    def analyse_weight(self):
 
        logging.critical("analysis_mode: {}".format(self.analysis_mode))

        out_dict = dict()
        
        if self.analysis_mode == "per_layer":
            for i in tqdm(range(len(self.model.graph.node))):
                node_name = self.model.graph.node[i].name

                if self.model.graph.node[i].op_type == "Conv":
                    conv_node = self.model.graph.node[i]
                    w, w_kernel, w_bias = self.parse_conv_weight(conv_node)
                    kernel_num = w_kernel.shape[0]
                    if len(w) == 0: #means that there's no weight for this layer
                        continue
                    logging.info("node id: {} layer_name: {}".format(i, node_name))

                    out_dict[node_name] = {}
                    w = self._remove_outliers(w, self.CONV_PERCENT)
                    out_dict[node_name]['onnx_node_id'] = i
                    out_dict[node_name] = self._analyse_data(w, out_dict[node_name], 'per_layer')

                    if len(w_kernel) > 0:
                        w_kernel = self._remove_outliers(w_kernel, self.CONV_PERCENT)
                        out_dict[node_name] = self._analyse_data(w_kernel, out_dict[node_name], 'kernel')
                        out_dict[node_name]['kernel_num'] = kernel_num

                    if len(w_bias) > 0:
                        out_dict[node_name] = self._analyse_data(w_bias, out_dict[node_name], 'bias')
                
                elif self.model.graph.node[i].op_type == "BatchNormalization":
                    bn_node = self.model.graph.node[i]
                    A_list, B_list = self.parse_bn_parameters(bn_node)
                    out_dict[node_name] = {}
                    if len(A_list) > 0:
                        A_list = self._remove_outliers(A_list, self.BN_PERCENT)
                        out_dict[node_name] = self._analyse_data(A_list, out_dict[node_name], 'a')
                    if len(B_list) > 0:
                        B_list = self._remove_outliers(B_list, self.BN_PERCENT)
                        out_dict[node_name] = self._analyse_data(B_list, out_dict[node_name], 'b')
                
                elif self.model.graph.node[i].op_type == "LeakyRelu":
                    out_dict[node_name] = {}
                    alpha = self.extractor_a.get_info_leakyrelu(self.model.graph.node[i])
                    out_dict[node_name]['alpha'] = alpha

                elif self.model.graph.node[i].op_type == "PRelu":
                    weight_dict = self.get_node_weights(self.model.graph.node[i])
                    [slope] = [v for v in weight_dict.values()]
                    out_dict[node_name] = {}
                    out_dict[node_name]['slope_max'] = np.max(np.array(slope))
                    out_dict[node_name]['slope_min'] = np.min(np.array(slope))

        elif self.analysis_mode == "per_channel_weight":
            for i in tqdm(range(len(self.model.graph.node))):
                node_name = self.model.graph.node[i].name

                if self.model.graph.node[i].op_type == "Conv":
                    conv_node = self.model.graph.node[i]
                    w, w_kernel, w_bias = self.parse_conv_weight(conv_node)
                    if len(w) == 0: #means that there's no weight for this layer
                        continue
                    kernel_num = w_kernel.shape[0]
                    group_max_list, group_min_list = [], []
                    group_num = int(kernel_num // self.GROUP_SIZE)
                    rest_num = int(kernel_num % self.GROUP_SIZE)
                    for j in range(group_num):
                        slice_start, slice_end = int(self.GROUP_SIZE * j), int(self.GROUP_SIZE * (j + 1))
                        group_data = w_kernel[slice_start : slice_end, :, :, :]
                        group_data = self._remove_outliers(group_data, self.CONV_PERCENT)
                        group_max_list.append(np.max(group_data))
                        group_min_list.append(np.min(group_data))
                    if rest_num != 0:
                        rest_group_data = w_kernel[: , : , :, -rest_num :]
                        rest_group_data = self._remove_outliers(rest_group_data, self.CONV_PERCENT)
                        group_max_list.append(np.max(rest_group_data))
                        group_min_list.append(np.min(rest_group_data))

                    out_dict[node_name] = {}
                    w = self._remove_outliers(w, self.CONV_PERCENT)
                    out_dict[node_name]['onnx_node_id'] = i
                    out_dict[node_name] = self._analyse_data(w, out_dict[node_name], 'per_layer')

                    if len(w_kernel) > 0:
                        w_kernel = self._remove_outliers(w_kernel, self.CONV_PERCENT)
                        out_dict[node_name] = self._analyse_data(w_kernel, out_dict[node_name], 'kernel')
                        out_dict[node_name]['group_max_list'] = group_max_list
                        out_dict[node_name]['group_min_list'] = group_min_list

                elif self.model.graph.node[i].op_type == "BatchNormalization":
                    bn_node = self.model.graph.node[i]
                    A_list, B_list = self.parse_bn_parameters(bn_node)
                    out_dict[node_name] = {}
                    if len(A_list) > 0:
                        out_dict[node_name]['a_max_list'] = A_list
                        out_dict[node_name]['a_min_list'] = A_list
                    if len(B_list) > 0:
                        out_dict[node_name]['b_max_list'] = B_list
                        out_dict[node_name]['b_min_list'] = B_list
                
                elif self.model.graph.node[i].op_type == "LeakyRelu":
                    out_dict[node_name] = {}
                    alpha = self.extractor_a.get_info_leakyrelu(self.model.graph.node[i])
                    out_dict[node_name]['alpha'] = alpha
                
                elif self.model.graph.node[i].op_type == "PRelu":
                    weight_dict = self.get_node_weights(self.model.graph.node[i])
                    [slope] = [v for v in weight_dict.values()]
                    out_dict[node_name] = {}
                    out_dict[node_name]['slope_max'] = np.max(np.array(slope))
                    out_dict[node_name]['slope_min'] = np.min(np.array(slope))

        convert2float64(out_dict) ## we do this cause json file doesn't support float32 data type
        json_str = json.dumps(out_dict, sort_keys=False, indent=4)
        weight_analysis_json_path = os.path.join(self.result_json_path, 'weight_analysis.json')
        with open(weight_analysis_json_path, 'w+') as json_file:
            json_file.write(json_str)

        return out_dict, weight_analysis_json_path

        
