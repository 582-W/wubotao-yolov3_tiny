import copy
import json
import logging
import math
import numpy as np 
import onnx
import os
import pdb
import sys,getopt
import shutil
import time
from .functions_mu import *
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
from tqdm import tqdm
sys.path.append("..")
from onnxparser.parser import OnnxParser
from inference.fl_inferencer import FloatInferencer
from utils import utils
from weight_analysis.weight_analyser import WeightAnalyser

class ModelUpdater(OnnxParser):

    def __init__(self, model_path, input_case_folder, img_folder_path, weight_analysis_json_path, datapath_analysis_json_path, config_file_path, output_res_path, \
                analysis_mode, run_update, img_preprocess_method, concat_change, list_of_customized_name, log_level=logging.INFO, log_file = ""):
        self.model_path = model_path
        onnxmodel = onnx.load_model(model_path)
        super().__init__(onnxmodel)
        self.img_folder_path = img_folder_path
        self.input_case_folder = input_case_folder
        self.list_of_customized_name = list_of_customized_name
        self.weight_analysis_json_path = weight_analysis_json_path
        self.datapath_analysis_json_path = datapath_analysis_json_path
        self.config_file_path = config_file_path
        self.output_res_path = output_res_path
        self.analysis_mode = analysis_mode
        self.run_update = run_update
        self.img_preprocess_method = img_preprocess_method
        self.concat_change = concat_change
        self.log_level = log_level
        self.log_file = log_file
        self.extra_info = {}
    
    
    def model_update(self, limit_shift, shift_upper_limit, shift_lower_limit):

        if self.log_file != "":
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(format='%(message)s', level=self.log_level, filename=self.log_file)
        else:
            logging.basicConfig(format='%(message)s', level=self.log_level)
        logging.critical("analysis_mode: {}".format(self.analysis_mode))

        utils.check_folder(self.output_res_path)

        ## Load config json file, weight_analysis json file, and datapath_analysis json file.
        f = open(self.config_file_path, encoding = 'utf-8')
        config_dict = json.load(f)
        kernel_bitwidth = config_dict['conv_bitwidth']['kernel']
        w_file = open(self.weight_analysis_json_path,'r',encoding='utf-8')
        weight_analy_dict = json.load(w_file)
        datapath_file = open(self.datapath_analysis_json_path,'r',encoding='utf-8')
        datapath_dict = json.load(datapath_file)
        updated_onnx_path = os.path.join(self.output_res_path, "update.onnx")

        if not self.run_update:
            shutil.copy(self.model_path, updated_onnx_path)
            self.extra_info['weight_analysis_info'] = weight_analy_dict
            self.extra_info['datapath_analysis_info'] = datapath_dict
            last_node_list = [ self.output_dict[i] for i in self.model_graph_output]

                        #get scale
            scale_dict = get_scale_info(datapath_dict,weight_analy_dict,config_dict,self.analysis_mode,limit_shift, \
                shift_upper_limit, shift_lower_limit,self.node_list,self.former_layer_dict,self.next_layer_dict,last_node_list,self.model_graph_input)
            self.extra_info['scale_info'] = scale_dict

            if self.concat_change == True:
                #adjust leakyrelu radix
                new_scale_dict = concat_relu_change_radix(datapath_dict,scale_dict,self.next_layer_dict,self.node_list,self.former_layer_dict,last_node_list)
                new_scale_dict = concat_lut_change_radix(datapath_dict,new_scale_dict,self.next_layer_dict,self.node_list,self.former_layer_dict,last_node_list)
            else :
                new_scale_dict =  scale_dict     
            new_scale_dict = add_change_radix(datapath_dict,new_scale_dict,self.next_layer_dict,self.node_list,self.former_layer_dict,last_node_list)    
            new_scale_dict = re_compute_conv_radix(self.analysis_mode,datapath_dict,scale_dict,self.next_layer_dict,self.node_list,self.former_layer_dict,last_node_list, \
                weight_analy_dict,config_dict,limit_shift,shift_upper_limit, shift_lower_limit)
            self.extra_info['scale_info'] = new_scale_dict
            #get io shift(add,concat)
            io_shift_dict = get_io_shift(self.extra_info,datapath_dict,config_dict,self.node_list,self.former_layer_dict)
            self.extra_info['io_shift_info'] = io_shift_dict
            self.extra_info = get_shift_bit(self.analysis_mode, config_dict, updated_onnx_path, datapath_dict, weight_analy_dict, \
                                            self.extra_info, self.output_dict, self.model_graph_input)
        
        ### run model update
        else:
            bn_count = 1
            new_model = copy.deepcopy(self.model)

            for i in range(len(self.model.graph.node)):
                node_name = self.model.graph.node[i].name
                if self.analysis_mode == 'per_layer':
                    if self.model.graph.node[i].op_type == "Conv" and self.model.graph.node[i].output[-1] not in self.model_graph_output:
                        w_kernel_max = weight_analy_dict[node_name]['kernel_max']
                        w_kernel_min = weight_analy_dict[node_name]['kernel_min']
                        weight_max = max(abs(w_kernel_max), abs(w_kernel_min))
                        if node_name in config_dict:
                            target_bitwidth = config_dict[node_name]["kernel"]
                        else:
                            target_bitwidth = kernel_bitwidth

                        w_bias = []
                        weight_dict = self.get_node_weights(self.model.graph.node[i])
                        if len(weight_dict)  == 1:
                            [(w_kernel_name,w_kernel)] = [(k,v) for k,v in weight_dict.items()]
                        elif len(weight_dict) == 2:
                            [(w_kernel_name,w_kernel),(w_bias_name,w_bias)] = [(k,v) for k,v in weight_dict.items()]
                        
                        channel_num = w_kernel.shape[0]
                        logging.info("node_name: {}".format(node_name))
                        
                        ### determine whether to update
                        scale_flag, scale = is_update_conv_weight(weight_max, target_bitwidth, threshold = 0.8)
                        logging.info("scale_flage: {}, scale: {}".format(scale_flag, scale))
                        
                        if w_kernel_name in self.initializer_dict.keys():                                                    
                            if scale_flag == 1:
                                logging.info("update_conv_weight...")
                                update_conv_weight_initializer(new_model, w_kernel, w_kernel_name, scale)
                                if w_bias!=[]:
                                    logging.info("update_conv_bias...")
                                    update_conv_weight_initializer(new_model, w_bias, w_bias_name, scale)
                                
                                ### add BN layer
                                logging.info("the channel num is: {}, bn scale is: {}".format(channel_num, scale))
                                next_node_list = OnnxParser.find_next_node(self.model,self.model.graph.node[i])[1]
                                if len(next_node_list) == 1 and next_node_list[0].op_type == 'BatchNormalization':
                                    ##if next node is bn, we modify weight in bn directly
                                    for node_ in new_model.graph.node:
                                        if node_.name == next_node_list[0].name:
                                            update_bn_weight_initializer(new_model, node_, scale, channel_num)
                                            break
                                else:
                                    for node_ in new_model.graph.node:
                                        if node_.name == self.model.graph.node[i].name:
                                            add_bn_after_conv_initializer_(new_model, node_, scale, channel_num, bn_count)
                                            bn_count += 1
                                            break
                            
                        elif w_kernel_name in self.constant_node_output:
                            ##TODO:constant node neeed to be supported
                            assert False, 'currently constant node is not supported in model update'
                        else:
                            assert False, 'weight is stored in unsupported place'


                    logging.info("node id : {} layer_name: {}".format(i, node_name))

                ## per_channel_weight update
                else:

                    if self.model.graph.node[i].op_type == "Conv" and self.model.graph.node[i].output[-1] not in self.model_graph_output:
                        w_kernel_max_list = weight_analy_dict[node_name]['group_max_list']
                        w_kernel_min_list = weight_analy_dict[node_name]['group_min_list']
                        if node_name in config_dict:
                            target_bitwidth = config_dict[node_name]["kernel"]
                        else:
                            target_bitwidth = kernel_bitwidth

                        w_bias = []
                        weight_dict = self.get_node_weights(self.model.graph.node[i])
                        if len(weight_dict)  == 1:
                            [(w_kernel_name,w_kernel)] = [(k,v) for k,v in weight_dict.items()]
                        elif len(weight_dict) == 2:
                            [(w_kernel_name,w_kernel),(w_bias_name,w_bias)] = [(k,v) for k,v in weight_dict.items()]
                        
                        channel_num = w_kernel.shape[0]
                        logging.info("node_name: {}".format(node_name))
                        ### analyasis radix
                        scale_flag_list, scale_list = is_update_conv_weight_channel(w_kernel_max_list, w_kernel_min_list, target_bitwidth, threshold = 0.8)                                    
                        logging.info("scale_flage_list: {}, scale_list: {}".format(scale_flag_list, scale_list))
                        
                        if w_kernel_name in self.initializer_dict.keys():   
                            if 1 in scale_flag_list:
                                logging.info("update_conv_weight...")
                                update_conv_weight_initializer(new_model, w_kernel, w_kernel_name, scale_list)
                                if w_bias != []:
                                    logging.info("update_conv_bias...")
                                    update_conv_weight_initializer(new_model, w_bias, w_bias_name, scale_list)
                                            
                                ### add BN layer
                                logging.info("the channel num is: {}, bn scale_list is: {}".format(channel_num, scale_list))
                                next_node_list = OnnxParser.find_next_node(self.model,self.model.graph.node[i])[1]
                                if len(next_node_list) == 1 and next_node_list[0].op_type == 'BatchNormalization':
                                    ##if next node is bn, we modify weight in bn directly
                                    for node_ in new_model.graph.node:
                                        if node_.name == next_node_list[0].name:
                                            update_bn_weight_initializer(new_model, node_, scale_list, channel_num)
                                            break
                                else:
                                    for node_ in new_model.graph.node:
                                        if node_.name == self.model.graph.node[i].name:
                                            add_bn_after_conv_initializer_(new_model, node_, scale_list, channel_num, bn_count)
                                            bn_count += 1
                                            break
                        
                        elif w_kernel_name in self.constant_node_output:
                            ##TODO:constant node neeed to be supported
                            assert False, 'currently constant node is not supported in model update'
                        
                        else:
                            assert False, 'weight is stored in unsupported place'

                    logging.info("node id : {} layer_name: {}".format(i, node_name))
            

            onnx.save_model(new_model, updated_onnx_path)
            model_upd = onnx.load(updated_onnx_path)

            update_model = add_size_info(self.model_path, updated_onnx_path)
            
            ## Put all information which contains config info, weight_analysis info, and datapath_analysis info into extra_info dict.
            weight_analy_dict_update, _ = WeightAnalyser(updated_onnx_path, self.output_res_path, analysis_mode = self.analysis_mode).analyse_weight()
            datapath_analysis_output = os.path.join(self.output_res_path,'datapath_analysis_output')
            datapath_analyser = FloatInferencer(updated_onnx_path, self.img_folder_path, self.input_case_folder, list_of_customized_name=self.list_of_customized_name, whether_analyse_datapath=True)
            datapath_dict_update, _ = datapath_analyser.float_inference(datapath_analysis_output,img_preprocess_method=self.img_preprocess_method, \
                                                                            concat_change=self.concat_change, log_level=self.log_level, log_file = self.log_file)
            shutil.move(os.path.join(datapath_analysis_output,'datapath_analysis.json'),self.output_res_path)
            shutil.rmtree(datapath_analysis_output)
            output_dict_update = OnnxParser(model_upd).output_dict
            update_model, datapath_dict_update = change_add_input_order(updated_onnx_path, datapath_dict_update, config_dict, output_dict_update)
            onnx.save_model(update_model, updated_onnx_path)
            model_upd = onnx.load(updated_onnx_path)

            # manage extra_info
            self.extra_info['weight_analysis_info'] = weight_analy_dict_update
            self.extra_info['datapath_analysis_info'] = datapath_dict_update
            # obtain info of updated model
            inferencer_update = OnnxParser(model_upd)
            output_dict_update = inferencer_update.output_dict
            model_graph_input_update = inferencer_update.model_graph_input
            node_list_update = inferencer_update.node_list
            former_layer_dict_update = inferencer_update.former_layer_dict
            next_layer_dict_update = inferencer_update.next_layer_dict
            last_node_list = [ inferencer_update.output_dict[i] for i in inferencer_update.model_graph_output]
            #get scale info
            scale_dict = get_scale_info(datapath_dict_update,weight_analy_dict_update,config_dict,self.analysis_mode, \
                limit_shift,shift_upper_limit,shift_lower_limit,node_list_update,former_layer_dict_update,next_layer_dict_update,last_node_list,model_graph_input_update)
            self.extra_info['scale_info'] = scale_dict
            if self.concat_change == True:
                #adjust leakyrelu radix
                last_node_list = [ inferencer_update.output_dict[i] for i in inferencer_update.model_graph_output]
                new_scale_dict = concat_relu_change_radix(datapath_dict_update,scale_dict,next_layer_dict_update,node_list_update,former_layer_dict_update,last_node_list)
                new_scale_dict = concat_lut_change_radix(datapath_dict_update,new_scale_dict,next_layer_dict_update,node_list_update,former_layer_dict_update,last_node_list)
                # self.extra_info['scale_info'] = new_scale_dict

            new_scale_dict = add_change_radix(datapath_dict_update,new_scale_dict,next_layer_dict_update,node_list_update,former_layer_dict_update,last_node_list)    
            new_scale_dict = re_compute_conv_radix(self.analysis_mode,datapath_dict_update,scale_dict,next_layer_dict_update,node_list_update,former_layer_dict_update,last_node_list, \
                weight_analy_dict_update,config_dict,limit_shift,shift_upper_limit,shift_lower_limit)
            self.extra_info['scale_info'] = new_scale_dict
            #get shift info
            self.extra_info = get_shift_bit(self.analysis_mode, config_dict, updated_onnx_path, datapath_dict_update, weight_analy_dict_update, \
                                            self.extra_info, output_dict_update, model_graph_input_update)

            #get io shift(add,concat)
            io_shift_dict = get_io_shift(self.extra_info,datapath_dict_update,config_dict,node_list_update,former_layer_dict_update)
            self.extra_info['io_shift_info'] = io_shift_dict

        json_str = json.dumps(self.extra_info, indent=4)
        updated_json_path = os.path.join(self.output_res_path, "update.json")
        with open(updated_json_path, 'w+') as json_file:
            json_file.write(json_str)


