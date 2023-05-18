from onnxparser.parser import OnnxParser,AttributeExtractor
from .common import DataProcessor
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt

import numpy as np
import os
import json
#import keras
#from keras.models import *
import shutil
import logging
from tqdm import tqdm
from .functions import functions_fl
from .functions import custom_fl
from utils import utils
#from preprocessing.img_to_txt import ImgPreprocessor

class FloatInferencer(OnnxParser):

    def __init__(self, model_path, img_folder_path, input_case_folder, list_of_customized_name=None, whether_analyse_datapath=False):
        self.model_path = model_path
        onnxmodel = onnx.load_model(model_path)
        super().__init__(onnxmodel)
        self.img_folder_path = img_folder_path
        self.input_case_folder = input_case_folder
        self.list_of_customized_name = list_of_customized_name
        self.datapath_res = {}
        self.whether_analyse_datapath = whether_analyse_datapath
        self.extractor_a = AttributeExtractor()
        self.processor = DataProcessor()

    
    def float_inference(self, output_folder_path, output_per_layer = False, log_level = logging.INFO, accelerate_option="img2col", log_file="", \
                        img_preprocess_method = "yolo", concat_change = True, top_k=10, percent=1.0, draw_figure=False, pick_img_num=100):
        ##TODO:top_k=10, percent=1.0, draw_figure=False,these settings need to be removed or added. Prelu function need to update
        ##inference, and save datapath result when whether_analyse_datapath is True
        logging.basicConfig(format='%(message)s', level=log_level)
  
        utils.check_folder(output_folder_path)
        temp_folder_path = os.path.join(output_folder_path, "float_temp")
        utils.check_folder(temp_folder_path)

        if log_file != "":
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(format='%(message)s', level=log_level, filename=log_file)
        else:
            logging.basicConfig(format='%(message)s', level=log_level)


        datapath_analysis_json = ''
        
        #prepare for the inputs
        img_path = utils.read_directory(self.img_folder_path)
        if self.whether_analyse_datapath:
            img_path = img_path[:pick_img_num]
            #data_max = ImgPreprocessor().get_datapath_input(img_preprocess_method)
            data_max_path = os.path.join(self.input_case_folder, 'data_max.txt')
            data_max_file = open(data_max_path, 'r')
            data_max = data_max_file.readline().replace('\n','')
            data_max = float(data_max)
            self.datapath_res['input']={
                    "node_id": -1,
                    "layer_max": data_max,
                    "layer_min": data_max,
                    "percent": percent,
                    "top_k_layer_max": [],
                    "top_k_layer_min": [],
                }

        logging.info('### total image number:{}'.format(len(img_path)))
        img_name_list = []
        for img in img_path:
            img_name = os.path.splitext(os.path.split(img)[1])[0]
            img_name_list.append(img_name)
            shutil.copy(img, '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.model_graph_input[0], img_name))

        for i in tqdm(range(len(self.node_list))):
            node_name = self.node_list[i].name
            a_max, a_min = -sys.maxsize, sys.maxsize

            if self.node_list[i].op_type == "Constant": # means this node is a constant node
                logging.info("{}/{} {}  \top_type: {}".format(i, len(self.model.graph.node), node_name, self.node_list[i].op_type))                
                continue
            else:
                logging.info("{}/{} {}  \top_type: {}".format(i, len(self.model.graph.node), node_name, self.node_list[i].op_type))
                
                for img_name in img_name_list:

                    if self.node_list[i].name in self.list_of_customized_name:
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = custom_fl.custom_node_fl(temp_res, self.node_list[i].name)   ### the custom function call
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Conv":
                        dilation, group, pad, stride = self.extractor_a.get_info_conv(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        if len(temp_res_list) == 3:
                            temp_res, w_kernel, w_bias = temp_res_list
                            [dim_bias] = w_bias.shape
                            w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))
                        elif len(temp_res_list) == 2:
                            temp_res, w_kernel = temp_res_list
                            w_bias = []
                        else:
                            assert False, 'input number of this node is wrong!'
                        
                        w_kernel = functions_fl.update_w_kernel_for_dilation_conv(w_kernel, dilation)
                        if group != 1:
                            temp_res,temp_res_before_bias = functions_fl.depth_wise_conv(temp_res, w_kernel, w_bias, stride, pad, group, dilation = [1, 1]) 
                        else:  
                            temp_res,temp_res_before_bias = functions_fl.conv(temp_res, w_kernel, w_bias, stride, pad, dilation = [1, 1], option=accelerate_option)

                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res                                           

                    elif self.node_list[i].op_type == "BatchNormalization":
                        epsilon, momentum = self.extractor_a.get_info_bn(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 5, 'input number of this node is wrong!'
                        temp_res, gamma, beta, mean, var = temp_res_list
                        temp_res,temp_res_before_bias = functions_fl.batchnormalization(temp_res, gamma, beta, mean, var, epsilon, momentum)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "L2Normalization":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, gamma_list = temp_res_list
                        temp_res = functions_fl.l2_norm(temp_res, gamma_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res                        

                    elif self.node_list[i].op_type == "MaxPool":
                        kernel_size, pad, stride = self.extractor_a.get_info_maxpool(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.maxpool(temp_res, kernel_size, pad, stride)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
            
                    elif self.node_list[i].op_type == "Relu":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        # temp_array = temp_res         #do not change data distribution in this layer when conducting datapathanalysis
                        temp_res = functions_fl.relu(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res        

                    elif self.node_list[i].op_type == "Sigmoid":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.sigmoid(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                                        
                    elif self.node_list[i].op_type == "Softplus":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.softplus(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Tanh":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.tanh(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Mul":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.mul(temp_res_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Mish":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = functions_fl.mish(temp_res_list[0])
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Div":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.div(temp_res_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Sub":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.sub(temp_res_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "upsample_yolov4":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.upsample_yolov4(temp_res_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                                        
                    elif self.node_list[i].op_type == "Flatten":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.flatten(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "LeakyRelu":
                        alpha = self.extractor_a.get_info_leakyrelu(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        # temp_array = temp_res  #do not change data distribution in this layer when conducting datapathanalysis
                        temp_res = functions_fl.LeakyRelu(temp_res, alpha)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Upsample":
                        mode, scales = self.extractor_a.get_info_upsample(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        if len(temp_res_list) == 2:
                            temp_res, scales = temp_res_list
                        elif len(temp_res_list) == 1:
                            temp_res = temp_res_list[0]
                        else:
                            assert False, 'input number of this node is wrong!'
                        temp_res = functions_fl.upsample(temp_res, scales, mode)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Add":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.add(temp_res_list)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Concat":
                        axis = self.extractor_a.get_info_cancat(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        temp_res = functions_fl.concatenate(temp_res_list, axis)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Clip":
                        # x_max, x_min = self.extractor_a.get_info_clip(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        if len(temp_res_list)==3 :
                            x_min = temp_res_list[1]
                            x_max = temp_res_list[2]
                        else :
                            x_max, x_min = self.extractor_a.get_info_clip(self.node_list[i]) 
                        # assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.clip(temp_res, x_min, x_max)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "AveragePool":
                        kernel_size, pad, stride = self.extractor_a.get_info_averagepool(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.averagepool(temp_res, kernel_size, pad, stride)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "PRelu":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, slope = temp_res_list
                        temp_array = temp_res #do not change data distribution in this layer when conducting datapathanalysis
                        temp_res = functions_fl.PRelu(temp_res, slope)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)

                    elif self.node_list[i].op_type == "Silu":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.silu(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                        
                    elif self.node_list[i].op_type == "Shape":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res= temp_res_list[0]
                        temp_res = functions_fl.shape(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Gather":
                        axis = self.extractor_a.get_info_gather(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, indices = temp_res_list
                        temp_res = functions_fl.gather(temp_res, indices, axis)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Unsqueeze":
                        axes = self.extractor_a.get_info_unsqueeze(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.unsqueeze(temp_res,axes)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Cast":
                        to = self.extractor_a.get_info_cast(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.cast(temp_res,to)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Reshape":
                        allowzero = self.extractor_a.get_info_reshape(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res, shape = temp_res_list
                        temp_res = functions_fl.reshape(temp_res,shape,allowzero)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "ConstantOfShape":
                        value = self.extractor_a.get_info_constantofshape(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.constantofshape(temp_res, value)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Exp":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.exp(temp_res)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Equal":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.equal(temp_res_list[0],temp_res_list[1])
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Where":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 3, 'input number of this node is wrong!'
                        temp_res = functions_fl.where(temp_res_list[0],temp_res_list[1], temp_res_list[2])
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Expand":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 2, 'input number of this node is wrong!'
                        temp_res = functions_fl.expand(temp_res_list[0],temp_res_list[1])
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    elif self.node_list[i].op_type == "Transpose":
                        perm = self.extractor_a.get_info_transpose(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 1, 'input number of this node is wrong!'
                        temp_res = temp_res_list[0]
                        temp_res = functions_fl.transpose(temp_res, perm)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Slice":
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 5, 'input number of this node is wrong!'
                        temp_res, starts, ends, axes, steps = temp_res_list
                        temp_res = functions_fl.slice(temp_res, starts, ends, axes, steps)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res
                    
                    elif self.node_list[i].op_type == "Resize":
                        coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode = \
                            self.extractor_a.get_info_resize(self.node_list[i])
                        weight_dict = self.get_node_weights(self.node_list[i])
                        temp_res_list = self.processor.get_input_tensor(self.node_list[i], weight_dict, self.input_dict, temp_folder_path, img_name)
                        assert len(temp_res_list) == 3, 'input number of this node is wrong!'
                        temp_res, roi, scales = temp_res_list
                        sizes = None
                        temp_res = functions_fl.resize(temp_res,roi,scales,sizes, \
                            coordinate_transformation_mode,cubic_coeff_a,exclude_outside,extrapolation_value,mode,nearest_mode)
                        res_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, self.node_list[i].output[0], img_name)
                        np.save(res_path, temp_res)
                        temp_array = temp_res

                    else:
                        assert False, "Unsupport op type: {}".format(self.node_list[i].op_type)

                    #output layer              
                    if not self.whether_analyse_datapath and (output_per_layer == True or (self.node_list[i].output[0] in self.model_graph_output)):
                        np.save(output_folder_path + '/' + 'layer_' + node_name + '_img_' + img_name + '.npy', temp_res)
                        logging.info('saving to: ' + output_folder_path + '/' + 'layer_' + node_name + '_img_' + img_name + '.npy')
                    
                    if self.whether_analyse_datapath:
                        logging.info('res_path = {}'.format(res_path))
                        if self.node_list[i].op_type in ["Conv", "BatchNormalization"]:
                            a_max = max(a_max, np.max(temp_array), np.max(temp_res_before_bias))
                            a_min = min(a_min, np.min(temp_array), np.min(temp_res_before_bias))
                        else:
                            a_max = max(a_max, np.max(temp_array))
                            a_min = min(a_min, np.min(temp_array))

                logging.info('   \t{}'.format(temp_res.shape))
                
                if self.whether_analyse_datapath:
                    #structure info
                    node_inputs = self.node_list[i].input
                    node_inputs = [name for name in node_inputs if name not in self.weight_name_list]
                    node_outputs = self.node_list[i].output
                    node_type = self.node_list[i].op_type
                    self.datapath_res[node_name] = {
                        "node_id": i,
                        "layer_max": a_max,
                        "layer_min": a_min,
                        "percent": percent,
                        "top_k_layer_max": [],
                        "top_k_layer_min": [],
                        "inputs": list(node_inputs),
                        "outputs": list(node_outputs),
                        "op_type": node_type,
                    }

        ##Based on hardware requirements, give all the input layers of concat_layer with the same datapath statistical results
        if self.whether_analyse_datapath:
            if concat_change:
                for i in range(len(self.model.graph.node)):
                    if self.model.graph.node[i].op_type == "Concat":
                        node_name = self.model.graph.node[i].name
                        cur_concat_max = self.datapath_res[node_name]["layer_max"]
                        cur_concat_min = self.datapath_res[node_name]["layer_min"]
                        input_layer_list = self.former_layer_dict[node_name]
                        for input_layer_name in input_layer_list:
                            assert input_layer_name in self.datapath_res.keys()
                            self.datapath_res[input_layer_name]["layer_max"] = cur_concat_max
                            self.datapath_res[input_layer_name]["layer_min"] = cur_concat_min

            
            json_str = json.dumps(self.datapath_res)
            datapath_analysis_json = os.path.join(output_folder_path,'datapath_analysis.json')
            with open(datapath_analysis_json, 'w+') as json_file:
                json_file.write(json_str)

        shutil.rmtree(temp_folder_path)
        
        return self.datapath_res, datapath_analysis_json