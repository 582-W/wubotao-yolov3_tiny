import onnx
import os
import numpy as np
import logging


class DataProcessor():
    def __init__(self):
        pass
    
    
    def rm_tempfile(self,node_name,input_name,input_dict,temp_file_path):
        if node_name in input_dict[input_name]:
            input_dict[input_name].remove(node_name)
        if len(input_dict[input_name]) == 0:
            os.remove(temp_file_path)
    
    def save_compute_results(self, temp_res, int_folder_path, float_folder_path, node_name, img_name, temp_res_radix):
        res_path = '{}/layer_{}_img_{}_radix_{}.npy'.format(int_folder_path, node_name, img_name, temp_res_radix)
        np.save(res_path, temp_res)
        float_temp_res = temp_res * 2 ** (-temp_res_radix)
        float_res_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, node_name, img_name)
        np.save(float_res_path, float_temp_res)


    def get_input_tensor(self,node,weight_dict,input_dict,temp_folder_path,img_name):
        '''node: onnx layer
           weight_dict: output of Inferencer().get_node_weights
        '''
        temp_res_list = []
        for input_name in node.input:
            if input_name in weight_dict:
                temp_res_list.append(weight_dict.pop(input_name))
            else: ## feature map of last layer
                temp_file_path = '{}/layer_{}_img_{}.npy'.format(temp_folder_path, input_name, img_name)
                logging.debug("temp_file_path: {}".format(temp_file_path))
                assert os.path.exists(temp_file_path), "fail to find the temp file: {}".format(temp_file_path)
                temp_res = np.load(temp_file_path)
                temp_res_list.append(temp_res)
                self.rm_tempfile(node.name,input_name,input_dict,temp_file_path)
        
        return temp_res_list
    

    def get_input_tensor_fp(self,node,weight_dict,input_dict,output_dict,int_folder_path,float_folder_path,img_name,output_per_layer):
        '''node: onnx layer
           weight_dict: output of Inferencer().get_node_weights
        '''
        temp_res_list = []
        temp_res_radix_list = []
        for input_name in node.input:
            if input_name in weight_dict:
                temp_res_list.append(weight_dict.pop(input_name))
            else: ## feature map of last layer
                for file in os.listdir(int_folder_path):
                    file_name = os.path.splitext(file)[0]
                    index = file_name.index('radix_')
                    if file_name[: index - 1] == 'layer_{}_img_{}'.format(output_dict[input_name], img_name):
                        temp_int_file_path = '{}/{}.npy'.format(int_folder_path, file_name)
                        temp_res_radix = int(file_name[index + 6: ])
                        temp_res_radix_list.append(temp_res_radix)
                        break
                temp_res = np.load(temp_int_file_path)
                temp_res_list.append(temp_res)   
                temp_float_file_path = '{}/layer_{}_img_{}.npy'.format(float_folder_path, output_dict[input_name], img_name)
                if not output_per_layer:
                    self.rm_tempfile(node.name,input_name,input_dict,temp_int_file_path)
                    self.rm_tempfile(node.name,input_name,input_dict,temp_float_file_path)
          
        return temp_res_list, temp_res_radix_list