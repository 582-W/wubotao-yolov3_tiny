import json
import logging
import numpy as np
import unittest
import os
import sys
sys.path.append("..")
sys.path.append("../..")
from inference.functions import functions_fl as flfunctions

class TestFloatInferencer(unittest.TestCase):
    def setUp(self):
        self.test_obj = flfunctions  ### object or module to be tested
        self.case_path = "case_files"   ### case path
        self.conv_case_path = os.path.join(self.case_path, "case_conv")
        self.bn_case_path = os.path.join(self.case_path, "case_bn")        
        self.upsample_case_path = os.path.join(self.case_path, "case_upsample")
        self.dwconv_case_path = os.path.join(self.case_path, "case_dwconv")
        self.maxpool_case_path = os.path.join(self.case_path, "case_maxpool")
        self.averagepool_case_path = os.path.join(self.case_path, "case_averagepool")
        self.relu_case_path = os.path.join(self.case_path, "case_relu")
        self.leakyrelu_case_path = os.path.join(self.case_path, "case_leakyrelu")
        self.concat_case_path = os.path.join(self.case_path, "case_concat")
        self.add_case_path = os.path.join(self.case_path, "case_add")

    def test_conv(self):
        '''
        test the float conv function
        '''
        conv_input_path = os.path.join(self.conv_case_path, "conv2d_input.npy")
        conv_w_kernel_path = os.path.join(self.conv_case_path, "conv2d_weight.npy")
        conv_w_bias_path = os.path.join(self.conv_case_path, "conv2d_bias.npy")
        conv_out_path = os.path.join(self.conv_case_path, "conv2d_output_fl.npy")
        conv_info_path = os.path.join(self.conv_case_path, "conv_info.json")
        
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))
        
        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]

        for option in ['normal','img2col','cuda']:
            conv_out, _ = self.test_obj.conv(x, w_kernel, w_bias, stride, pad, dilation, option)
            self.assertTrue((conv_out == np.load(conv_out_path)).all())
    
    def test_dwconv(self):
        '''
        test the float dwconv function
        '''
        conv_input_path = os.path.join(self.dwconv_case_path, "dwconv_input.npy")
        conv_w_kernel_path = os.path.join(self.dwconv_case_path, "dwconv_weight.npy")
        conv_w_bias_path = os.path.join(self.dwconv_case_path, "dwconv_bias.npy")
        conv_out_path = os.path.join(self.dwconv_case_path, "dwconv_output_fl.npy")
        conv_info_path = os.path.join(self.dwconv_case_path, "dwconv_info.json")
        
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))

        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]
        group = conv_info_dict["group"]

        conv_out, _ = self.test_obj.depth_wise_conv(x, w_kernel, w_bias, stride, pad, group, dilation)
        self.assertTrue((conv_out == np.load(conv_out_path)).all())
    
    def test_bn(self):
        '''
        test the float bn function
        '''
        bn_input_path = os.path.join(self.bn_case_path, "bn_input.npy")
        bn_a_path = os.path.join(self.bn_case_path, "bn_gamma.npy")
        bn_b_path = os.path.join(self.bn_case_path, "bn_beta.npy")
        bn_mean_path = os.path.join(self.bn_case_path, "bn_mean.npy")
        bn_var_path = os.path.join(self.bn_case_path, "bn_var.npy")
        bn_out_path = os.path.join(self.bn_case_path, "bn_output_fl.npy")
        bn_info_path = os.path.join(self.bn_case_path, "bn_info.json")
        
        f = open(bn_info_path, encoding = 'utf-8')
        bn_info_dict = json.load(f)
        f.close()
        x = np.load(bn_input_path)
        gamma = np.load(bn_a_path)
        beta = np.load(bn_b_path)
        mean = np.load(bn_mean_path)
        var = np.load(bn_var_path)

        epsilon = bn_info_dict["epsilon"]
        momentum = bn_info_dict["momentum"]
        
        bn_out, _ = self.test_obj.batchnormalization(x, gamma, beta, mean, var, epsilon, momentum)
        self.assertTrue((bn_out == np.load(bn_out_path)).all())
    
    def test_leakyrelu(self):
        '''
        test the float leakyrelu function
        '''
        leakyrelu_input_path = os.path.join(self.leakyrelu_case_path, "leakrely_input.npy")
        leakyrelu_out_path = os.path.join(self.leakyrelu_case_path, "leakyrelu_output_fl.npy")
        leakyrelu_info_path = os.path.join(self.leakyrelu_case_path, "leakyrelu_info.json")
        
        f = open(leakyrelu_info_path, encoding = 'utf-8')
        leakyrelu_info_dict = json.load(f)
        f.close()
        x = np.load(leakyrelu_input_path)
        alpha = leakyrelu_info_dict["alpha"]
        
        leakyrelu_out = self.test_obj.LeakyRelu(x, alpha)
        self.assertTrue((leakyrelu_out == np.load(leakyrelu_out_path)).all())
    
    def test_add(self):
        '''
        test the float add function
        '''
        add_input1_path = os.path.join(self.add_case_path, "add_input1.npy")
        add_input2_path = os.path.join(self.add_case_path, "add_input2.npy")
        add_out_path = os.path.join(self.add_case_path, "add_output_fl.npy")

        x_list = [np.load(add_input1_path), np.load(add_input2_path)]
        add_out = self.test_obj.add(x_list)
        self.assertTrue((add_out == np.load(add_out_path)).all())

    def test_upsample(self):
        '''
        test the float upsample funciton
        '''
        upsample_input_path = os.path.join(self.upsample_case_path, "upsample_input.npy")
        upsample_out_path = os.path.join(self.upsample_case_path, "upsample_out_fl.npy")
        upsample_info_path = os.path.join(self.upsample_case_path, "upsample_info.json")
        upsample_scale_path = os.path.join(self.upsample_case_path, "up_sampling2d_1_scales.npy")
        
        f = open(upsample_info_path, encoding = 'utf-8')
        upsample_info_dict = json.load(f)
        f.close()
        mode = upsample_info_dict["mode"].encode()
        x = np.load(upsample_input_path)
        scales = np.load(upsample_scale_path)
        upsample_out = self.test_obj.upsample(x, scales, mode)
        self.assertTrue((upsample_out == np.load(upsample_out_path)).all())

    def test_concat(self):
        '''
        test the float concat function
        '''
        concat_input1_path = os.path.join(self.concat_case_path, "concat_input1.npy")
        concat_input2_path = os.path.join(self.concat_case_path, "concat_input2.npy")
        concat_out_path = os.path.join(self.concat_case_path, "concat_output_fl.npy")
        concat_info_path = os.path.join(self.concat_case_path, "concat_info.json")
        
        f = open(concat_info_path, encoding = 'utf-8')
        concat_info_dict = json.load(f)
        f.close()
        x_list = [np.load(concat_input1_path), np.load(concat_input2_path)]
        axis = concat_info_dict["axis"]
        concat_out = self.test_obj.concatenate(x_list, axis)
        self.assertTrue((concat_out == np.load(concat_out_path)).all())
    
    
if __name__ == '__main__': 
    unittest.main()