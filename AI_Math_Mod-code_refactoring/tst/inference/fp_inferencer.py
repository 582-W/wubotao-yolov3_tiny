import numpy as np
import os
import json
import unittest
import logging
import sys
sys.path.append("..")
sys.path.append("../..")
from inference.functions import functions_fp as fpFunctions
from inference.functions.functions_fp import (depthwise_conv, conv,
                                    conv_by_channel, relu, PRelu, clip, averagepool, maxpool, 
                                    flatten, LeakyRelu, add, concatenate, upsample, BatchNormalization,
                                    BN_per_channel, depth_wise_conv_by_channel)

class TestFixedPointInferencer(unittest.TestCase):
    def setUp(self):
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

        self.config_file_path = os.path.join(self.case_path, "hardware.json")
        
        f = open(self.config_file_path, encoding = 'utf-8')
        self.config_json_dict = json.load(f)
        f.close()

    def test_conv(self):
        '''
        test the dfp conv function
        '''
        conv_input_path = os.path.join(self.conv_case_path, "conv2d_input.npy")
        conv_w_kernel_path = os.path.join(self.conv_case_path, "conv2d_weight.npy")
        conv_w_bias_path = os.path.join(self.conv_case_path, "conv2d_bias.npy")
        conv_out_path = os.path.join(self.conv_case_path, "conv2d_perlayer_output.npy")
        conv_info_path = os.path.join(self.conv_case_path, "conv_info.json")
        
        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))

        x_radix = conv_info_dict["x_radix"]
        y_radix = conv_info_dict["y_radix"]
        w_kernel_radix = conv_info_dict["w_kernel_radix"]
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]

        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        kernel_bitwidth = self.config_json_dict["conv_bitwidth"]["kernel"]
        bias_bitwidth = self.config_json_dict["conv_bitwidth"]["bias"]
       
        w_bias_radix = y_radix
        w_kernel = np.round(w_kernel * 2 ** w_kernel_radix)
        w_bias = np.round(w_bias * 2 ** w_bias_radix)
        
        kernel_overflow = 2 ** (kernel_bitwidth - 1) - 1
        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]
        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1

        conv_out, _, _, _ = conv(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, 
            working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], 
            is_hardware = False)
        
        self.assertTrue((conv_out == np.load(conv_out_path)).all())
    
    def test_dwconv(self):
        '''
        test the dfp dwconv function
        '''
        conv_input_path = os.path.join(self.dwconv_case_path, "dwconv_input.npy")
        conv_w_kernel_path = os.path.join(self.dwconv_case_path, "dwconv_weight.npy")
        conv_w_bias_path = os.path.join(self.dwconv_case_path, "dwconv_bias.npy")
        conv_out_path = os.path.join(self.dwconv_case_path, "dwconv_perlayer_output.npy")
        conv_info_path = os.path.join(self.dwconv_case_path, "dwconv_info.json")
        
        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))

        x_radix = conv_info_dict["x_radix"]
        y_radix = conv_info_dict["y_radix"]
        w_kernel_radix = conv_info_dict["w_kernel_radix"]
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]
        group = conv_info_dict["group"]

        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        kernel_bitwidth = self.config_json_dict["conv_bitwidth"]["kernel"]
        bias_bitwidth = self.config_json_dict["conv_bitwidth"]["bias"]
       
        w_bias_radix = y_radix
        w_kernel = np.round(w_kernel * 2 ** w_kernel_radix)
        w_bias = np.round(w_bias * 2 ** w_bias_radix)
        
        kernel_overflow = 2 ** (kernel_bitwidth - 1) - 1
        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]
        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1

        conv_out, _, _, _ = depthwise_conv(x, x_radix, w_kernel, w_kernel_radix, w_bias, group, y_radix, 
            working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1])
        
        self.assertTrue((conv_out == np.load(conv_out_path)).all())
    
    
    def test_bn(self):
        '''
        test the dfp bn function
        '''
        bn_input_path = os.path.join(self.bn_case_path, "bn_input.npy")
        bn_a_path = os.path.join(self.bn_case_path, "bn_gamma.npy")
        bn_b_path = os.path.join(self.bn_case_path, "bn_beta.npy")
        bn_mean_path = os.path.join(self.bn_case_path, "bn_mean.npy")
        bn_var_path = os.path.join(self.bn_case_path, "bn_var.npy")
        bn_out_path = os.path.join(self.bn_case_path, "bn_perlayer_output.npy")
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

        A_list, B_list = [], []
        for bn_channel in range(len(gamma)):
            A_list.append(gamma[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
            B_list.append(beta[bn_channel] - gamma[bn_channel]*mean[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
                
        x_radix = bn_info_dict["x_radix"]
        y_radix = bn_info_dict["y_radix"]
        a_radix = bn_info_dict["a_radix"]
       
        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        bn_bitwidth = self.config_json_dict["bn_bitwidth"]

        A_list = np.round(np.array(A_list) * 2 ** a_radix)
        B_list = np.round(np.array(B_list) * 2 ** y_radix)

        temp_shift = y_radix - x_radix - a_radix + 8

        SHIFT_LIST = [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]
        if temp_shift not in SHIFT_LIST:
            if temp_shift < -10:
                a_radix -= (-10 - temp_shift)
                A_list = np.round(np.array(A_list) / 2 ** (-10 - temp_shift))
            else:
                a_radix -= 1
                A_list = np.round(np.array(A_list) / 2 )

        A_overflow = 2 ** (bn_bitwidth - 1) - 1
        B_overflow = 2 ** (datapath_bitwidth - 1) -1
        A_list = np.array(A_list)
        B_list = np.array(B_list)
        A_list[np.where(A_list > A_overflow)] = A_overflow
        A_list[np.where(A_list < -A_overflow - 1)] = -A_overflow - 1
        B_list[np.where(B_list > B_overflow)] = B_overflow
        B_list[np.where(B_list < -B_overflow - 1)] = -B_overflow - 1
        bn_out, _, _, _ = BatchNormalization(x, x_radix, A_list, a_radix, B_list, y_radix, working_bitwidth, datapath_bitwidth)
        self.assertTrue((bn_out == np.load(bn_out_path)).all())
    
    def test_leakyrelu(self):
        '''
        test the dfp leakyrelu function
        '''
        leakyrelu_input_path = os.path.join(self.leakyrelu_case_path, "leakrely_input.npy")
        leakyrelu_out_path = os.path.join(self.leakyrelu_case_path, "leakyrelu_out.npy")
        leakyrelu_info_path = os.path.join(self.leakyrelu_case_path, "leakyrelu_info.json")
        
        f = open(leakyrelu_info_path, encoding = 'utf-8')
        leakyrelu_info_dict = json.load(f)
        f.close()
        x = np.load(leakyrelu_input_path)
        alpha = leakyrelu_info_dict["alpha"]
        x_radix = leakyrelu_info_dict["x_radix"]
        y_radix = leakyrelu_info_dict["y_radix"]
        alpha_radix = leakyrelu_info_dict["alpha_radix"]
        alpha = round(alpha * 2 ** alpha_radix)

        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        
        leakyrelu_out, _, _, _ = fpFunctions.LeakyRelu(x, x_radix, alpha, alpha_radix, y_radix, working_bitwidth, datapath_bitwidth)
        self.assertTrue((leakyrelu_out == np.load(leakyrelu_out_path)).all())
    
    def test_add(self):
        '''
        test the dfp add function
        '''
        add_input1_path = os.path.join(self.add_case_path, "add_input1.npy")
        add_input2_path = os.path.join(self.add_case_path, "add_input2.npy")
        
        add_out_path = os.path.join(self.add_case_path, "add_output.npy")
        add_info_path = os.path.join(self.add_case_path, "add_info.json")
        
        f = open(add_info_path, encoding = 'utf-8')
        add_info_dict = json.load(f)
        f.close()
        datapath_bitwidth = add_info_dict["datapath_bitwidth"]

        x_radix_list = add_info_dict["x_radix_list"]
        y_radix = add_info_dict["y_radix"]
        x_list = [np.load(add_input1_path), np.load(add_input2_path)]
        add_out, _, _ = add(x_list, x_radix_list, y_radix, datapath_bitwidth)
        self.assertTrue((add_out == np.load(add_out_path)).all())

    def test_upsample(self):
        '''
        test the dfp upsample funciton
        '''
        upsample_input_path = os.path.join(self.upsample_case_path, "upsample_input.npy")
        upsample_out_path = os.path.join(self.upsample_case_path, "upsample_out.npy")
        upsample_info_path = os.path.join(self.upsample_case_path, "upsample_info.json")
        upsample_scale_path = os.path.join(self.upsample_case_path, "up_sampling2d_1_scales.npy")
        
        f = open(upsample_info_path, encoding = 'utf-8')
        upsample_info_dict = json.load(f)
        f.close()
        datapath_bitwidth = upsample_info_dict["datapath_bitwidth"]

        x_radix = upsample_info_dict["x_radix"]
        y_radix = upsample_info_dict["y_radix"]
        mode = upsample_info_dict["mode"].encode()
        x = np.load(upsample_input_path)
        scales = np.load(upsample_scale_path)
        upsample_out = upsample(x, scales, mode)
        upsample_out = np.floor(upsample_out * 2 ** (y_radix - x_radix))
        self.assertTrue((upsample_out == np.load(upsample_out_path)).all())

    def test_concat(self):
        '''
        test the dfp concat function
        '''
        concat_input1_path = os.path.join(self.concat_case_path, "concat_input1.npy")
        concat_input2_path = os.path.join(self.concat_case_path, "concat_input2.npy")
        
        concat_out_path = os.path.join(self.concat_case_path, "concat_output.npy")
        concat_info_path = os.path.join(self.concat_case_path, "concat_info.json")
        
        f = open(concat_info_path, encoding = 'utf-8')
        concat_info_dict = json.load(f)
        f.close()
        datapath_bitwidth = concat_info_dict["datapath_bitwidth"]

        x_radix_list = concat_info_dict["x_radix_list"]
        y_radix = concat_info_dict["y_radix"]
        x_list = [np.load(concat_input1_path), np.load(concat_input2_path)]
        axis = concat_info_dict["axis"]
        concat_out, _, _ = concatenate(x_list, x_radix_list, y_radix, axis, datapath_bitwidth)
        self.assertTrue((concat_out == np.load(concat_out_path)).all())

    def test_conv_perchannel(self):
        '''
        test the dfp conv_perchannel function
        '''
        conv_input_path = os.path.join(self.conv_case_path, "conv2d_input.npy")
        conv_w_kernel_path = os.path.join(self.conv_case_path, "conv2d_weight.npy")
        conv_w_bias_path = os.path.join(self.conv_case_path, "conv2d_bias.npy")
        conv_out_path = os.path.join(self.conv_case_path, "conv2d_perchannel_output.npy")
        conv_info_path = os.path.join(self.conv_case_path, "conv_info.json")
        
        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))

        x_radix = conv_info_dict["x_radix"]
        y_radix = conv_info_dict["y_radix"]
        w_kernel_radix_list = conv_info_dict["w_kernel_radix_list"]
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]

        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        kernel_bitwidth = self.config_json_dict["conv_bitwidth"]["kernel"]
        bias_bitwidth = self.config_json_dict["conv_bitwidth"]["bias"]
       
        w_bias_radix = y_radix
        for c in range(w_kernel.shape[3]):
            w_kernel[:, :, :, c] = np.round(w_kernel[:, :, :, c] * 2.0 ** w_kernel_radix_list[c])

        w_bias = np.round(w_bias * 2 ** w_bias_radix)
        
        kernel_overflow = 2 ** (kernel_bitwidth - 1) - 1
        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]
        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1

        conv_out, _, _, _ = conv_by_channel(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False)
        self.assertTrue((conv_out == np.load(conv_out_path)).all())
    
    def test_dwconv_perchannel(self):
        '''
        test the dfp dwconv perchannel function
        '''
        conv_input_path = os.path.join(self.dwconv_case_path, "dwconv_input.npy")
        conv_w_kernel_path = os.path.join(self.dwconv_case_path, "dwconv_weight.npy")
        conv_w_bias_path = os.path.join(self.dwconv_case_path, "dwconv_bias.npy")
        conv_out_path = os.path.join(self.dwconv_case_path, "dwconv_perchannel_output.npy")
        conv_info_path = os.path.join(self.dwconv_case_path, "dwconv_info.json")
        
        f = open(conv_info_path, encoding = 'utf-8')
        conv_info_dict = json.load(f)
        f.close()
        x = np.load(conv_input_path)
        w_kernel = np.load(conv_w_kernel_path)
        dim_kernel = w_kernel.shape
        w_kernel = np.reshape(w_kernel, dim_kernel)
        w_kernel = w_kernel.transpose(2, 3, 1, 0)

        w_bias = np.load(conv_w_bias_path)
        dim_bias = w_bias.shape
        [dim_bias] = dim_bias
        w_bias = np.reshape(w_bias, (1, 1, 1, dim_bias))

        x_radix = conv_info_dict["x_radix"]
        y_radix = conv_info_dict["y_radix"]
        w_kernel_radix_list = conv_info_dict["w_kernel_radix_list"]
        stride = conv_info_dict["strides"]
        pad = conv_info_dict["pads"]
        dilation = conv_info_dict["dilations"]
        group = conv_info_dict["group"]

        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        kernel_bitwidth = self.config_json_dict["conv_bitwidth"]["kernel"]
        bias_bitwidth = self.config_json_dict["conv_bitwidth"]["bias"]
       
        w_bias_radix = y_radix
        for c in range(w_kernel.shape[3]):
            w_kernel[:, :, :, c] = np.round(w_kernel[:, :, :, c] * 2.0 ** w_kernel_radix_list[c])

        w_bias = np.round(w_bias * 2 ** w_bias_radix)
        
        kernel_overflow = 2 ** (kernel_bitwidth - 1) - 1
        kernel_overflow_time = np.where(w_kernel > kernel_overflow)[0].shape[0] + np.where(w_kernel < -kernel_overflow - 1)[0].shape[0]
        w_kernel[np.where(w_kernel > kernel_overflow)] = kernel_overflow
        w_kernel[np.where(w_kernel < -kernel_overflow - 1)] = -kernel_overflow - 1

        conv_out, _, _, _ = depth_wise_conv_by_channel(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, 
            working_bitwidth, datapath_bitwidth, bias_bitwidth,  group, stride, pad, dilation = [1, 1])
        
        self.assertTrue((conv_out == np.load(conv_out_path)).all())
        
    
    def test_bn_perchannel(self):
        '''
        test the dfp bn perchannel function
        '''
        bn_input_path = os.path.join(self.bn_case_path, "bn_input.npy")
        bn_a_path = os.path.join(self.bn_case_path, "bn_gamma.npy")
        bn_b_path = os.path.join(self.bn_case_path, "bn_beta.npy")
        bn_mean_path = os.path.join(self.bn_case_path, "bn_mean.npy")
        bn_var_path = os.path.join(self.bn_case_path, "bn_var.npy")
        bn_out_path = os.path.join(self.bn_case_path, "bn_perchannel_output.npy")
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

        A_list, B_list = [], []
        for bn_channel in range(len(gamma)):
            A_list.append(gamma[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
            B_list.append(beta[bn_channel] - gamma[bn_channel]*mean[bn_channel] / np.sqrt(var[bn_channel] + epsilon))
                
        x_radix = bn_info_dict["x_radix"]
        y_radix = bn_info_dict["y_radix"]
        a_radix_list = bn_info_dict["a_radix_list"]
        for k in range(len(A_list)):
            A_list[k] = np.round(A_list[k] * 2 ** a_radix_list[k])
        
        working_bitwidth = self.config_json_dict["working_bitwidth"]
        datapath_bitwidth = self.config_json_dict["datapath_bitwidth"]
        bn_bitwidth = self.config_json_dict["bn_bitwidth"]

        SHIFT_LIST = [-10, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10, 12]
        for k in range(len(A_list)):
            temp_shift = y_radix - x_radix - a_radix_list[k] + 8
            assert (temp_shift <= 12),  "temp_shift out of range"
            if temp_shift not in SHIFT_LIST:
                if temp_shift < -10:
                    a_radix_list[k] -= (-10 - temp_shift)
                    A_list[k] = np.round(A_list[k] / 2 ** (-10 - temp_shift))
                else:
                    a_radix_list[k] -= 1
                    A_list[k] = np.round(A_list[k] / 2 )

        B_list = np.round(np.array(B_list) * 2 ** y_radix)

        A_overflow = 2 ** (bn_bitwidth - 1) - 1
        B_overflow = 2 ** (datapath_bitwidth - 1) -1
        A_list = np.array(A_list)
        B_list = np.array(B_list)
        A_list[np.where(A_list > A_overflow)] = A_overflow
        A_list[np.where(A_list < -A_overflow - 1)] = -A_overflow - 1
        B_list[np.where(B_list > B_overflow)] = B_overflow
        B_list[np.where(B_list < -B_overflow - 1)] = -B_overflow - 1
        bn_out, _, _, _ = BN_per_channel(x, x_radix, A_list, a_radix_list, B_list, y_radix, working_bitwidth, datapath_bitwidth)
        self.assertTrue((bn_out == np.load(bn_out_path)).all())
    
    
if __name__ == '__main__': 
    unittest.main() 