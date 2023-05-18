import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np 
## accelerate by cuda using cupy
# import cupy as cp
import os
import matplotlib.pyplot as plt
import json
#import keras
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input,decode_predictions
import math
import logging
from .functions_fl import mish as floatmish
from .functions_fl import softplus as floatsoftplus
from .functions_fl import sigmoid as floatsigmoid
from .functions_fl import tanh as floattanh
from .functions_fl import silu as floatsilu

def statistic_overflow(overflow_time, total_cnt, overflow_th, data):
    overflow_info = {}
    overflow_info["overflow_time"] = overflow_time
    overflow_info["data_cnt"] = total_cnt
    overflow_info["overflow_th"] = overflow_th
    overflow_info["overflow_rate"] = overflow_time / total_cnt
    if data.size > 0:
        overflow_info["data_max"] = abs(data).max()
    return overflow_info

def read_directory(directory_name):
    file_path = []
    for filename in os.listdir(directory_name):
        filepath = os.path.join(directory_name, filename)
        file_path.append(filepath)
    
    return file_path

def getradix(xmax, bitwidth):
    if xmax == 0: 
        return 0
    radix = bitwidth - 1 - (math.floor(math.log2(xmax) + 1))
    return radix

def GetRadixList(channel_max_list, channel_min_list, bitwidth):

    bitwidth -= 1
    assert len(channel_max_list) == len(channel_min_list)
    SFT_Bit_List = []
    for i in range(len(channel_max_list)):
        channel_max = max(abs(channel_max_list[i]), abs(channel_min_list[i]))
        SFT_Bit = bitwidth - (math.floor(math.log2(channel_max)+1))
        SFT_Bit_List.append(SFT_Bit)
    return SFT_Bit_List


def hardware_sum(conv_box, w_kernel, calc_overflow, calc_overflow_time):
    conv_out = 0
    conv_x = 0
    C = conv_box.shape[2]
    step = 32
    group_num = C // step
    for i in range(group_num):
        conv_x = np.sum(conv_box[:, :, i * step:(i + 1) * step] * w_kernel[:, :, i * step:(i + 1) * step])
        if conv_x > calc_overflow:
            conv_x = calc_overflow
            calc_overflow_time += 1
        elif conv_x < -calc_overflow - 1:
            conv_x = -calc_overflow - 1
            calc_overflow_time += 1
        conv_out += conv_x
        if conv_out > calc_overflow:
            conv_out = calc_overflow
            calc_overflow_time += 1
        elif conv_out < -calc_overflow - 1:
            conv_out = -calc_overflow - 1
            calc_overflow_time += 1
    conv_out += np.sum(conv_box[:, :, group_num * step:] * w_kernel[:, :, group_num * step:])
    if conv_out > calc_overflow:
        conv_out = calc_overflow
        calc_overflow_time += 1
    elif conv_out < -calc_overflow - 1:
        conv_out = -calc_overflow - 1
        calc_overflow_time += 1
    return conv_out, calc_overflow_time


def conv_normal(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False):
    x = x.transpose(0,2,3,1)
    w_kernel = w_kernel.transpose(2,3,1,0)
    (m, Height_prev, Width_prev, C_prev) = np.shape(x)
    #Retrieve dimensions from W's shape
    (f, f, C_prev, C) = w_kernel.shape
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    for c in range(C):
        if w_bias[0, 0, 0, c] > bias_overflow:
            w_bias[0, 0, 0, c] = bias_overflow
            bias_overflow_time += 1
        elif w_bias[0, 0, 0, c] < -bias_overflow - 1:
            w_bias[0, 0, 0, c] = -bias_overflow - 1
            bias_overflow_time += 1
    if bias_overflow_time > 0:
        logging.warning('bias overflow: {}/{}'.format(bias_overflow_time, C))
    
    #Retrieve information of stride and pad based on keras' vgg16
    [stride1, stride2] = stride
    [pad1, pad2, pad3, pad4] = pad
    [dilation1, dilation2] = dilation
    # Compute the dimensions of the CONV output volume
    Height = int((Height_prev - f + pad1 + pad3) / stride1) + 1
    Width = int((Width_prev - f + pad2 + pad4) / stride2) + 1
    # Initialize the output volume Z with zeros.
    conv_out = np.zeros((m, Height, Width, C))
    # pad the input data(input_img) with zeros
    img_padded = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
                        'constant', constant_values=0)
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1
    #convolution
    for i in range(m):
        image_in = img_padded[i]
        for h in range(Height):                  # loop over vertical axis of the output volume
            for w in range(Width):               # loop over horizontal axis of the output volume
                for c in range(C):               # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    h_start = h * stride1
                    h_end = h * stride1 + f
                    w_start = w * stride2
                    w_end = w * stride2 + f
                    # Use the corners to define the (3D) slice of image_in
                    conv_box = image_in[h_start:h_end, w_start:w_end, :]
                    #print('conv_box size:{}'.format(conv_box.shape))
                    if is_hardware == True:
                        conv_out[i, h, w, c], calc_overflow_time = hardware_sum(conv_box, w_kernel[:, :, :, c], calc_overflow, calc_overflow_time)
                    else:
                        conv_out[i, h, w, c] = np.sum(conv_box * w_kernel[:, :, :, c])
                        if conv_out[i, h, w, c] > calc_overflow:
                            conv_out[i, h, w, c] = calc_overflow
                            calc_overflow_time += 1
                        elif conv_out[i, h, w, c] < -calc_overflow - 1:
                            conv_out[i, h, w, c] = -calc_overflow - 1
                            calc_overflow_time += 1
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time, m * Height * Width * C))
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    output_shift = y_radix - (x_radix + w_kernel_radix)
    conv_out = np.floor(conv_out * 2 ** output_shift)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    for i in range(m):
        for h in range(Height):
            for w in range(Width):
                for c in range(C):
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
                    conv_out[i, h, w, c] += w_bias[0, 0, 0, c]
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, m * Height * Width * C))
    
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix, w_bias_radix, y_radix))                 
    conv_out = conv_out.transpose(0,3,1,2)
    return conv_out, y_radix



def im2col_cpu(input_data, filter_h, filter_w, stride, pad):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    out_h = (H + pad1 + pad3 - filter_h)//stride1 + 1#输出图的计算公式
    out_w = (W + pad2 + pad4 - filter_w)//stride2 + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad1, pad3), (pad2, pad4)], 'constant')
    #在通道的之后维度，长和宽的纬度进行填充
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype="float32")

    for y in range(filter_h):
        y_max = y + stride1*out_h
        for x in range(filter_w):
            x_max = x + stride2*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride1, x:x_max:stride2]#

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #-1表示第二个维度需要程序进行推理，即总个数除以N*out_h*out_w
    return col

def conv_cpu(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False):
    
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)

    
    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1

    if bias_overflow_time > 0:
        logging.warning('bias overflow: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=FN, overflow_th=bias_overflow, data=statis_data)

    # 利用im2col转换为行
    col = im2col_cpu(x, FH, FW, stride, pad)
    # 卷积核转换为列，展开为2维数组
    col_W = w_kernel.reshape(FN, -1).T
    # 计算正向传播

    if is_hardware == True:
        pass
    else:
        conv_out = np.dot(col, col_W)

    # calc_overflow_time conv_out
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
    conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
    
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    output_shift = y_radix - (x_radix + w_kernel_radix)
    conv_out = np.floor(conv_out * 2 ** output_shift)

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1

    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    if len(w_bias) != 0: ## change by mengxiao @ 2021-01-27
        conv_out = conv_out + w_bias
    conv_out = conv_out.reshape(N, out_h, out_w, -1)
    statis_data = conv_out.copy()
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=datapath_overflow, data=statis_data)
    
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix, w_bias_radix, y_radix))                 
    conv_out = conv_out.transpose(0,3,1,2)
    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def im2col_gpu(input_data, filter_h, filter_w, stride, pad):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    out_h = (H + pad1 + pad3 - filter_h)//stride1 + 1#输出图的计算公式
    out_w = (W + pad2 + pad4 - filter_w)//stride2 + 1

    img = cp.pad(input_data, [(0,0), (0,0), (pad1, pad3), (pad2, pad4)], 'constant')
    #在通道的之后维度，长和宽的纬度进行填充
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype="float32")

    for y in range(filter_h):
        y_max = y + stride1*out_h
        for x in range(filter_w):
            x_max = x + stride2*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride1, x:x_max:stride2]#

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #-1表示第二个维度需要程序进行推理，即总个数除以N*out_h*out_w
    return col

def conv_gpu(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False):
    print("conv by gpu start, mengxiao")
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)
    
    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    
    bias_overflow_time = cp.where(w_bias > bias_overflow)[0].shape[0] + cp.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[cp.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[cp.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1

    if bias_overflow_time > 0:
        logging.warning('bias overflow: {}/{}'.format(bias_overflow_time, FN))
    ####

    # 利用im2col转换为行
    col = im2col_gpu(x, FH, FW, stride, pad)
    # 卷积核转换为列，展开为2维数组
    col_W = w_kernel.reshape(FN, -1).T
    # 计算正向传播

    if is_hardware == True:
        pass
    else:
        conv_out = cp.dot(col, col_W)

    # calc_overflow_time conv_out
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1

    calc_overflow_time = cp.where(conv_out > calc_overflow)[0].shape[0] + cp.where(conv_out < -calc_overflow - 1)[0].shape[0]
    conv_out[cp.where(conv_out > calc_overflow)] = calc_overflow
    conv_out[cp.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
    ####
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * C))
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    output_shift = y_radix - (x_radix + w_kernel_radix)
    #need to change to floor
    conv_out = cp.around(conv_out * 2 ** output_shift)

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1

    datapath_overflow_time = cp.where(conv_out > datapath_overflow)[0].shape[0] + cp.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[cp.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[cp.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    conv_out = conv_out + w_bias
    conv_out = conv_out.reshape(N, out_h, out_w, -1)
    
    datapath_overflow_time += cp.where(conv_out > datapath_overflow)[0].shape[0] + cp.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[cp.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[cp.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1

    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * C))
    
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix, w_bias_radix, y_radix))                 
    
    # print("conv by gpu end, mengxiao")
    conv_out = conv_out.transpose(0,3,1,2)
    return conv_out, y_radix

def depthwise_conv(input, x_radix, w_kernel, w_kernel_radix, w_bias, group, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1]):
    """Two-dimensional depthwise convolution dfp-perlayer.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    input = input.transpose(0,2,3,1)
    assert (group == input.shape[-1]), "Depth_Wise_Conv Group Error! This function do not support."
    w_kernel = w_kernel.transpose(2,3,0,1)

    (m, Height_prev, Width_prev, C_prev) = np.shape(input)
    # print("C_prev: ", C_prev)
    (f, f, C, C_prev) = w_kernel.shape
    # print("C_prev: ", C_prev)
    [stride1, stride2] = stride
    [pad1, pad2, pad3, pad4] = pad

    Height = int((Height_prev - f + pad1 + pad3) / stride1) + 1
    Width = int((Width_prev - f + pad2 + pad4) / stride2) + 1
    conv_out = np.zeros((m, Height, Width, C))

    img_padded = np.pad(input, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
                        'constant', constant_values=0)

    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    for c in range(C):
        if w_bias[0, 0, 0, c] > bias_overflow:
            w_bias[0, 0, 0, c] = bias_overflow
            bias_overflow_time += 1
        elif w_bias[0, 0, 0, c] < -bias_overflow - 1:
            w_bias[0, 0, 0, c] = -bias_overflow - 1
            bias_overflow_time += 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow: {}/{}'.format(bias_overflow_time, C))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1

    #convolution
    for i in range(m):
        image_in = img_padded[i]
        for h in range(Height):                  # loop over vertical axis of the output volume
            for w in range(Width):               # loop over horizontal axis of the output volume
                for c in range(C):               # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    h_start = h * stride1
                    h_end = h * stride1 + f
                    w_start = w * stride2
                    w_end = w * stride2 + f
                    # Use the corners to define the (3D) slice of image_in
                    conv_box = image_in[h_start:h_end, w_start:w_end, c]
                    
                    ######
                    weig = np.reshape(w_kernel[: ,:, c, :], (f, f))
                    conv_out[i, h, w, c] = np.sum(conv_box * weig) 
                    #conv_out[i, h, w, c] = np.sum(conv_box * weig) 

    statis_data = conv_out.copy()
    calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
    conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1

    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=m * Height * Width * c, overflow_th=calc_overflow, data=statis_data)

    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    output_shift = y_radix - (x_radix + w_kernel_radix)
    conv_out = np.floor(conv_out * 2 ** output_shift)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    for i in range(m):
        for h in range(Height):
            for w in range(Width):
                for c in range(C):
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
                    conv_out[i, h, w, c] += w_bias[0, 0, 0, c]
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=m * Height * Width * C, overflow_th=calc_overflow, data=statis_data)
    
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix, w_bias_radix, y_radix))
    conv_out = conv_out.transpose(0,3,1,2)
    
    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def conv(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False, option='img2col'):
    if option == 'normal':
        # print("convolution layer by normal")
        return conv_normal(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False)

    elif option == 'img2col':
        return conv_cpu(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False)
    
    elif option == 'cuda':
        x_gpu = cp.asarray(x)  # move the data to the device 
        w_kernel_gpu = cp.asarray(w_kernel)
        w_bias_gpu = cp.asarray(w_bias)
        [conv_out_gpu, y_radix] = conv_gpu(x_gpu, x_radix, w_kernel_gpu, w_kernel_radix, w_bias_gpu, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False)
        return cp.asnumpy(conv_out_gpu), y_radix
    ##TODO: add function_hw.conv_n900
    # elif option == 'hardware':
    #     return function_hw.conv_n900(x, x_radix, w_kernel, w_kernel_radix, w_bias, y_radix, stride, pad)

    else:
        raise ValueError("please fill the right option: (normal/img2col/cuda/hardware)")

def conv_by_channel(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], is_hardware = False):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)

    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow time: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)

    # 利用im2col转换为行
    col = im2col_cpu(x, FH, FW, stride, pad)
    # 卷积核转换为列，展开为2维数组
    col_W = w_kernel.reshape(FN, -1).T
    # 计算正向传播

    if is_hardware == True:
        pass
    else:
        conv_out = np.dot(col, col_W)

    
    # calc_overflow_time conv_out
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
    conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
    
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('calculation overflow time: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    for c in range(FN):
        temp_shift = y_radix - x_radix - w_kernel_radix_list[c]
        #print(temp_shift)
        conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** temp_shift)
    
    w_bias_radix = y_radix

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    if len(w_bias) != 0:
        conv_out = conv_out + w_bias
    conv_out = conv_out.reshape(N, out_h, out_w, -1)
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow time: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)
   
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))                 
    conv_out = conv_out.transpose(0,3,1,2)

    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def conv_by_channel_iee(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], node_name='', is_hardware = False, iee_version = 'n900'):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    if dilation != [1, 1]:
        return conv_by_channel_iee_dilated(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = dilation, node_name=node_name, is_hardware = False, iee_version = iee_version)
    elif (FH == 3 and FW == 3) or (FH == 1 and FW == 1):
        return conv_by_channel_iee_1x1_3x3(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = dilation, node_name=node_name, is_hardware = False, iee_version = iee_version)
    else:
        return conv_by_channel_iee_other(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = dilation, node_name=node_name, is_hardware = False, iee_version = iee_version)

def conv_by_channel_iee_1x1_3x3_im2col(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], node_name='', is_hardware = False, iee_version = 'n900'):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)

    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow time: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    
    if iee_version == 'n901':
        if C <= 4:
            per_ch = 2
        else:
            per_ch = 16
    elif iee_version == 'n900':
        per_ch = 32
    ch_num = int(math.ceil(C/per_ch)) # divide C into ch_num * per_ch
    partial_sum = np.zeros((out_w * out_h, FN))
    for curr_ch in range(ch_num):
        conv_out = 0
        if curr_ch == ch_num - 1:
            data = x[:, per_ch*curr_ch : C+1, :, :]
            weight = w_kernel[:, per_ch*curr_ch : C+1, :, :]
        else:
            data = x[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
            weight = w_kernel[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
        # 利用im2col转换为行
        col = im2col_cpu(data, FH, FW, stride, pad)
        # 卷积核转换为列，展开为2维数组
        col_W = weight.reshape(FN, -1).T
        # 计算正向传播
        if is_hardware == True:
            pass
        else:
            conv_out = np.dot(col, col_W)
        # calc_overflow_time conv_out
        calc_overflow_time = 0
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = conv_out.copy()
        calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
        conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
        conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
        #coarse shift
        for c in range(FN):
            temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
            if temp_shift > 35 or temp_shift < -4:
                print('temp_shift out of range')
            if temp_shift > 15:
                coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
            elif temp_shift < 0:
                coarse_shift = -4
            else:
                coarse_shift = 0
            fine_shift = temp_shift-coarse_shift
            conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-coarse_shift))
        # calc_overflow_time conv_out
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = conv_out.copy()
        calc_overflow_time += np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
        conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
        conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
        # sum
        partial_sum += conv_out
        # calc_overflow_time conv_out
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = partial_sum.copy()
        calc_overflow_time += np.where(partial_sum > calc_overflow)[0].shape[0] + np.where(partial_sum< -calc_overflow - 1)[0].shape[0]
        partial_sum[np.where(partial_sum > calc_overflow)] = calc_overflow
        partial_sum[np.where(partial_sum < -calc_overflow - 1)] = -calc_overflow - 1
        
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('calculation overflow time: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    conv_out = partial_sum
    #fine shift
    for c in range(FN):
        temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
        if temp_shift > 35 or temp_shift < -4:
            print('temp_shift out of range')
        if temp_shift > 15:
            coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
        elif temp_shift < 0:
            coarse_shift = -4
        else:
            coarse_shift = 0
        fine_shift = temp_shift-coarse_shift
        conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-fine_shift))

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    
    if len(w_bias) != 0:
        conv_out = conv_out + w_bias
    conv_out = conv_out.reshape(N, out_h, out_w, -1)
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow time: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)
   
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))                 
    conv_out = conv_out.transpose(0,3,1,2)

    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def conv_by_channel_iee_1x1_3x3(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], node_name='', is_hardware = False, iee_version = 'n900'):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)
    #data crop and pad
    x_paded = np.pad(x, ((0, 0), (0, 0), (pad1, pad3), (pad2, pad4)), 'constant', constant_values=0)
    data_in = x_paded[:, :, 0:(out_h-1)*stride1+FH, 0:(out_w-1)*stride2+FW]
    data_in_paded = np.pad(data_in, ((0, 0), (0, 0), (0, 2), (0, 2)), 'constant', constant_values=0)
    weight_paded = np.pad(w_kernel, ((0, 0), (0, 0), (0, 2), (0, 2)), 'constant', constant_values=0)

    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow time: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    
    if iee_version == 'n901':
        if C <= 4 and FH == 3 and FW == 3:
            per_ch = 2
        else:
            per_ch = 16
    elif iee_version == 'n900':
        per_ch = 32
    ch_num = int(math.ceil(C/per_ch)) # divide C into ch_num * per_ch
    #partial_sum = np.zeros((out_w * out_h, FN))    #img2col
    partial_sum = np.zeros((N, FN, out_h, out_w))

    for curr_ch in range(ch_num):
        conv_out = 0
        if curr_ch == ch_num - 1:
            data = data_in_paded[:, per_ch*curr_ch : C+1, :, :]
            weight = weight_paded[:, per_ch*curr_ch : C+1, :, :]
        else:
            data = data_in_paded[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
            weight = weight_paded[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
        # # 利用im2col转换为行
        # #col = im2col_cpu(data, FH, FW, stride, pad)
        # col = im2col_cpu(data, 3, 3, stride, pad)
        # # 卷积核转换为列，展开为2维数组
        # col_W = weight.reshape(FN, -1).T
        # # 计算正向传播
        # if is_hardware == True:
        #     pass
        # else:
        #     conv_out = np.dot(col, col_W)
        conv_out = cu_n900_without_c_shift(data, weight, stride)
        # calc_overflow_time conv_out
        calc_overflow_time = 0
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = conv_out.copy()
        calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
        conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
        conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
        #coarse shift
        for c in range(FN):
            temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
            if temp_shift > 35 or temp_shift < -4:
                print('temp_shift out of range')
            if temp_shift > 15:
                coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
            elif temp_shift < 0:
                coarse_shift = -4
            else:
                coarse_shift = 0
            fine_shift = temp_shift-coarse_shift
            conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-coarse_shift))
        # calc_overflow_time conv_out
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = conv_out.copy()
        calc_overflow_time += np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
        conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
        conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
        # sum
        partial_sum += conv_out
        # calc_overflow_time conv_out
        calc_overflow = 2 ** (working_bitwidth - 1) - 1
        statis_data = partial_sum.copy()
        calc_overflow_time += np.where(partial_sum > calc_overflow)[0].shape[0] + np.where(partial_sum< -calc_overflow - 1)[0].shape[0]
        partial_sum[np.where(partial_sum > calc_overflow)] = calc_overflow
        partial_sum[np.where(partial_sum < -calc_overflow - 1)] = -calc_overflow - 1
        
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('calculation overflow time: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    conv_out = partial_sum
    #fine shift
    for c in range(FN):
        temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
        if temp_shift > 35 or temp_shift < -4:
            print('temp_shift out of range')
        if temp_shift > 15:
            coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
        elif temp_shift < 0:
            coarse_shift = -4
        else:
            coarse_shift = 0
        fine_shift = temp_shift-coarse_shift
        conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-fine_shift))

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    
    if len(w_bias) != 0:
        w_bias = w_bias.transpose(0,3,1,2)
        conv_out = conv_out + w_bias
    #conv_out = conv_out.reshape(N, out_h, out_w, -1)
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow time: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)
   
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))                 
    #conv_out = conv_out.transpose(0,3,1,2)

    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def conv_by_channel_iee_other(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], node_name='', is_hardware = False, iee_version = 'n900'):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    # 计算输出数据大小
    out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)
    #data crop and pad
    x_paded = np.pad(x, ((0, 0), (0, 0), (pad1, pad3), (pad2, pad4)), 'constant', constant_values=0)
    data_in = x_paded[:, :, 0:(out_h-1)*stride1+FH, 0:(out_w-1)*stride2+FW]
    data_in_paded = np.pad(data_in, ((0, 0), (0, 0), (0, 2), (0, 2)), 'constant', constant_values=0)
    weight_paded = np.pad(w_kernel, ((0, 0), (0, 0), (0, 2), (0, 2)), 'constant', constant_values=0)
    kernel_row_num = math.ceil(FH/3)#row num of 3x3 conv
    kernel_col_num = math.ceil(FW/3)#col num of 3x3 conv

    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow time: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    
    if iee_version == 'n901':
        per_ch = 16
    elif iee_version == 'n900':
        per_ch = 32
    ch_num = int(math.ceil(C/per_ch)) # divide C into ch_num * per_ch
    #partial_sum = np.zeros((out_w * out_h, FN))    #img2col
    partial_sum = np.zeros((N, FN, out_h, out_w))

    for k_r in range(kernel_row_num):
        for k_c in range(kernel_col_num):
            #3x3 conv
            w_r_st = 3 * k_r
            w_r_ed = 3 * k_r + 3
            w_c_st = 3 * k_c
            w_c_ed = 3 * k_c + 3
            d_r_st = w_r_st
            d_r_ed = d_r_st + (out_h - 1) * stride1 + 3
            d_c_st = w_c_st
            d_c_ed = d_c_st + (out_w - 1) * stride2 + 3
            data_3x3 = data_in_paded[:, :, d_r_st:d_r_ed, d_c_st:d_c_ed]
            weight_3x3 = weight_paded[:, :, w_r_st:w_r_ed, w_c_st:w_c_ed]
            for curr_ch in range(ch_num):
                conv_out = 0
                if curr_ch == ch_num - 1:
                    data = data_3x3[:, per_ch*curr_ch : C+1, :, :]
                    weight = weight_3x3[:, per_ch*curr_ch : C+1, :, :]
                else:
                    data = data_3x3[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
                    weight = weight_3x3[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
                # # 利用im2col转换为行
                # #col = im2col_cpu(data, FH, FW, stride, pad)
                # col = im2col_cpu(data, 3, 3, stride, pad)
                # # 卷积核转换为列，展开为2维数组
                # col_W = weight.reshape(FN, -1).T
                # # 计算正向传播
                # if is_hardware == True:
                #     pass
                # else:
                #     conv_out = np.dot(col, col_W)
                conv_out = cu_n900_without_c_shift(data, weight, stride)
                # calc_overflow_time conv_out
                calc_overflow_time = 0
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = conv_out.copy()
                calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
                conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
                conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
                #coarse shift
                for c in range(FN):
                    temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
                    if temp_shift > 35 or temp_shift < -4:
                        print('temp_shift out of range')
                    if temp_shift > 15:
                        coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
                    elif temp_shift < 0:
                        coarse_shift = -4
                    else:
                        coarse_shift = 0
                    fine_shift = temp_shift-coarse_shift
                    conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-coarse_shift))
                # calc_overflow_time conv_out
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = conv_out.copy()
                calc_overflow_time += np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
                conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
                conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
                # sum
                partial_sum += conv_out
                # calc_overflow_time conv_out
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = partial_sum.copy()
                calc_overflow_time += np.where(partial_sum > calc_overflow)[0].shape[0] + np.where(partial_sum< -calc_overflow - 1)[0].shape[0]
                partial_sum[np.where(partial_sum > calc_overflow)] = calc_overflow
                partial_sum[np.where(partial_sum < -calc_overflow - 1)] = -calc_overflow - 1
        
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('calculation overflow time: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    conv_out = partial_sum
    #fine shift
    for c in range(FN):
        temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
        if temp_shift > 35 or temp_shift < -4:
            print('temp_shift out of range')
        if temp_shift > 15:
            coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
        elif temp_shift < 0:
            coarse_shift = -4
        else:
            coarse_shift = 0
        fine_shift = temp_shift-coarse_shift
        conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-fine_shift))

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    
    if len(w_bias) != 0:
        w_bias = w_bias.transpose(0,3,1,2)
        conv_out = conv_out + w_bias
    #conv_out = conv_out.reshape(N, out_h, out_w, -1)
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow time: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)
   
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))                 
    #conv_out = conv_out.transpose(0,3,1,2)

    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def conv_by_channel_iee_dilated(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, stride, pad, dilation = [1, 1], node_name='', is_hardware = False, iee_version = 'n900'):
    # 卷积核大小
    FN, C, FH, FW = w_kernel.shape
    # 数据数据大小
    N, C, H, W = x.shape
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    [dilation1, dilation2] = dilation
    dilated_f1 = FH + (FH - 1) * (dilation1 - 1)
    dilated_f2 = FW + (FW - 1) * (dilation2 - 1)
    out_h = int((H - dilated_f1 + pad1 + pad3) / stride1) + 1
    out_w = int((W - dilated_f2 + pad2 + pad4) / stride2) + 1
    # 计算输出数据大小
    # out_h = 1 + int((H + pad1 + pad3 - FH) / stride1)
    # out_w = 1 + int((W + pad2 + pad4 - FW) / stride2)
    #data crop and pad
    x_paded = np.pad(x, ((0, 0), (0, 0), (pad1, pad3), (pad2, pad4)), 'constant', constant_values=0)

    # bias_overflow_time w_bias
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    bias_overflow_time = np.where(w_bias > bias_overflow)[0].shape[0] + np.where(w_bias < -bias_overflow - 1)[0].shape[0]
    w_bias[np.where(w_bias > bias_overflow)] = bias_overflow
    w_bias[np.where(w_bias < -bias_overflow - 1)] = -bias_overflow - 1
    
    bias_overflow_info = {}
    if bias_overflow_time > 0:
        logging.warning('bias overflow time: {}/{}'.format(bias_overflow_time, FN))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    
    if iee_version == 'n901':
        per_ch = 16
    elif iee_version == 'n900':
        per_ch = 32
    ch_num = int(math.ceil(C/per_ch)) # divide C into ch_num * per_ch
    #partial_sum = np.zeros((out_w * out_h, FN))    #img2col
    #pdb.set_trace()
    partial_sum = np.zeros((N, FN, out_h, out_w))

    for k_r in range(FH):
        for k_c in range(FW):
            ## get weight crop
            w_r_st = k_r
            w_r_ed = w_r_st + 1
            w_c_st = k_c
            w_c_ed = k_c + 1
            ## get data crop
            d_r_st = dilation1 * k_r
            d_r_ed = d_r_st + (out_h - 1) * stride1 + 1
            d_c_st = dilation2 * k_c
            d_c_ed = d_c_st + (out_w - 1) * stride2 + 1
            data_ = x_paded[:, :, d_r_st:d_r_ed, d_c_st:d_c_ed]
            weight_ = w_kernel[:, :, w_r_st: w_r_ed, w_c_st: w_c_ed]
            for curr_ch in range(ch_num):
                conv_out = 0
                if curr_ch == ch_num - 1:
                    data = data_[:, per_ch*curr_ch : C+1, :, :]
                    weight = weight_[:, per_ch*curr_ch : C+1, :, :]
                else:
                    data = data_[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
                    weight = weight_[:, per_ch*curr_ch:per_ch*curr_ch+per_ch, :, :]
                # # 利用im2col转换为行
                # #col = im2col_cpu(data, FH, FW, stride, pad)
                # col = im2col_cpu(data, 3, 3, stride, pad)
                # # 卷积核转换为列，展开为2维数组
                # col_W = weight.reshape(FN, -1).T
                # # 计算正向传播
                # if is_hardware == True:
                #     pass
                # else:
                #     conv_out = np.dot(col, col_W)
                conv_out = cu_n900_without_c_shift(data, weight, stride)
                # calc_overflow_time conv_out
                calc_overflow_time = 0
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = conv_out.copy()
                calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
                conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
                conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
                #coarse shift
                for c in range(FN):
                    temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
                    if temp_shift > 35 or temp_shift < -4:
                        print('temp_shift out of range')
                    if temp_shift > 15:
                        coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
                    elif temp_shift < 0:
                        coarse_shift = -4
                    else:
                        coarse_shift = 0
                    fine_shift = temp_shift-coarse_shift
                    conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-coarse_shift))
                # calc_overflow_time conv_out
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = conv_out.copy()
                calc_overflow_time += np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
                conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
                conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
                # sum
                partial_sum += conv_out
                # calc_overflow_time conv_out
                calc_overflow = 2 ** (working_bitwidth - 1) - 1
                statis_data = partial_sum.copy()
                calc_overflow_time += np.where(partial_sum > calc_overflow)[0].shape[0] + np.where(partial_sum< -calc_overflow - 1)[0].shape[0]
                partial_sum[np.where(partial_sum > calc_overflow)] = calc_overflow
                partial_sum[np.where(partial_sum < -calc_overflow - 1)] = -calc_overflow - 1
        
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('calculation overflow time: {}/{}'.format(calc_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)

    conv_out = partial_sum
    #fine shift
    for c in range(FN):
        temp_shift = x_radix + w_kernel_radix_list[c] - y_radix
        if temp_shift > 35 or temp_shift < -4:
            print('temp_shift out of range')
        if temp_shift > 15:
            coarse_shift = 4*math.floor((temp_shift - 16)/4)+4
        elif temp_shift < 0:
            coarse_shift = -4
        else:
            coarse_shift = 0
        fine_shift = temp_shift-coarse_shift
        conv_out[:, c] = np.floor(conv_out[:, c] * 2.0 ** (-fine_shift))

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    datapath_overflow_time = np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    
    if len(w_bias) != 0:
        w_bias = w_bias.transpose(0,3,1,2)
        conv_out = conv_out + w_bias
    #conv_out = conv_out.reshape(N, out_h, out_w, -1)
    datapath_overflow_time += np.where(conv_out > datapath_overflow)[0].shape[0] + np.where(conv_out < -datapath_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > datapath_overflow)] = datapath_overflow
    conv_out[np.where(conv_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow time: {}/{}'.format(datapath_overflow_time, N * FN * out_h * out_w))##m * Height * Width * c))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(N * FN * out_h * out_w), overflow_th=calc_overflow, data=statis_data)
   
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))                 
    #conv_out = conv_out.transpose(0,3,1,2)

    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info

def cu_n900_without_c_shift(x, w_kernel, stride):
    (m, C_prev, Height_prev, Width_prev) = np.shape(x)
    (C, C_prev, f1, f2) = w_kernel.shape
    [stride1, stride2] = stride
    Height = int((Height_prev - f1) / stride1) + 1
    Width = int((Width_prev - f2) / stride2) + 1
    # Initialize the output volume with zeros.
    conv_out = np.zeros((m, C, Height, Width))
    for i in range(m):
        image_in = x[i]
        for h in range(Height):                  # loop over vertical axis of the output volume
            for w in range(Width):               # loop over horizontal axis of the output volume
                for c in range(C):               # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    h_start = h * stride1
                    h_end = h * stride1 + f1
                    w_start = w * stride2
                    w_end = w * stride2 + f2
                    # Use the corners to define the (3D) slice of image_in
                    conv_box = image_in[:, h_start:h_end, w_start:w_end]
                    #print('conv_box size:{}'.format(conv_box.shape))
                    conv_out[i, c, h, w] = np.sum(conv_box * w_kernel[c, :, :, :])
    return conv_out

def relu(x):
    x = np.where(x < 0, 0., x)
    return x

def PRelu(x, x_radix, alpha, alpha_radix, y_radix, working_bitwidth, datapath_bitwidth):
    x = x.transpose(0,2,3,1)
    if x_radix != y_radix:
        logging.error('prelu datapath error!')
    (m, Height, Width, C) = np.shape(x)
    
    data_out = np.zeros((m, Height, Width, C))
    for i in range(m):
        for ch in range(C):
            for row in range(Height):
                for col in range(Width):
                    if(x[i, row, col, ch] >= 0):
                        data_out[i, row, col, ch] = x[i, row, col, ch]
                    else:
                        data_out[i, row, col, ch] = x[i, row, col, ch] * alpha[ch]
                        # product = math.floor(product/2)
                        # data_out[i, row, col, ch] = math.floor(product * 2 ** (-17))

    overflow_time = 0
    overflow = 2 ** (working_bitwidth - 1) - 1
    statis_data = data_out.copy()
    overflow_time = np.where(data_out > overflow)[0].shape[0] + np.where(data_out < -overflow - 1)[0].shape[0]
    data_out[np.where(data_out > overflow)] = overflow
    data_out[np.where(data_out < -overflow - 1)] = -overflow - 1

    calc_overflow_info = {}
    if overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * C * Height * Width), overflow_th=overflow, data=statis_data)
    
    output_shift = -alpha_radix
    data_out[np.where(x < 0)] = np.floor(data_out[np.where(x < 0)] * 2 ** output_shift)
    # data_out[np.where(data_out < 0)] = np.floor(data_out[np.where(data_out < 0)] * 2 ** output_shift)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = data_out.copy()
    datapath_overflow_time = np.where(data_out > datapath_overflow)[0].shape[0] + np.where(data_out < -datapath_overflow - 1)[0].shape[0]
    data_out[np.where(data_out > datapath_overflow)] = datapath_overflow
    data_out[np.where(data_out < -datapath_overflow - 1)] = -datapath_overflow - 1
    
    datapath_overflow_info={}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(m * C * Height * Width), overflow_th=datapath_overflow, data=statis_data)
    
    logging.info('input_radix:{}, alpha_radix:{}, output_radix:{}'.format(x_radix, alpha_radix, y_radix))                 
    data_out = data_out.transpose(0,3,1,2)
    return data_out, calc_overflow_info, datapath_overflow_info

def clip(x, x_min, x_max):
    y = np.clip(x, x_min, x_max)
    return y

def averagepool(x, alpha_radix, kernel_size, pad, stride):
    x = x.transpose(0,2,3,1)
    [kernel_size1, kernel_size2] = kernel_size
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    (m, Height_prev, Width_prev, C) = x.shape
    Height = int((Height_prev - kernel_size1 + pad1 + pad3)/stride1) + 1
    Width = int((Width_prev - kernel_size2 + pad2 + pad4)/stride2) + 1
    y = np.zeros((m, Height, Width, C))
    x = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
        'constant', constant_values = 0)
    for i in range(m):
        for h in range(Height):
            for w in range(Width):
                for c in range(C):
                    h_start = h * stride1
                    h_end = h * stride1 + kernel_size1
                    w_start = w * stride2
                    w_end = w * stride2 + kernel_size2
                    poolField = x[0, h_start:h_end, w_start:w_end, c]
                    poolOut = np.sum(poolField)
                    len1, len2 = kernel_size1, kernel_size2
                    if h == 0:
                        len1 = kernel_size1 - pad1
                    elif h == Height - 1:
                        len1 = kernel_size1 - pad3
                    if w == 0:
                        len2 = kernel_size2 - pad2
                    elif w == Width - 1:
                        len2 = kernel_size2 - pad4
                    area = len1 * len2
                    if area == 1:
                        area = 2 ** alpha_radix - 1
                    else:
                        area = math.floor((1 / area) * 2 ** alpha_radix)
                    y[i, h, w, c] = poolOut * area
    output_shift = -alpha_radix
    y = np.floor(y * 2 ** output_shift)
    y = y.transpose(0,3,1,2)
    return y
'''
def maxpool(x, kernel_size, pad, stride):
    [kernel_size1, kernel_size2] = kernel_size
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    # inputMap sizes
    m, Height_prev, Width_prev, C = np.shape(x)
    # outputMap sizes
    #Height, Width = int(np.floor(Height_prev/stride1)), int(np.floor(Width_prev/stride2))
    Height = int((Height_prev - kernel_size1 + pad1 + pad3)/stride1) + 1
    Width = int((Width_prev - kernel_size2 + pad2 + pad4)/stride2) + 1
    y = np.zeros((m, Height, Width, C))
    
    # padding
    x = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
        'constant', constant_values = 0)
    
    # maxpooling
    for i in range(m):
        for h in range(Height):
            for w in range(Width):
                for c in range(C):
                    h_start = h * stride1
                    h_end = h * stride1 + kernel_size1
                    w_start = w * stride2
                    w_end = w * stride2 + kernel_size2
                    poolField = x[0, h_start:h_end, w_start:w_end, c]
                    poolOut = np.max(poolField)
                    y[i, h, w, c] = poolOut
    return y
'''
def maxpool(x, kernel_size, pad, stride):
    x = x.transpose(0,2,3,1)
    [kernel_size1, kernel_size2] = kernel_size
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    # inputMap sizes
    m, Height_prev, Width_prev, C_prev = np.shape(x)
    # outputMap sizes
    # output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
    Height, Width = int(np.floor((Height_prev+pad1+pad3-kernel_size1)/stride1)+1), int(np.floor((Width_prev+pad2+pad4-kernel_size2)/stride2)+1)
    C = C_prev
    pool_out = np.zeros((Height, Width, C))
    
    # padding
    temp_map = np.lib.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 'edge')
    
    # maxpooling
    for h in range(Height):
        for w in range(Width):
            for c in range(C):
                h_start = h * stride1
                h_end = h * stride1 + kernel_size1
                w_start = w * stride2
                w_end = w * stride2 + kernel_size2
                poolField = temp_map[0, h_start:h_end, w_start:w_end, c]
                poolOut = np.max(poolField)
                pool_out[h, w, c] = poolOut



    pool_out = np.reshape(pool_out, (1,) + pool_out.shape)
    pool_out = pool_out.transpose(0,3,1,2)
    return pool_out


def flatten(x):
    x = x.flatten()
    x = np.reshape(x, (1, ) + x.shape)
    return x


def LeakyRelu(x, x_radix, alpha, alpha_radix, y_radix, working_bitwidth, datapath_bitwidth):
    x = x.transpose(0,2,3,1)
    if x_radix != y_radix:
        logging.error('leakyrelu datapath error!')
    (m, Height, Width, C) = np.shape(x)
    
    #x[np.where(x >= 0)] *= 2 ** alpha_radix
    x[np.where(x < 0)] *= alpha
    leakyrelu_out = x
    
    overflow_time = 0
    overflow = 2 ** (working_bitwidth - 1) - 1
    statis_data = leakyrelu_out.copy()
    overflow_time = np.where(leakyrelu_out > overflow)[0].shape[0] + np.where(leakyrelu_out < -overflow - 1)[0].shape[0]
    leakyrelu_out[np.where(leakyrelu_out > overflow)] = overflow
    leakyrelu_out[np.where(leakyrelu_out < -overflow - 1)] = -overflow - 1

    calc_overflow_info = {}
    if overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * Height * Width * C), overflow_th=overflow, data=statis_data)
    
    output_shift = -alpha_radix
    leakyrelu_out[np.where(leakyrelu_out < 0)] = np.floor(leakyrelu_out[np.where(leakyrelu_out < 0)] * 2 ** output_shift)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = leakyrelu_out.copy()
    datapath_overflow_time = np.where(leakyrelu_out > datapath_overflow)[0].shape[0] + np.where(leakyrelu_out < -datapath_overflow - 1)[0].shape[0]
    leakyrelu_out[np.where(leakyrelu_out > datapath_overflow)] = datapath_overflow
    leakyrelu_out[np.where(leakyrelu_out < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(m * Height * Width * C), overflow_th=datapath_overflow, data=statis_data)
    
    logging.info('input_radix:{}, alpha_radix:{}, output_radix:{}'.format(x_radix, alpha_radix, y_radix))
    leakyrelu_out = leakyrelu_out.transpose(0,3,1,2)

    return leakyrelu_out, y_radix, calc_overflow_info, datapath_overflow_info
    

def add(x, x_radix_list, y_radix, datapath_bitwidth):
    x[0], x[1] = x[0].transpose(0,2,3,1), x[1].transpose(0,2,3,1)
    if x[0].shape != x[1].shape:
        logging.error('add shape error!')

    overflow_time = 0
    overflow = 2 ** (datapath_bitwidth - 1) - 1
    if abs(x_radix_list[0] - x_radix_list[1]) > 2:
        x[0] = np.floor(x[0] * 2 ** (y_radix - x_radix_list[0]))
        x[0][np.where(x[0] > overflow)] = overflow
        x[0][np.where(x[0] < -overflow - 1)] = -overflow - 1 
        x[1] = np.floor(x[1] * 2 ** (y_radix - x_radix_list[1]))
        x[1][np.where(x[1] > overflow)] = overflow
        x[1][np.where(x[1] < -overflow - 1)] = -overflow - 1 
        add_out = x[0] + x[1]
    else:
        x[1] = np.floor(x[1] * 2 ** (x_radix_list[0] - x_radix_list[1]))
        add_out = x[0] + x[1]
        add_out = np.floor(add_out * 2 ** (y_radix - x_radix_list[0]))

    (m, Height, Width, C) = np.shape(add_out)
    statis_data = add_out.copy()
    overflow_time = np.where(add_out > overflow)[0].shape[0] + np.where(add_out < -overflow - 1)[0].shape[0]
    add_out[np.where(add_out > overflow)] = overflow
    add_out[np.where(add_out < -overflow - 1)] = -overflow - 1    
    
    datapath_overflow_info={}
    if overflow_time > 0:
        logging.warning('add overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * Height * Width * C), overflow_th=overflow, data=statis_data)
    
    output_radix = y_radix
    add_out = add_out.transpose(0,3,1,2)
    return add_out, output_radix, datapath_overflow_info

# concatenate in the channel axis
# concatenate in the channel axis
def concatenate(x, x_radix_list, y_radix, axis, datapath_bitwidth):
    for i in range(len(x)):
        x[i] = np.floor(x[i] * 2 ** (y_radix - x_radix_list[i]))
    output = x[0]
    for i in range(1,len(x)):
        output = np.concatenate((output, x[i]), axis = axis)
    output = output.transpose(0,2,3,1)
    (m, Height, Width, C) = np.shape(output)
    output_radix = y_radix
    overflow_time = 0
    overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = output.copy()
    overflow_time = np.where(output > overflow)[0].shape[0] + np.where(output < -overflow - 1)[0].shape[0]
    output[np.where(output > overflow)] = overflow
    output[np.where(output < -overflow - 1)] = -overflow - 1 
    
    datapath_overflow_info = {}
    if overflow_time > 0:
        logging.warning('concat overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * Height * Width * C), overflow_th=overflow, data=statis_data)
    output = output.transpose(0,3,1,2)
    return output, output_radix, datapath_overflow_info


# define unsamling2d
def upsample(input, scales, mode):
    input = input.transpose(0,2,3,1)
    [alpha, alpha_C, alpha_H, alpha_W] = scales
    # find the size of input
    (m, Height_prev, Width_prev, C_prev) = np.shape(input)
    # find the size of output
    Height = np.int(Height_prev * alpha_H)
    Width = np.int(Width_prev * alpha_W)
    C = C_prev
    # initialize the output with zeros.
    ups_output = np.zeros((m, Height, Width, C))
    if mode == b'nearest':
        for i in range(m):                           # loop over the batch of examples            
            for h in range(Height):                  # loop over vertical axis of the input volume
                for w in range(Width):               # loop over horizontal axis of the input volume
                    for c in range(C):               # loop over channels (= #filters) of the input volume
                        src_h = int(h / alpha_H)
                        src_w = int(w / alpha_W)
                        #comput the upsampling output
                        ups_output[i, h, w, c] = input[i, src_h, src_w, c]
    else:
        logging.error('upsample mode error!')

    if (ups_output.shape != (m, Height, Width, C)):
        logging.error('upsample shape error!')
    ups_output = ups_output.transpose(0,3,1,2)
    return ups_output

def BatchNormalization(x, x_radix, A_list, a_radix, B_list, y_radix, working_bitwidth, datapath_bitwidth):
    x = x.transpose(0,2,3,1)
    (m, Height, Width, C) = np.shape(x)
    bn_out = np.zeros((m, Height, Width, C))

    ### calc_overflow 
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1

    for c in range(C):
        bn_out[:, :, :, c] = x[:, :, :, c] * A_list[c] 

    output_shift = y_radix - (x_radix + a_radix)
    #print("output_shift is" ,output_shift)
    bn_out = np.floor(bn_out * 2 ** (output_shift))
    statis_data = bn_out.copy()
    calc_overflow_time = np.where(bn_out > calc_overflow)[0].shape[0] + np.where(bn_out < -calc_overflow - 1)[0].shape[0]
    bn_out[np.where(bn_out > calc_overflow)] = calc_overflow
    bn_out[np.where(bn_out < -calc_overflow - 1)] = -calc_overflow - 1 

    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('bn calc_overflow: {}/{}'.format(calc_overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(m * Height * Width * C), overflow_th=calc_overflow, data=statis_data)
    
    for c in range(C):
        bn_out[:, :, :, c] += B_list[c]

    assert(bn_out.shape == (m, Height, Width, C))

    overflow_time = 0
    overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = bn_out.copy()
    overflow_time = np.where(bn_out > overflow)[0].shape[0] + np.where(bn_out < -overflow - 1)[0].shape[0]
    bn_out[np.where(bn_out > overflow)] = overflow
    bn_out[np.where(bn_out < -overflow - 1)] = -overflow - 1 

    datapath_overflow_info = {}
    if overflow_time > 0:
        logging.warning('bn_out overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * Height * Width * C), overflow_th=overflow, data=statis_data)
    bn_out = bn_out.transpose(0,3,1,2)
    return bn_out, y_radix, calc_overflow_info, datapath_overflow_info


def BN_per_channel(x, x_radix, A_list, a_radix_list, B_list, y_radix, working_bitwidth, datapath_bitwidth):
    x = x.transpose(0,2,3,1)
    (m, Height, Width, C) = np.shape(x)
    bn_out = np.zeros((m, Height, Width, C))

    ### calc_overflow 
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1

    for c in range(C):
        bn_out[:, :, :, c] = x[:, :, :, c] * A_list[c] 

    for c in range(C):
        temp_shift = y_radix - x_radix -a_radix_list[c]
        bn_out[:, :, :, c] = np.floor(bn_out[:, :, :, c] * 2.0 ** (temp_shift))
    
    statis_data = bn_out.copy()
    calc_overflow_time = np.where(bn_out > calc_overflow)[0].shape[0] + np.where(bn_out < -calc_overflow - 1)[0].shape[0]
    bn_out[np.where(bn_out > calc_overflow)] = calc_overflow
    bn_out[np.where(bn_out < -calc_overflow - 1)] = -calc_overflow - 1 

    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('bn calc_overflow: {}/{}'.format(calc_overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(m * Height * Width * C), overflow_th=calc_overflow, data=statis_data)
    
    for c in range(C):
        bn_out[:, :, :, c] += B_list[c]

    assert(bn_out.shape == (m, Height, Width, C))

    overflow_time = 0
    overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = bn_out.copy()
    overflow_time = np.where(bn_out > overflow)[0].shape[0] + np.where(bn_out < -overflow - 1)[0].shape[0]
    bn_out[np.where(bn_out > overflow)] = overflow
    bn_out[np.where(bn_out < -overflow - 1)] = -overflow - 1 

    datapath_overflow_info = {}
    if overflow_time > 0:
        logging.warning('bn_out overflow: {}/{}'.format(overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(overflow_time, total_cnt=(m * Height * Width * C), overflow_th=overflow, data=statis_data)
    
    bn_out = bn_out.transpose(0,3,1,2)
    return bn_out, y_radix, calc_overflow_info, datapath_overflow_info

def depth_wise_conv_by_channel(x, x_radix, w_kernel, w_kernel_radix_list, w_bias, y_radix, working_bitwidth, datapath_bitwidth, bias_bitwidth, group, stride, pad, dilation = [1, 1]):
    x = x.transpose(0,2,3,1)
    assert (group == x.shape[-1]), "Depth_Wise_Conv Group Error! This function do not support."
    w_kernel = w_kernel.transpose(2,3,0,1)
    
    (m, Height_prev, Width_prev, C_prev) = np.shape(x)
    #Retrieve dimensions from W's shape
    (f, f, C, C_prev) = w_kernel.shape
    bias_overflow_time = 0
    bias_overflow = 2 ** (bias_bitwidth - 1) - 1
    statis_data = w_bias.copy()
    for c in range(C):
        if w_bias[0, 0, 0, c] > bias_overflow:
            w_bias[0, 0, 0, c] = bias_overflow
            bias_overflow_time += 1
        elif w_bias[0, 0, 0, c] < -bias_overflow - 1:
            w_bias[0, 0, 0, c] = -bias_overflow - 1
            bias_overflow_time += 1
    if bias_overflow_time > 0:
        logging.warning('bias overflow: {}/{}'.format(bias_overflow_time, C))
        bias_overflow_info = statistic_overflow(bias_overflow_time, total_cnt=C, overflow_th=bias_overflow, data=statis_data)
    #Retrieve information of stride and pad based on keras' vgg16
    [stride1, stride2] = stride
    [pad1, pad2, pad3, pad4] = pad
    [dilation1, dilation2] = dilation
    # Compute the dimensions of the CONV output volume
    Height = int((Height_prev - f + pad1 + pad3) / stride1) + 1
    Width = int((Width_prev - f + pad2 + pad4) / stride2) + 1
    # Initialize the output volume Z with zeros.
    conv_out = np.zeros((m, Height, Width, C))
    # pad the input data(input_img) with zeros
    img_padded = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
                        'constant', constant_values=0)
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1

    #convolution
    for i in range(m):
        image_in = img_padded[i]
        for h in range(Height):                  # loop over vertical axis of the output volume
            for w in range(Width):               # loop over horizontal axis of the output volume
                for c in range(C):               # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice" 
                    h_start = h * stride1
                    h_end = h * stride1 + f
                    w_start = w * stride2
                    w_end = w * stride2 + f
                    # Use the corners to define the (3D) slice of image_in
                    conv_box = image_in[h_start:h_end, w_start:w_end, c]
                    #print('conv_box size:{}'.format(conv_box.shape))
                    weig = np.reshape(w_kernel[: ,:, c, :], (f, f))
                    conv_out[i, h, w, c] = np.sum(conv_box * weig)
                    if conv_out[i, h, w, c] > calc_overflow:
                        conv_out[i, h, w, c] = calc_overflow
                        calc_overflow_time += 1
                    elif conv_out[i, h, w, c] < -calc_overflow - 1:
                        conv_out[i, h, w, c] = -calc_overflow - 1
                        calc_overflow_time += 1
    
    statis_data = conv_out.copy()
    calc_overflow_time = np.where(conv_out > calc_overflow)[0].shape[0] + np.where(conv_out < -calc_overflow - 1)[0].shape[0]
    conv_out[np.where(conv_out > calc_overflow)] = calc_overflow
    conv_out[np.where(conv_out < -calc_overflow - 1)] = -calc_overflow - 1
    
    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time, m * Height * Width * C))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=m * Height * Width * C, overflow_th=calc_overflow, data=statis_data)

    #Before datapath_bitwidth limit, output_radix = w_bias_radix
    w_bias_radix = y_radix
    for c in range(C):
        temp_shift = y_radix - x_radix - w_kernel_radix_list[c]
        #print(temp_shift)
        conv_out[:, :, :, c] = np.floor(conv_out[:, :, :, c] * 2 ** temp_shift)

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = conv_out.copy()
    for i in range(m):
        for h in range(Height):
            for w in range(Width):
                for c in range(C):
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
                    conv_out[i, h, w, c] += w_bias[0, 0, 0, c]
                    if conv_out[i, h, w, c] > datapath_overflow:
                        conv_out[i, h, w, c] = datapath_overflow
                        datapath_overflow_time += 1
                    if conv_out[i, h, w, c] < -datapath_overflow - 1:
                        conv_out[i, h, w, c] = -datapath_overflow - 1
                        datapath_overflow_time += 1
    
    datapath_overflow_info={}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time, m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=m * Height * Width * C, overflow_th=calc_overflow, data=statis_data)
    
    logging.info('input_radix:{}, kernel_radix:{}, bias_radix:{}, output_radix:{}'.format(x_radix, w_kernel_radix_list, w_bias_radix, y_radix))

    conv_out = conv_out.transpose(0,3,1,2)
    
    return conv_out, y_radix, calc_overflow_info, datapath_overflow_info


def mish(x, x_radix, x_bitwidth, y_radix, datapath_bitwidth):
    '''we use LUT to implementation Mish function, lenth of LUT is 2**16
    '''
    ##gen LUT
    #从0b0000_0000到0b1111_1111每一个二进制数都对应一个查找表的输出

    
    '''we use LUT to implementation Mish function
    '''  
    lookup_table = {}

    xmax = 2 ** (x_bitwidth-1) 

    for fp_x in range(-xmax,xmax):
        float_x = fp_x/2**x_radix
        float_y = floatmish(float_x)
        fp_y = np.round(float_y * 2 ** y_radix)
        lookup_table[fp_x] = fp_y

    Batch, Channel, Weight, Height = x.shape
    x = x.flatten().tolist()
    x = list(map(lambda x: lookup_table[x], x))
    x = np.array(x)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = x.copy()
    datapath_overflow_time = np.where(x > datapath_overflow)[0].shape[0] + \
                             np.where(x < -datapath_overflow - 1)[0].shape[0]
    x[np.where(x > datapath_overflow)] = datapath_overflow
    x[np.where(x < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 5:
        assert False,"For Mish,radix error because it is not Monotonic increasing"
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          x.size))  ##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(x.size),
                                                    overflow_th=datapath_overflow, data=statis_data)




    res = np.reshape(x, (Batch, Channel, Weight, Height))

    return res, y_radix,datapath_overflow_info

def upsample_yolov4(x_list,x_radix_list):
    '''custom node in yolov4, more details in https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
    '''
    x1, x2 = x_list[0], x_list[1]
    shape_ori = x1.shape
    scale = [x2.shape[2]//shape_ori[2], x2.shape[3]//shape_ori[3]]

    x1 = np.expand_dims(x1,axis=3)
    x1 = np.expand_dims(x1,axis=5)
    x1 = np.repeat(x1,scale[0],axis=3)
    x1 = np.repeat(x1,scale[1],axis=5)
    x1 = np.reshape(x1,(shape_ori[0],shape_ori[1],shape_ori[2]*scale[0],shape_ori[3]*scale[1]))
    y_radix = x_radix_list[0]

    return x1, y_radix


def silu(x, x_radix, x_bitwidth, y_radix, datapath_bitwidth):
    '''we use LUT to implementation Mish function
    '''  
    lookup_table = {}

    # xmax = 2 ** (x_bitwidth - 1 - x_radix)

    # for i in range(2**x_bitwidth):
    #     float_x = -xmax + 2 * i * xmax / (2**x_bitwidth)
    #     float_y = floatsilu(float_x)
    #     fp_x = np.round(float_x * 2 ** x_radix)
    #     fp_y = np.round(float_y * 2 ** y_radix)
    #     lookup_table[fp_x] = fp_y
    xmax = 2 ** (x_bitwidth-1) 
    for fp_x in range(-xmax,xmax):
        float_x = fp_x/2**x_radix
        float_y = floatsilu(float_x)
        fp_y = np.round(float_y * 2 ** y_radix)
        lookup_table[fp_x] = fp_y
    # print(lookup_table)
    # assert False
    Batch, Channel, Weight, Height = x.shape
    x = x.flatten().tolist()
    x = list(map(lambda x: lookup_table[x], x))
    x = np.array(x)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = x.copy()
    datapath_overflow_time = np.where(x > datapath_overflow)[0].shape[0] + \
                             np.where(x < -datapath_overflow - 1)[0].shape[0]
    x[np.where(x > datapath_overflow)] = datapath_overflow
    x[np.where(x < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 5:
        assert False,"For Silu,radix error because it is not Monotonic increasing"
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          x.size))  ##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(x.size),
                                                    overflow_th=datapath_overflow, data=statis_data)




    res = np.reshape(x, (Batch, Channel, Weight, Height))

    return res, y_radix,datapath_overflow_info


def tanh(x, x_radix, y_radix, datapath_bitwidth, lenth_lut=16):
    '''we use LUT to implementation Mish function, lenth of LUT is 2**16
    '''
    ##gen LUT
    lookup_table = {}
    xmax = 2 ** (datapath_bitwidth - 1 - x_radix)
    ymax = floattanh(xmax)
    ymin = floattanh(-xmax)
    lookuptable_max = max(abs(ymax),abs(ymin))
    y_radix = getradix(lookuptable_max, lenth_lut)
    for i in range(2**lenth_lut):
        float_x = -xmax + 2 * i * xmax / (2**lenth_lut)
        float_y = floattanh(float_x)
        fp_x = np.floor(float_x * 2 ** x_radix)
        fp_y = np.floor(float_y * 2 ** y_radix)
        lookup_table[fp_x] = fp_y

    Batch, Channel, Weight, Height = x.shape
    x = x.flatten().tolist()
    x = list(map(lambda x: lookup_table[x], x))
    x = np.array(x)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = x.copy()
    datapath_overflow_time = np.where(x > datapath_overflow)[0].shape[0] + \
                             np.where(x < -datapath_overflow - 1)[0].shape[0]
    x[np.where(x > datapath_overflow)] = datapath_overflow
    x[np.where(x < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          x.size))  ##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(x.size),
                                                    overflow_th=datapath_overflow, data=statis_data)


    res = np.reshape(x, (Batch, Channel, Weight, Height))

    return res, y_radix,datapath_overflow_info

def sigmoid(x, x_radix, y_radix, datapath_bitwidth, lenth_lut=16):
    '''we use LUT to implementation Mish function, lenth of LUT is 2**16
    '''
    ##gen LUT   
    lookup_table = {}

    xmax = 2 ** (datapath_bitwidth - 1 - x_radix)
    ymax = floatsigmoid(xmax)
    ymin = floatsigmoid(-xmax)
    lookuptable_max = max(abs(ymax),abs(ymin))
    y_radix = getradix(lookuptable_max, lenth_lut)
    for i in range(2**lenth_lut):
        float_x = -xmax + 2 * i * xmax / (2**lenth_lut)
        float_y = floatsigmoid(float_x)
        fp_x = np.floor(float_x * 2 ** x_radix)
        fp_y = np.floor(float_y * 2 ** y_radix)
        lookup_table[fp_x] = fp_y

    Batch, Channel, Weight, Height = x.shape
    x = x.flatten().tolist()
    x = list(map(lambda x: lookup_table[x], x))
    x = np.array(x)
    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = x.copy()
    datapath_overflow_time = np.where(x > datapath_overflow)[0].shape[0] + \
                             np.where(x < -datapath_overflow - 1)[0].shape[0]
    x[np.where(x > datapath_overflow)] = datapath_overflow
    x[np.where(x < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          x.size))  ##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(x.size),
                                                    overflow_th=datapath_overflow, data=statis_data)




    res = np.reshape(x, (Batch, Channel, Weight, Height))

    return res, y_radix,datapath_overflow_info

def mul(x, x_radix_list,y_radix,working_bitwidth,datapath_bitwidth):

    x1 = np.array(x[0])
    x2 = np.array(x[1])
    y = x1 * x2
    
    calc_overflow_time = 0
    calc_overflow = 2 ** (working_bitwidth - 1) - 1
    statis_data = y.copy()
    calc_overflow_time = np.where(y > calc_overflow)[0].shape[0] + \
                         np.where(y < -calc_overflow - 1)[0].shape[0]
    y[np.where(y > calc_overflow)] = calc_overflow
    y[np.where(y < -calc_overflow - 1)] = -calc_overflow - 1

    calc_overflow_info = {}
    if calc_overflow_time > 0:
        logging.warning('engine internal overflow: {}/{}'.format(calc_overflow_time,y.size))  ##m * Height * Width * c))
        calc_overflow_info = statistic_overflow(calc_overflow_time, total_cnt=(y.size),
                                                overflow_th=calc_overflow, data=statis_data)
    y = np.floor(y * 2 ** (y_radix - x_radix_list[0] - x_radix_list[1]))

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = y.copy()
    datapath_overflow_time = np.where(y > datapath_overflow)[0].shape[0] + \
                             np.where(y < -datapath_overflow - 1)[0].shape[0]
    y[np.where(y > datapath_overflow)] = datapath_overflow
    y[np.where(y < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          y.size)) 
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(y.size),
                                                    overflow_th=datapath_overflow, data=statis_data)

    return y ,y_radix,calc_overflow_info,datapath_overflow_info


def softplus(x, x_radix, y_radix, datapath_bitwidth, lenth_lut=16):

    '''we use LUT to implementation Mish function, lenth of LUT is 2**16
    # '''
    # ##gen LUT

    lookup_table = {}
    xmax = 2 ** (datapath_bitwidth - 1 - x_radix)
    ymax = floatsoftplus(xmax)
    ymin = floatsoftplus(-xmax)
    lookuptable_max = max(abs(ymax),abs(ymin))
    y_radix = getradix(lookuptable_max, lenth_lut)
    for i in range(2**lenth_lut):
        float_x = -xmax + 2 * i * xmax / (2**lenth_lut)
        float_y = floatsoftplus(float_x)
        fp_x = np.floor(float_x * 2 ** x_radix)
        fp_y = np.floor(float_y * 2 ** y_radix)
        lookup_table[fp_x] = fp_y

    Batch, Channel, Weight, Height = x.shape
    x = x.flatten().tolist()
    x = list(map(lambda x: lookup_table[x], x))

    x = np.array(x)

    datapath_overflow_time = 0
    datapath_overflow = 2 ** (datapath_bitwidth - 1) - 1
    statis_data = x.copy()
    datapath_overflow_time = np.where(x > datapath_overflow)[0].shape[0] + \
                             np.where(x < -datapath_overflow - 1)[0].shape[0]
    x[np.where(x > datapath_overflow)] = datapath_overflow
    x[np.where(x < -datapath_overflow - 1)] = -datapath_overflow - 1

    datapath_overflow_info = {}
    if datapath_overflow_time > 0:
        logging.warning('datapath overflow: {}/{}'.format(datapath_overflow_time,
                                                          x.size))  ##m * Height * Width * C))
        datapath_overflow_info = statistic_overflow(datapath_overflow_time, total_cnt=(x.size),
                                                    overflow_th=datapath_overflow, data=statis_data)



    res = np.reshape(x, (Batch, Channel, Weight, Height))

    return res, y_radix,datapath_overflow_info

def resize(x,roi,scales,sizes,coordinate_transformation_mode,cubic_coeff_a,exclude_outside,extrapolation_value,mode,nearest_mode):
    ''' Resize the input tensor, more details in https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
        In general, it calculates every value in the output tensor as a weighted average of neighborhood in the input tensor.
        #TODO(weihuadong 2021.05.23): this functon has many parameters and options and it need to be perfected in the future.
                                      Currently we only support one specific case in yolov5.
    '''
    assert  str(coordinate_transformation_mode,encoding='utf-8') == 'asymmetric' and \
            cubic_coeff_a == -0.75 and \
            exclude_outside == 0 and \
            extrapolation_value == 0.0 and \
            str(mode,encoding='utf-8') == 'nearest' and \
            str(nearest_mode,encoding='utf-8') == 'floor', 'unsupported case right now'
    assert not sizes, 'unsupported case right now'
    assert not roi, 'unsupported case right now'

    [alpha, alpha_C, alpha_H, alpha_W] = scales
    # find the size of input
    x = x.transpose(0,2,3,1)
    (m, Height_prev, Width_prev, C_prev) = np.shape(x)
    # find the size of output
    Height = np.int(Height_prev * alpha_H)
    Width = np.int(Width_prev * alpha_W)
    C = C_prev
    # initialize the output with zeros.
    ups_output = np.zeros((m, Height, Width, C))
    if str(mode,encoding='utf-8') == 'nearest':
        for i in range(m):                           # loop over the batch of examples
            for h in range(Height):                  # loop over vertical axis of the input volume
                for w in range(Width):               # loop over horizontal axis of the input volume
                    for c in range(C):               # loop over channels (= #filters) of the input volume
                        src_h = int(h / alpha_H)
                        src_w = int(w / alpha_W)
                        #comput the upsampling output
                        ups_output[i, h, w, c] = x[i, src_h, src_w, c]
    ups_output = ups_output.transpose(0,3,1,2)
    assert(ups_output.shape == (m, C, Height, Width))
    return ups_output

def reshape(data, shape, shape_radix,allowzero):  # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    shape = shape*2**(-shape_radix)
    new_shape = np.copy(shape).astype(np.int64)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    return np.reshape(data, new_shape)


def expand(input,shape):
    '''Broadcast the input tensor following the given shape and the broadcast rule.
    '''
    return input * np.ones(shape)