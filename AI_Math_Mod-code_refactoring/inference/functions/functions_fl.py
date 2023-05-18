# import cupy as cp
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
import numpy as np
import os
import matplotlib.pyplot as plt
import json

def depth_wise_conv(x: list, w_kernel: list, bias: list, stride: list, pad: list, group: int, dilation = [1, 1]) -> list:
    """depth_wise_conv function 
    Args:
        x (list): input from last layer
        w_kernel (list): weights in conv layer
        bias (list): bias in conv layer
        stride (list): conv stride
        pad (list): pad mode
        group (int): depth_wise group, generally this ought to be equal with input_channel size
        dilation (list, optional): a parameter related to dialted conv. Defaults to [1, 1].

    Returns:
        list: output 
    """
    x = x.transpose(0,2,3,1)
    assert (group == x.shape[-1]), "Depth_Wise_Conv Group Error! This function do not support."
    w_kernel = w_kernel.transpose(2,3,0,1)
    
    (m, Height_prev, Width_prev, C_prev) = np.shape(x)
    (f, f, C, C_prev) = w_kernel.shape
    [stride1, stride2] = stride
    [pad1, pad2, pad3, pad4] = pad
    [dilation1, dilation2] = dilation

    Height = int((Height_prev - f + pad1 + pad3) / stride1) + 1
    Width = int((Width_prev - f + pad2 + pad4) / stride2) + 1
    conv_out = np.zeros((m, Height, Width, C))
    conv_out_before_bias = np.zeros((m, Height, Width, C))

    img_padded = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
                        'constant', constant_values=0)
    
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
                    weig = np.reshape(w_kernel[: ,:, c, :], (f, f))
                    conv_out_before_bias[i, h, w, c] = np.sum(conv_box * weig)
                    if bias !=[]:
                        conv_out[i, h, w, c] = conv_out_before_bias[i, h, w, c] + bias[0, 0, 0, c]
                    else:
                        conv_out[i, h, w, c] = conv_out_before_bias[i, h, w, c]
                    #conv_out[i, h, w, c] = np.sum(conv_box * weig) 
                   
    conv_out  = conv_out.transpose(0,3,1,2)
    conv_out_before_bias  = conv_out_before_bias.transpose(0,3,1,2)
    return conv_out,conv_out_before_bias


def clip(x, x_min, x_max):
    y = np.clip(x, x_min, x_max)
    return y


def averagepool(x, kernel_size, pad, stride):
    [kernel_size1, kernel_size2] = kernel_size
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    x = x.transpose(0,2,3,1)
    (m, Height_prev, Width_prev, C) = x.shape
    Height = int(np.floor((Height_prev - kernel_size1 + pad1 + pad3)/stride1)) + 1
    Width = int(np.floor((Width_prev - kernel_size2 + pad2 + pad4)/stride2)) + 1
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
                        
                    y[i, h, w, c] = poolOut / (len1 * len2)
    y = y.transpose(0,3,1,2)
    return y


def PRelu(x, slope):
    x = x.transpose(0,2,3,1)
    (m, height, width, channel) = x.shape
    assert len(slope) == channel
    for i in range(m):
        for h in range(height):
            for w in range(width):
                for c in range(channel):
                    if x[i, h, w, c] < 0:
                        x[i, h, w, c] *= slope[c]
    x = x.transpose(0,3,1,2)
    return x


def update_w_kernel_for_dilation_conv(w_kernel, dilation):
    [dilation1, dilation2] = dilation
    if [dilation1, dilation2] != [1, 1]:
        w_kernel = w_kernel.transpose(2, 3, 1, 0)
        (f, f, C_prev, C) = w_kernel.shape
        kernel_size_effective = f + (f - 1) * (dilation1 - 1)
        w_kernel_dilated = np.zeros((kernel_size_effective, kernel_size_effective, C_prev, C))
        for j in range(f):
            for k in range(f):
                for kernel_channel in range(C_prev):
                    for kernel_num in range(C):
                        w_kernel_dilated[dilation1 * j, dilation2 * k, kernel_channel, kernel_num] = w_kernel[j, k, kernel_channel, kernel_num]
        w_kernel = w_kernel_dilated.transpose(3,2,0,1)
    
    return w_kernel


def conv_normal(x, w_kernel, w_bias, stride, pad, dilation = [1, 1]):
    x = x.transpose(0,2,3,1)
    w_kernel = w_kernel.transpose(2,3,1,0)
    (m, Height_prev, Width_prev, C_prev) = np.shape(x)
    #Retrieve dimensions from W's shape
    (f, f, C_prev, C) = w_kernel.shape
    #Retrieve information of stride and pad based on keras' vgg16
    [stride1, stride2] = stride
    [pad1, pad2, pad3, pad4] = pad
    [dilation1, dilation2] = dilation
    # Compute the dimensions of the CONV output volume
    Height = int((Height_prev - f + pad1 + pad3) / stride1) + 1
    Width = int((Width_prev - f + pad2 + pad4) / stride2) + 1
    # Initialize the output volume Z with zeros.
    conv_out = np.zeros((m, Height, Width, C))
    conv_out_before_bias = np.zeros((m, Height, Width, C))
    # pad the input data(input_img) with zeros
    img_padded = np.pad(x, ((0, 0), (pad1, pad3), (pad2, pad4), (0, 0)), 
                        'constant', constant_values=0)

    
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
                    conv_out_before_bias[i, h, w, c] = np.sum(conv_box * w_kernel[:, :, :, c])
                    conv_out[i, h, w, c] = np.sum(conv_box * w_kernel[:, :, :, c]) + w_bias[0, 0, 0, c]
                             
    conv_out  = conv_out.transpose(0,3,1,2)
    conv_out_before_bias  = conv_out_before_bias.transpose(0,3,1,2)
    return conv_out,conv_out_before_bias

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
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride1*out_h
        for x in range(filter_w):
            x_max = x + stride2*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride1, x:x_max:stride2]#

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #-1表示第二个维度需要程序进行推理，即总个数除以N*out_h*out_w
    return col

def conv_cpu(x, w_kernel, w_bias, stride, pad, dilation = [1, 1]):
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
    # 利用im2col转换为行
    col = im2col_cpu(x, FH, FW, stride, pad)
    # 卷积核转换为列，展开为2维数组
    col_W = w_kernel.reshape(FN, -1).T
    out_before_bias = np.dot(col, col_W)
    # 计算正向传播
    if w_bias !=[]:
        out = out_before_bias + w_bias
    else:
        out = out_before_bias
    out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
    out_before_bias = out_before_bias.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

    return out, out_before_bias


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
    col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride1*out_h
        for x in range(filter_w):
            x_max = x + stride2*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride1, x:x_max:stride2]#

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    #-1表示第二个维度需要程序进行推理，即总个数除以N*out_h*out_w
    return col

def conv_gpu(x, w_kernel, w_bias, stride, pad, dilation = [1, 1]):
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
    # 利用im2col转换为行
    col = im2col_gpu(x, FH, FW, stride, pad)
    # 卷积核转换为列，展开为2维数组
    col_W = w_kernel.reshape(FN, -1).T
    # 计算正向传播
    out_before_bias = cp.dot(col, col_W)
    out_before_bias = out_before_bias.reshape(N, out_h, out_w, -1)
    out = cp.dot(col, col_W) + w_bias
    out = out.reshape(N, out_h, out_w, -1)

    return out,out_before_bias

def conv(x, w_kernel, w_bias, stride, pad, dilation = [1, 1], option='img2col'):
    if option == 'normal':
        return conv_normal(x, w_kernel, w_bias, stride, pad, dilation)

    elif option == 'img2col':
        return conv_cpu(x, w_kernel, w_bias, stride, pad, dilation)
    
    elif option == 'cuda':
        x_gpu = cp.asarray(x)  # move the data to the device 
        w_kernel_gpu = cp.asarray(w_kernel)
        w_bias_gpu = cp.asarray(w_bias)
        conv_out_gpu,conv_out_before_bias_gpu = conv_gpu(x_gpu, w_kernel_gpu, w_bias_gpu, stride, pad, dilation)
        return cp.asnumpy(conv_out_gpu),cp.asnumpy(conv_out_before_bias_gpu)
    else:
        raise ValueError("please fill the right option: (normal/img2col/cuda)")


def relu(x):
    s = np.where(x < 0, 0., x)
    return s

def batchnormalization(x, gamma, beta, mean, var, epsilon, momentum):
    x = x.transpose(0,2,3,1)
    (m, Height, Width, C) = np.shape(x)
    bn_out = np.zeros((m, Height, Width, C))

    var = np.array(var)
    a = gamma / np.sqrt(var + epsilon)
    b = beta - gamma * mean / np.sqrt(var + epsilon)
    bn_out_before_b = x * a
    bn_out = x * a + b
    bn_out = bn_out.transpose(0,3,1,2)
    bn_out_before_b = bn_out_before_b.transpose(0,3,1,2)
    assert(bn_out.shape == (m, C, Height, Width))
    assert(bn_out_before_b.shape == (m, C, Height, Width))

    return bn_out,bn_out_before_b

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
    [kernel_size1, kernel_size2] = kernel_size
    [pad1, pad2, pad3, pad4] = pad
    [stride1, stride2] = stride
    # inputMap sizes
    x = x.transpose(0,2,3,1)
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


    
    pool_out = np.reshape(pool_out, (1,) + pool_out.shape).transpose(0,3,1,2)
    
    return pool_out


def flatten(x):
    x = x.flatten()
    x = np.reshape(x, (1, ) + x.shape)
    return x


def LeakyRelu(x, alpha):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * np.abs(x)


#define the add function which just adds to same shape array directly.
def add(x):
    assert(x[0].shape == x[1].shape)
    return x[0] + x[1]

# concatenate in the channel axis
def concatenate(x, axis):
    output = x[0]
    for i in range(1,len(x)):
        output = np.concatenate((output, x[i]), axis = axis)
    return output


# define unsamling2d
def upsample(x, scales, mode):
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
    if mode == b'nearest':
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


def l2_norm(input, gamma_list):
    """ L2_Normalization function -- do L2-norm per each pixel in a feature map instead of the whole.

    Arguments:
        input {list} -- input data getting from last layer
        gamma_list {list} -- scale parameter for each channel
    """
    input = input.transpose(0,2,3,1)
    (m, Height, Width, C) = np.shape(input)
    assert(C == len(gamma_list))
    norm_data = input ** 2
    norm_sum = np.sum(norm_data, (0, 3))
    sqr_norm_data = np.sqrt(norm_sum)
    l2_norm_output = np.zeros((m, Height, Width, C))
    for i in range(Height):
        for j in range(Width):
            for k in range(C):
                l2_norm_output[:,i,j, k] = (input[:,i,j, k] / sqr_norm_data[i][j])* gamma_list[k]
    l2_norm_output = l2_norm_output.transpose(0,3,1,2)
    assert(l2_norm_output.shape == (m, C, Height, Width))
    return l2_norm_output


def softplus(x, beta=1, threshold=20):
    '''
    torch version
    if beta * x > threshold:
        return x
    return 1 / beta * np.log(1 + np.exp(beta*x))
    '''
    return np.log(1 + np.exp(x)) ##onnx version

def tanh(x):
    return np.tanh(x)


def mul(x):
    x1 = np.array(x[0])
    x2 = np.array(x[1])
    return x1 * x2

def mish(x):
    return mul([x,tanh(softplus(x))])

def upsample_yolov4(x):
    '''custom node in yolov4, more details in https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
    '''
    x1, x2 = x[0], x[1]
    shape_ori = x1.shape
    scale = [x2.shape[2]//shape_ori[2], x2.shape[3]//shape_ori[3]]

    x1 = np.expand_dims(x1,axis=3)
    x1 = np.expand_dims(x1,axis=5)
    x1 = np.repeat(x1,scale[0],axis=3)
    x1 = np.repeat(x1,scale[1],axis=5)
    x1 = np.reshape(x1,(shape_ori[0],shape_ori[1],shape_ori[2]*scale[0],shape_ori[3]*scale[1]))

    return x1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.negative(x)))

def silu(x):
    return x*sigmoid(x)

def shape(x):
    '''Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor
    '''
    return np.array(x.shape).astype(np.int64)


def unsqueeze(x,axes):
    x = np.array(x)
    axes_list = [item for item in axes]
    axes_list.sort()
    for item in axes_list:
        x = np.expand_dims(x, axis=item)
    
    return x


def gather(x, indices, axis):
    y = np.take(x, indices, axis)
    return y


def constantofshape(x,value=0):
    '''Generate a tensor with given value and shape.
       value: The value of the output elements.Should be a one-element tensor
    '''
    x = np.array(x).astype(np.int64)
    value = float(value)
    if value == 1:
        return np.ones(x)
    elif value == 0:
        return np.zeros(x)
    else:
        raise ValueError('unsupported value')


def equal(x,y):
    return np.equal(x, y)


def where(condition,x,y):
    '''condition (non-differentiable) : When True (nonzero), yield X, otherwise yield Y
       X (differentiable) : values selected at indices where condition is True
       Y (differentiable) : values selected at indices where condition is False
    '''
    condition = np.array(condition, dtype=np.bool)
    x = np.array(x, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    return np.where(condition, x, y)


def reshape(data, shape, allowzero):  # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    new_shape = np.copy(shape).astype(np.int64)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    return np.reshape(data, new_shape)


def div(x):
    x1 = np.array(x[0])
    x2 = np.array(x[1])
    return x1 / x2


def cast(input, to):
    '''The operator casts the elements of a given input tensor to a data type specified by the 'to' argument 
    and returns an output tensor of the same size in the converted type.
    The 'to' argument must be one of the data types specified in the 'DataType' enum field in the TensorProto message
    
    Arguments:
    to: int64
    The data type to which the elements of the input tensor are cast. Strictly must be one of the types from DataType enum in TensorProto
    '''
    for k, v in vars(onnx.TensorProto).items():
        if v == to:
            if 'int64' in k.lower():
                return np.array(input).astype(np.int64)
            elif 'int32' in k.lower():
                return np.array(input).astype(np.int32)
            elif 'int16' in k.lower():
                return np.array(input).astype(np.int16)
            elif 'float16' in k.lower():
                return np.array(input).astype(np.float16)
            elif 'float' in k.lower():
                return np.array(input).astype(np.float32)
            else:
                raise KeyError(f'unsupported datatype:{k}')
    
    raise ValueError('unsupported value of to')
                

def expand(input,shape):
    '''Broadcast the input tensor following the given shape and the broadcast rule.
    '''
    return input * np.ones(shape)


def transpose(input, perm):
    perm_list = [item for item in perm]
    return np.transpose(input,perm_list)


def sub(x):
    x1 = np.array(x[0])
    x2 = np.array(x[1])
    return x1 - x2


def exp(x):
    return np.exp(x)


def slice(input, starts, ends, axes, steps):
    starts, ends, axes, steps = int(starts), int(ends), int(axes), int(steps)
    lenth_shape = len(list(input.shape))
    temp = [i for i in range(lenth_shape)]
    temp[0], temp[axes] = axes, 0
    input = np.transpose(input,temp)
    input = input[starts:ends:steps,...]
    input = np.transpose(input,temp)
    return input


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