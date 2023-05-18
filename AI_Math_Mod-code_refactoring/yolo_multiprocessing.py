import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import sys,getopt
from ModelInferencer.core import float_inference
from DatapathAnalyser.core import datapath_analyse
from utils.img_preprocess import img_preprocess, img_preprocess_multi

import numpy as np
import os
import matplotlib.pyplot as plt
import json
import keras
from keras.models import *
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
from tkinter import _flatten
import math
import shutil
import time
import multiprocessing

if __name__ == "__main__":
    multipro = 4
    method = 'yolo'
    output_folder_path = '/home/data1/output0'
    onnx_file_path = '/home/data1/yolov3_416.h5.onnx'
    image_npy_folder = '/home/data1/img0_npy'
    temp_folder_path = '/home/data1/tmp0'
    input_image_folder_path = '/home/data1/img1'
    lib_to_accelerate = 'float_inference'

    list_a = []
    img_preprocess_multi(input_image_folder_path, image_npy_folder, method, multipro)
    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)
    new_onnx_file = []
    for k in range(multipro):
        tmp_tmp = os.path.join(temp_folder_path, str(k))
        npy_tmp = os.path.join(image_npy_folder, str(k))
        output_tmp = os.path.join(output_folder_path, str(k))
        file_name, file_extend = os.path.splitext(onnx_file_path)
        new_onnx_file.append(file_name + '_' + str(k) + file_extend)
        shutil.copyfile(onnx_file_path, new_onnx_file[k])
        list_a.append([new_onnx_file[k], output_tmp, npy_tmp, tmp_tmp])
    for i in list_a:
        print(i)
    p = multiprocessing.Pool(multipro)        
    
    for i1 in list_a:
        #multi processing
        if lib_to_accelerate == 'datapath_analyse':    
            p.apply_async(datapath_analyse,args=i1)
        elif lib_to_accelerate == 'float_inference':
            p.apply_async(float_inference,args=i1)
    p.close()
    p.join()

    #remove tmp file
    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.mkdir(temp_folder_path)
    #remove onnx_x
    for i in new_onnx_file:
        os.remove(i)

    #copy result to output folder
    for root, dirs, files in os.walk(output_folder_path):
        for name in files:
            print(os.path.join(root, name))
            shutil.copy(os.path.join(root, name), output_folder_path)
    for i in list_a:
        if os.path.exists(i[1]):
            shutil.rmtree(i[1])

    print('Finish ' + lib_to_accelerate + '!')