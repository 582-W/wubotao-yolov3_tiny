from ModelEvaluator.core import evaluate
from utils.img_preprocess import img_preprocess
from ModelInferencer.core import float_inference
import shutil
import os
import logging

input_image_folder_path = '/data1/image'
image_npy_folder = '/data1/img_npy'
onnx_file_path = '/data1/vgg16_cut.onnx'
quantized_onnx_file_path = '/data1/vgg16_cut.onnx'
output_path = '/data1/output'
float_folder_path = os.path.join(output_path, 'float_inference_output')
quantized_folder_path = os.path.join(output_path, 'quantized_inference_output')
output_result_path = os.path.join(output_path, 'evaluate_result')
output_per_layer = False
img_size = (224, 224)
method = 'vgg'

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)

img_preprocess(input_image_folder_path, image_npy_folder, method, img_size)
#float result
float_inference(onnx_file_path, float_folder_path, image_npy_folder, output_per_layer)
#quantized weight result
float_inference(quantized_onnx_file_path, quantized_folder_path, image_npy_folder, output_per_layer)

if os.path.exists(output_result_path):
    shutil.rmtree(output_result_path)
os.mkdir(output_result_path)
#evaluate psnr,snr and worst
evaluate(float_folder_path, quantized_folder_path, output_result_path)
