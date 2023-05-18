# AI_Math_Mod

This is a framework to analyze the effectiveness of the dynamic fixed point method. It includes serveral parts: WeightAnalyser, DatapathAnalyser, ModelUpdater, ModelInferencer, ModelEvaluator, object detection and mAP evaluation. 

## 1. Prerequisites
(1) docker env: https://drive.google.com/file/d/1Q71Fm0BM4m4XOSD0TH9hDp9m9GnC-SWY/view?usp=sharing  
command:   
docker run -it -v path_of_docker_mounted_folder:/data1 johnanderson36555/converter:latest    
e.g. docker run -it -v /home/your_name/Downloads/docker_mount:/data1 -p 8888:8888 johnanderson36555/converter:latest

## 2. Introduction
### 2.1 WeightAnalyser
This is for analyze the distribution of the model's weight value

### 2.2 DatapathAnalyser
This library is for analyze the distribution of the datapath, which means the intermediate output of each layer of the model.

### 2.3 ModelUpdater
This library is for modifying the weight of the model for better inference, and combine the weight analysis and datapath analysis's result. Currently, there's no working in the modifying the weight of the model.

### 2.4 ModelInferencer
This library is for inference the model with dateset in different mode, i.e. floating mode and dynamic fixed point mode.

### 2.5 ModelEvalutor
This part is for evaluating the SNR/PSNR between the floating inference and dynamic fixed-point inference.

### 2.6 Object detection
This part is to convert the cnn outputs into object detection results, whose format is:
```
object_class probability x1 y1 x2 y2
```
### 2.7 mAP Evaluation
This part is to calculate the mAP result

## 3. Running tutorial
To simplify the workflow of the libs introduced in Part 2, there's a shell script named run_workflow.sh, and there are three steps for running the whole workflow:
### 3.1 configure input_params.json
(1) input_case_folder   
The path of the folder saving all the output info of AI_Math_Mode. The folder will be cleaned each time, thus it is recommended to use a new folder each time you want to start a new testcase. 

(2) input_onnx_file  
The path of onnx file to be analyzed.

(3) input_img_folder   
The path of the img folder to be analyzed.

(4) hardware_config_file   
The path of the hardware config json.

(5) img_preprocess_method   
Options: yolo ------- img / 255.0  
The method of img preprocess.   


(6) output_per_layer   
Option: Trur/False   
Whether output results of each layer when inferencing the model. If True, the output of modelEvaluator will contains SNR/PSNR of each layer, and the part of object detection and mAP evaluation will not work; otherwise, the output of modelEvaluator will only contains SNR/PSNR of the last layer.   


(7)log_level  
Option: debug/info/critical

(8)whether_run_mAP   
Option: True/False  
If True, the part of object detection and mAP calculation will work; otherwise, won't work.

(9)mAP_model_type  
option: yolo_v3/tiny_yolo_v3  
This option will configure the anchor.txt and classes.txt during the part of object detection. Currently, it only support the standard yolo_v3 and tiny_yolo_v3

(10)whether_run_float    
option: true/false   
If false, the step of weigth_analysis, datapath_analysis, model_update and float_inference will be ignored to accelerate the whole process; otherwise, all the steps will be run.    
Note: if set this option to be false, remember to store the result of weigth_analysis, datapath_analysis, model_update and float_inference in test case folder before running the script run_workflow.sh, because though these steps are ignored, the results of them will be utilized when calculating mAP and SNR.

### 3.2 run the script    
./run_workflow.sh

### 3.3 see the output

```bash
├── input_case_folder
│   ├── weight_analysis_output
│   ├── datapath_analysis_output
│   ├── img_npy
│   ├── model_update_output
│   ├── float_inference_output
│   ├── fp_inference_output
│   ├── float_detection_output
│   ├── fp_detection_output
│   ├── evaluator_output
│   └── mAP_result
```
