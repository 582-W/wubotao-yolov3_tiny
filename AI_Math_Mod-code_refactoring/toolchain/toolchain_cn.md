# Toolchain介绍  

## 目录
- [Toolchain介绍](#toolchain--)
  * [0.简介](#0--)
  * [1.安装](#1--)
    + [1.1 系统要求](#11-----)
    + [1.2 docker安装](#12-docker--)
    + [1.3 获取toolchain image](#13---toolchain-image)
    + [1.4 启动 toolchain image](#14----toolchain-image)
  * [2.Docker 文件](#2docker---)
    + [2.1文件夹结构](#21-----)
  * [3.toolchain使用方法](#3------)
    + [3.1 Converter](#31-converter)
      - [3.1.1 keras 转为 onnx](#311-keras-to-onnx)
      - [3.1.2 caffe 转为 onnx](#312-caffe-to-onnx)
      - [3.1.3 pytorch 转为 onnx](#313-pytorch-to-onnx)
      - [3.1.4 tensorflow 转为 onnx](#314-tensorflow-to-onnx)
      - [3.1.5 tf lite 转为 onnx](#315-tf-lite-to-onnx)
      - [3.1.6 onnx 转为 onnx](#316-onnx-to-onnx)
      - [3.1.7 model editor](#317-model-editor)
      - [3.1.8 Converter 流程示例](#318-converter-----)
    + [3.2 AI_Math_Mod](#32-ai-math-mod)
      - [3.2.1 Workflow 和 输出文件](#321-workflow-and-output-file)
      - [3.2.2 操作步骤](#322-----)
      - [3.2.3 参数配置](#323-----)
    + [3.3 Compiler](#33-compiler)
    + [3.4 workflow script](#34-workflow-script)
  * [4.FAQ](#4faq)
  * [5.Appendix](#5appendix)
    + [5.1 可配置的参数](#51-------)
      - [5.1.1 input_params.json](#511-input-paramsjson)
      - [5.1.2 hardware.json](#512-hardwarejson)


## 0.简介

&ensp;&ensp;目前我们的toolchain包含三部分：**Converter**, **AI_Math_Mod** 和 **Compiler**

&ensp;&ensp;**Converter** 将从深度学习框架（目前toolchain支持Keras、Caffe、Pytorch）生成的模型文件转换为onnx格式，并对模型执行一些优化。

&ensp;&ensp;**AI_Math_Mod** 将优化后的onnx model和dataset作为输入，分析动态定点数方法的有效性，然后生成经过优化的onnx模型和json文件供编译器使用。

&ensp;&ensp;**Compiler** 产生用于硬件仿真器使用的二进制文件

&ensp;&ensp;我们提供给一个docker，在上面运行toolchain，以简化使用的复杂性

<p align="center"><img src="pics/toolchain_figure.png" width="600"\></p>  
<p align="center">Fig.1 toolchain的运行流程</p>  

>在本文档中，您将了解到：

- 如何安装和使用包含toolchain的docker（请参阅第1节“安装”）。

+ toolchain中有哪些工具（请参见第2节Docker文件）。

* 如何使用脚本来使用这些工具（请参阅第3节脚本的使用）。

## 1.安装

### 1.1 系统要求

ubuntu 16+ 或 Windows 10 professional

### 1.2 docker安装

安装方法参照以下

(1) Windows https://docs.docker.com/docker-for-windows/install/

(2) Ubuntu https://docs.docker.com/engine/install/ubuntu/

### 1.3 获取toolchain image

(1)在本地登录dockerhub账户

```shell
sudo docker login -u username
```

(2)获取image

```shell
sudo docker pull wangl98/toolchain:v0.4.1
```

(3)检查image是否被成功获取

```shell
sudo docker image ls			
```

### 1.4 启动 toolchain image

(1)启动image

```shell
sudo docker run -it --rm -v absolute_path_of_your_folder:/data1 wangl98/toolchain:v0.4.1
```

`absolute_path_of_your_folder`是您挂载到docker容器中的文件夹路径，`/data1`是docker中的挂载文件夹，`wangl98/toolchain:v0.3.1`是我们的toolchain名字，这样我们可以在主机中访问需要的文件，并在docker中保存下来，例如：

```shell
sudo docker run -it --rm -v /home/nju/Downloads/docker_mount:/data1 wangl98/toolchain:v0.4.1
```

(2)将`/workspace/example/`里的文件拷贝到挂载文件夹下

```shell
cp -r /workspace/example/script/ /data1/ && cp -r /workspace/example/config/ /data1/ && cp -r /workspace/example/model/ /data1/ && cp -r /workspace/example/dataset/* /data1/ && cp -r /workspace/example/example/* /data1/
```

## 2.Docker 文件

### 2.1文件夹结构

进入容器之后，您应该位于`/workspace`路径下，所有的工具都位于该文件夹中。文件夹的结构和描述如下：

```shell
/workspace
|--Converter: 将原始模型转换为优化的onnx模型
|--AI_Math_Mod: 分析onnx模型，提供动态定点推理信息
|--Compiler: 生成硬件所需的二进制文件
|--example: 示例文件
    |--example: yolov3_tiny运行整个进程的完整示例文件(包含模型、配置和脚本)
    |--script: shell脚本示例用于运行各种模型的完整进程，包括Converter、AI_Math_Mod和Compiler
    |--config: 为各种模型配置示例
    |--model: 各种模型 
    |--dataset: 数据集文件
```

## 3.toolchain使用方法

### 3.1 Converter

模型转换的步骤如下：

(1)使用我们的Converter将model转换为onnx格式

(2)使用`onnx2onnx.py`优化onnx文件，如果model是从pytorch导出的那么可以略过这一步，因为`pytorch2onnx.py`包含了`onnx2onnx.py`中的绝大部分优化。

(3)检查模型，并使用`editor.py`对model做进一步的自定义

> /data1/model/中有可以进行实验的示例&ensp;&ensp;

#### 3.1.1 keras 转为 onnx

对于keras而言，我们仅支持从`keras 2.2.4`导出的model。需要指出的是`keras 2.3`和`tf.keras`是不支持的，您或许需要先将其导出为tflite model。

对应的转换命令为：

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o path_of_output_onnx_file path_of_input_keras_file -O 3
```

例如：

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o /data1/model/yolov3_416.h5.onnx /data1/model/yolov3_416.h5 -O 3
```

可选项`-O [level]`:执行特定级别的优化。`-O 1`意味着将消除dropout层；`-O 2` 意味着将paddings和下一层融合在一起，并且将model中的 average pooling 置换为 global average poolibng；`-O 3`意味着将batch normalization和convolution layer融合在一起。高级别的优化将包含低级别里的优化，例如，`-O 3`包含`-O 2`和`-O 1`里的所有优化。

#### 3.1.2 caffe 转为 onnx

对于caffe而言，我们仅支持可以被 intel Caffe 1.0 加载的模型

对应的转换命令为：

```shell
cd /workspace/Converter && python3 caffe-onnx/generate_onnx.py -o path_of_output_onnx_file -n path_of_input_prototxt_file -w path_of_input_caffemodel_file
```

例如：

```shell
cd /workspace/Converter && python3 caffe-onnx/generate_onnx.py -o /data1/model/ssd.caffemodel.onnx -n /data1/model/deploy.prototxt -w /data1/model/ssd.caffemodel
```

#### 3.1.3 pytorch 转为 onnx

`pytorch2onnx.py`不仅支持`pth`文件作为输入，也支持将从pytorch导出的`onnx`文件作为输入，事实上。由于pytorch保存和加载model的API不是很好，所以我们更建议将`onnx`文件作为输入。您可以额外查看将pytorch model导出为onnx的方法。对于`pth`和`onnx`，我们都仅支持从version>=1.0.0,<1.6.0的pytorch导出的model。 

(1)提示

> 您可以使用`torch.onnx`将您的模型导出为onnx，在pytorch 1.3.0版本下，转换的示例代码为：

```python
import torch.onnx
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'output.onnx', keep_initializers_as_inputs=True, opset_version=9)
```

在上面的例子中，`(1,3,224,224)`中的每一个维度分别代表batch size、input channel、input height、input width，`model`是想要导出的pytorch model，`output.onnx`是输出的文件，对于1.3.0版本以下的pytorch，不需要参数`keep_initializers_as_inputs=True`。

(2)当输入是`pth`文件时，命令如下：

```shell
cd /workspace/Converter && python3 optimizer_scripts/pytorch2onnx.py path_of_input_path_file path_of_output_onnx_file --input-size input_shape_of_input_pytorch_model 
```

例如：

```shell
cd /workspace/Converter && python3 optimizer_scripts/pytorch2onnx.py /data1/model/resnet50.pth /data1/model/resnet50.pth.onnx --input-size 3 224 224
```

(3)当输入是`onnx`文件时，运行`pytorch2onnx.py`以优化`onnx`,命令如下：

```shell
cd /workspace/Converter && python3 optimizer_scripts/pytorch2onnx.py path_of_input_onnx_file path_of_output_onnx_file
```

#### 3.1.4 tensorflow 转为 onnx

我们仅支持1.x版本的tensorflow而且支持的operator也非常有限，如果这不能满足您的需要，您可以尝试先将tensorflow model导出为tflite ，再进行后续的转换。

命令如下：

```shell
cd /workspace/Converter && python3 optimizer_scripts/tensorflow2onnx.py path_of_input_tensorflow_file path_of_output_onnx_file
```

#### 3.1.5 tf lite 转为 onnx

命令如下：

```shell
cd /workspace/Converter && python3 tflite-onnx/onnx_tflite/tflite2onnx.py -tflite path_of_input_tflite_file -save_path path_of_output_onnx_file -release_mode True
```

#### 3.1.6 onnx 转为 onnx

如果您的onnx model是从以上未提到的其他的框架中转换得到的，或者是从一些在线网站下载得到的，那么您需要执行以下的命令来优化您的onnx model

命令如下：

```shell
cd /workspace/Converter && python3 optimizer_scripts/onnx2onnx.py path_of_input_onnx_file -o path_of_output_onnx_file
```

#### 3.1.7 model editor

我们的NPU支持计算绝大多数需要密集计算的算子，例如Convolution，BatchNormalization，Fully Connect/GEMM。另一方面，我们也有一些算子支持的不是很好，例如Softmax 和 Sigmod。，但是这些算子不需要大量的计算资源，可以由CPU完成计算，因为我们提供了一个名为`editor.py`的model editor来帮助用户更改model，以使其在我们的NPU上运行地更加高效。

(1)editor.py文件位于`/workspace/Converter/optimizer_scripts`路径下，它是一个简单的ONNX编辑器，实现了以下的功能：

- 增加nop BN 或 Conv nodes

- 删除某些nodes或者inputs
- 删除某个节点下的图（删除某个节点下的所有节点）
- 重塑（Reshape）输入和输出
- 重命名（Rename）输出
- 重命名输入，AI_Math_Mod 仅仅支持model的input name包含字符串`input`的情形，使用`--rename-input`,input name 'xxx' 会被更改为 'xxx_input`
- 重命名BatchNormalization layer的输入，AI_Math_Mod 仅仅支持the constant inputs(scale,B,mean,var) name of BN node包含字符串`gamma` ,`beta`,`mean`,`var`的情形， 使用`--rename-bn`, 原来的constant inputs(scale,B,mean,var) name 'xxx'会被更改为'xxx_gamma','xxx_beta','xxx_mean',or 'xxx_var'
- 为所有节点添加名字，如果模型的所有节点都没有名字，使用`--add-all-node-name`, 节点名称将会和节点的输出名称一致

(2)用法：

```shell
usage: editor.py [-h] [-c CUT_NODE [CUT_NODE ...]]
                 [--cut-type CUT_TYPE [CUT_TYPE ...]]
                 [-d DELETE_NODE [DELETE_NODE ...]]
                 [--delete-input DELETE_INPUT [DELETE_INPUT ...]]
                 [-i INPUT_CHANGE [INPUT_CHANGE ...]]
                 [-o OUTPUT_CHANGE [OUTPUT_CHANGE ...]]
                 [--add-conv ADD_CONV [ADD_CONV ...]]
                 [--add-bn ADD_BN [ADD_BN ...]]
                 [--rename-output RENAME_OUTPUT [RENAME_OUTPUT ...]]
                 [--rename-input RENAME_INPUT [RENAME_INPUT ...]]
                 [--add-all-node-name ADD_ALL_NODE_NAME [ADD_ALL_NODE_NAME ...]]
                 [--rename-bn RENAME_BN [RENAME_BN ...]]
                 in_file out_file

编辑ONNX模型。处理顺序是 '删除节点/值' -> '添加节点' -> '改变形状'。 切割（Cutting）不能与其他操作一起进行

位置参数:
  in_file               input ONNX FILE
  out_file              ouput ONNX FILE

可选参数:
  -h, --help            显示此帮助信息并退出
  -c CUT_NODE [CUT_NODE ...], --cut CUT_NODE [CUT_NODE ...]
                        从给定的节点中删除节点（包括）
  --cut-type CUT_TYPE [CUT_TYPE ...]
                        从给定的节点(包括)中按类型删除节点
  -d DELETE_NODE [DELETE_NODE ...], --delete DELETE_NODE [DELETE_NODE ...]
                        按名称删除节点，并且只删除这些节点
  --delete-input DELETE_INPUT [DELETE_INPUT ...]
                        按名称删除输入
  -i INPUT_CHANGE [INPUT_CHANGE ...], --input INPUT_CHANGE [INPUT_CHANGE ...]
                        改变输入形状 (e.g. -i 'input_0 1 3 224 224')
  -o OUTPUT_CHANGE [OUTPUT_CHANGE ...], --output OUTPUT_CHANGE [OUTPUT_CHANGE ...]
                        改变输出形状 (e.g. -o 'input_0 1 3 224 224')
  --add-conv ADD_CONV [ADD_CONV ...]
                        使用特定的输入添加nop conv
  --add-bn ADD_BN [ADD_BN ...]
                        使用特定的输入添加nop bn  
  --rename-output RENAME_OUTPUT [RENAME_OUTPUT ...]
                        重命名特定的输出 (e.g. --rename-output old_name new_name)
  --rename-input RENAME_INPUT [RENAME_INPUT ...]
                        重命名模型的输入
  --add-all-node-name ADD_ALL_NODE_NAME [ADD_ALL_NODE_NAME ...]
                        为所有节点添加名称
  --rename-bn RENAME_BN [RENAME_BN ...]
                        重命名BatchNormalization层的输入
```

#### 3.1.8 Converter 流程示例

在`/data1/model/`路径下有各种模型可以用来尝试

(1)yolov3_416.h5

转换为onnx模型:

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o /data1/model/yolov3_416.onnx /data1/model/yolov3_416.h5 -O 3  
```

优化onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/onnx2onnx.py /data1/model/yolov3_416.onnx -o /data1/model/yolov3_416_optimized.onnx 
```

(2)yolov3_tiny_416.h5

转换为onnx模型:

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o /data1/model/yolov3_tiny_416.onnx /data1/model/yolov3_tiny_416.h5 -O 3
```

优化onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/onnx2onnx.py /data1/model/yolov3_tiny_416.onnx -o /data1/model/yolov3_tiny_416_optimized.onnx
```

(3)mobilenet_v2.h5

转换为onnx模型:

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o /data1/model/mobilenet.onnx /data1/model/mobilenet_v2.h5 -O 3  
```

优化onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/onnx2onnx.py /data1/model/mobilenet.onnx -o /data1/model/mobilenet_optimized.onnx  
```

编辑模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/editor.py /data1/model/mobilenet_optimized.onnx  /data1/model/mobilenet_edited.onnx --cut-type GlobalAveragePool --rename-output "out_relu_o0" "output"  
```

(4)vgg16.h5

转换为onnx模型:

```shell
cd /workspace/Converter && python3 keras-onnx/generate_onnx.py -o /data1/model/vgg16.onnx /data1/model/vgg16.h5 -O 3  
```

优化onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/onnx2onnx.py /data1/model/vgg16.onnx -o /data1/model/vgg16_optimized.onnx  
```

编辑模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/editor.py /data1/model/vgg16_optimized.onnx /data1/model/vgg16_edited.onnx --cut-type Flatten --rename-output "block5_pool_o0" "output"  
```

(5)ssd.caffemodel

转换为onnx模型:

```shell
cd /workspace/Converter && python3 caffe-onnx/generate_onnx.py -o /data1/model/ssd_caffe.onnx -n /data1/model/deploy.prototxt -w /data1/model/ssd.caffemodel  
```

优化onnx模型:

```shell
cd /workspace/Converter && python3 onnx2onnx2.py /data1/model/ssd_caffe.onnx -o /data1/model/ssd_caffe_optimized.onnx -t Concat Reshape Identity Transpose Flatten Dropout Mystery Constant Squeeze Unsqueeze Softmax
```

(6)resnet34.pth

转换为onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/pytorch2onnx.py /data1/model/resnet34.pth /data1/model/resnet34.onnx --input-size 3 224 224  
```

编辑模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/editor.py /data1/model/resnet34.onnx /data1/model/resnet34_edited.onnx --add-all-node-name --rename-input --rename-bn --cut-type GlobalAveragePool Unsqueeze --rename-output "340" "output"  
```

(7)resnet50.pth

转换为onnx模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/pytorch2onnx.py /data1/model/resnet50.pth /data1/model/resnet50.onnx --input-size 3 224 224  
```

编辑模型:

```shell
cd /workspace/Converter && python3 optimizer_scripts/editor.py /data1/model/resnet50.onnx /data1/model/resnet50_edited.onnx --add-all-node-name --rename-input --rename-bn --cut-type GlobalAveragePool Unsqueeze --rename-output "492" "output"  
```

### 3.2 AI_Math_Mod

#### 3.2.1 Workflow 和输出文件

(1)Workflow
<p align="center"><img src="pics/AI_Math_Mod.PNG" width="720"\></p>
<p align="center">Fig.2 AI_Math_Mod流程 </p>  

(2)输出文件

```
├── input_case_folder
│   ├── weight_analysis_output:  weight分析的输出结果
│   ├── datapath_analysis_output:  datapath分析的输出结果
│   ├── img_npy:  图像预处理的输出numpy文件
│   ├── model_update_output:  model_update的输出结果
│   ├── float_inference_output:  浮点推理的输出结果
│   ├── fp_inference_output:  定点推理的输出结果
│   ├── evaluator_output: 模型评估的输出结果
│   ├── float_detection_output: 浮点检测输出结果
│   ├── fp_detection_output: 定点检测输出结果
│   ├── mAP_result: 浮点mAP 和 定点mAP
│   └── log: 日志文件
```

#### 3.2.2 操作步骤

为了方便用户，AI_Math_Mod的使用被包装在`run_workflow.sh`中，您只需要完成以下的步骤就可以运行整个分析流程。

1. 设置`/data1/input_params.json`文件
2. 设置`/data1/hardware.json`文件
3. 执行命令`cd /workspace/AI_Math_Mod && ./run_workflow.sh`

#### 3.2.3 参数配置

1. input_params.json

一个例子如下：

```json
{
    "input_case_folder": "/data1/testcases",
    "input_onnx_file": "/data1/yolov3_tiny_416_optimized.onnx",
    "input_img_folder": "/data1/voc_3imgs/",
    "hardware_config_file": "/data1/hardware.json",
    "img_preprocess_method": "yolo_keras",
    "output_per_layer": false,
    "log_level": "warning",
    "whether_run_mAP": true,
    "mAP_model_type": "tiny_yolo_v3",
    "whether_run_float": true,
    "analysis_mode": "per_layer",
    "whether_run_update": true,
    "concat_input_change": true,   
    "whether_limit_shift":true,
    "shift_upper_limit":35,
    "shift_lower_limit":-4,
    "img_target_size": [416, 416],
    "keep_aspect_ratio": true,
    "channel_num": "RGB",
    "list_of_customized_name": ["identity", "inputmeannormalization", "inputchannelswap"],
    "whether_run_iee":false
}
```

2. hardware.json

一个例子如下：

```json
{
    "conv_bitwidth": {
        "kernel": 10,
        "bias": 16
    },
    "bn_bitwidth": 8,
    "working_bitwidth": 32,
    "datapath_bitwidth": 16,
    "leaky_relu_alpha": 16,
    "average_pool_radix": 16,
    "example_customized_config_conv2d_1":{
        "kernel": 10,
        "bias": 16
    }
}  
```

### 3.3 Compiler

Complier将`scale factor`和`optimized onnx`作为输入，输入用于硬件仿真器的bin文件。

可以使用以下命令运行Complier：

```shell
python3 Compiler.py -o onnx_file_path -hw hw_config_file_path -s scale_file_path -m memory_file_path -r result_path -l log_level  
```

参数解释：

```
可选参数:
        '-o', '--onnx_file_path', help="path of input onnx model file, to compiler into the binary files", default="",  
        '-hw', '--hw_config_file_path', help="path of hardware config file, which saves the hardware bitwidth info", default="",  
        '-s', '--scale_file_path', help="path of scaling_factor file, which saves the scaling factor info of the model", default="",  
        '-m', '--memory_file_path', help="path of memory config, which saves the hardware memory config", default="",  
        '-r', '--result_path', help="path of result file, which will save the outputs of the compiler", default="",  
        '-l', '--log_level', help="log_level, options: info, warning, error, critical", default="CRITICAL", 
```

例如：

```shell
python3 Compiler.py -o /data1/testcases/model_update_output/update.onnx -hw config/hardware.json -s /data1/testcases/model_update_output/update.json -m config/memory.json -r /data1/compiler_yolov3_tiny/ -l DEBUG
```

### 3.4 workflow script

`/data1/run.sh`是一个针对yolov3_tiny 的脚本示例，运行它可以运行所有步骤，包括Converter, AI_Math_Mod 和 Compiler

命令如下：

```shell
cd /data1 && bash run.sh
```

在这个例子中，参数配置的文件为`/data1/input_params.json`，用于推断的数据集为`voc_3imgs`，这个数据集包含了`voc_imgs`里的三张图片，因此mAP仅为1.2%，耗时大约400s（服务器上使用的是CPU：2 chip, 18 cores per chip, Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz）。如果您想用`voc_imgs`做推断需要改变相应的参数，这种情况下耗时大约2个小时。在`/data1/script/run_*.sh`下有针对其他模型的运行整个流程的更多的例子，`/data1/script/run_*.sh`对应的参数配置文件为`/data1/config/input_params_*.json`。

## 4.FAQ

## 5.Appendix

### 5.1 可配置的参数

#### 5.1.1 input_params.json

1. input_case_folder:这个文件夹用于存放AI_Math_Mode输入的结果，在每一次开始时它都会被清空，因此我们建议您在开始一次新的测试时使用一个新的文件夹来存放结果
2. input_onnx_file: 输入的ONNX 文件
3. input_img_folder：输入的数据集
4. hardware_config_file：硬件配置文件路径
5. img_preprocess_method：图片预处理的方式，可选项： yolo_keras / ssd_keras / vgg_keras / mobilenet_keras
6. output_per_layer：可选项： True/False。对模型进行推断时是否输出每一层的结果，如果True：modelEvaluator的输出将会包含每一层的SNR/PSNR，同时目标检测 和 mAP计算 将不起作用。如果False：modelEvaluator的输出将仅包含最后一层的SNR/PSNR
7. log_level：可选项：debug/info/critical
8. whether_run_mAP：可选项：True/False。如果True：目标检测 和 mAP计算 会被执行。如果False：目标检测 和 mAP计算 将不会被执行
9. mAP_model_type：可选项：yolo_v3/tiny_yolo_v3。此选项将在 目标检测 部分配置anchor.txt和classes.txt。现在仅支持标准的 yolo_v3 和 tiny_yolo_v3
10. whether_run_float：可选项：true/false。如果False： weigth_analysis, datapath_analysis, model_update 和 float_inference都 会被略过，以此来加速整个流程。如果True：这些步骤都会被执行。
    注意：如果该选项被设定为False，在运行`run_workflow.sh`之前，记得存储weigth_analysis, datapath_analysis, model_update and float_inference的输出结果，这些结果位于test case文件夹中，因为尽管略过了这些步骤，但是在计算 mAP 和 SNR时它们的结果依然会被利用。
11. analysis_mode：可选项：per layer/per channel weight。如果per layer：weight_analysis 将会计算每一层权重值的范围和分布。如果per channel weight：weight_analysis 将会计算每个通道的权重值的范围和分布
12. whether_run_update：可选项：true/false。如果True：AI_Math_Mod会生成经过优化的ONNX model
13. concat_input_change：可选项：true/false。
14. whether_limit_shift：可选项：true/false，默认值：true
15. shift_upper_limit：默认值：35
16. shift_lower_limit：默认值：-4
17. img_target_size：输入图片的大小
18. keep_aspect_ratio：可选项：true/false
19. channel_num：可选项：RGB/BGR

#### 5.1.2 hardware.json

1. conv_bitwidth：卷积层中卷积核参数和偏置的位宽
2. bn_bitwidth：BatchNormalization 层中A的位宽
3. working_bitwidth：每层中计算过程中的位宽
4. datapath_bitwidth：层之间的传播位宽
5. leaky_relu_alpha：LeakyRelu 层中alpha的位宽
6. average_pool_radix：AveragePool层中的位宽
7. example_customized_config_conv2d_1：特定的层的位宽



