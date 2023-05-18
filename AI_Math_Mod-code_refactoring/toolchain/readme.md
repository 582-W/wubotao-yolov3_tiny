v0.1.0  

v0.2.0  
2021.06.17  
1.支持caffe  
2.为每个模型（yolov3,ssd,resnet50,mobilenetv2,vgg16）提供run.sh和input_params.json  
3.整理文件结构  
``````
/workspace
|--Converter
|--AI_Math_Mod
|--Compiler
|--example
    |--example
    |--script
    |--config
    |--model
    |--dataset
``````

v0.3.0  
2021.07.12  
1.新安装python3.6及所需的第三方库，原python3.5不满足新的开发需求  
2.更新了AI_Math_Mod(版本:request #173),支持initializer和constant node两种权重参数存放方式,并修改了一些bug   
v0.3.1  
2021.07.13  
1.安装jupyter  

v0.4.1  
2021.09.17
1.隐藏了weight_analysis,datapath_analysis,model_update,inference的代码
