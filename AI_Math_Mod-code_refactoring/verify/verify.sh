# verify for 5 models(ssd/vgg16/mobilenetv2/yolov3/resnet50), 2 analysis mode(per channel weight/per layer), whether run model update(true/false)
# how to run this shell script: cd verify && bash verify.sh
config_list=(yolov3_pc_false.json yolov3_pc_true.json yolov3_pl_false.json yolov3_pl_true.json \
             yolov3tiny_pc_false.json yolov3tiny_pc_true.json yolov3tiny_pl_false.json yolov3tiny_pl_true.json \
             mobilenet_pc_false.json mobilenet_pc_true.json mobilenet_pl_false.json mobilenet_pl_true.json \
             vgg_pc_false.json vgg_pc_true.json vgg_pl_false.json vgg_pl_true.json \
             resnet50_pc_false.json resnet50_pc_true.json resnet50_pl_false.json resnet50_pl_true.json \
             resnet34_pc_false.json resnet34_pc_true.json resnet34_pl_false.json resnet34_pl_true.json \
             ssd_pc_false.json ssd_pc_true.json ssd_pl_false.json ssd_pl_true.json)
for i in ${config_list[@]}
do
    cp config/$i input_params.json
    bash run_workflow.sh
    rm -rf input_params.json
done

# verify
python draft.py
