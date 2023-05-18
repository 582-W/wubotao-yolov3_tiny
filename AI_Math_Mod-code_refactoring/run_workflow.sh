#!/bin/bash
# $1 input_case_folder
# $2 input_onnx_file
# $3 input_img_folder
# $4 hardware_config_file
# $5 img_preprocess_method
# $6 output_per_layer
# $7 log_level
# $8 mAP_model_type
# $9 whether_run_float
# $10 analysis_mode

delFolder(){
    if [ -d $1 ]; then 
        if [ -L $1 ]; then
            # It is a symlink!
            # Symbolic link specific commands go here.
            rm $1
        else
            # It's a directory!
            # Directory command goes here.
            rm -r $1
        fi
    fi
}


delFile(){
    if [ -e $1 ];then
        rm $1
    fi
}

chk_flow () {
	if [ $1 -ne 0 ]; then
		echo "[FAULT] $2" >> $ResultDir/log.txt
		exit 1
	fi
}

createFolderIfNo(){
    if [ ! -d "$1" ]; then
        mkdir $1
    fi
}




#parse input params
echo "parse input params..."
python3 /home/wbt/work/AI_Math_Mod-code_refactoring/utils/extract_input_params.py -i /home/wbt/work/AI_Math_Mod-code_refactoring/yolov3_tiny.json -o /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt
chk_flow $? "parse input params fails!"

input_case_folder=$(sed -n '1 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
input_onnx_file=$(sed -n '2 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
input_img_folder=$(sed -n '3 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
hardware_config_file=$(sed -n '4 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
img_preprocess_method=$(sed -n '5 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
output_per_layer=$(sed -n '6 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
log_level=$(sed -n '7 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
whether_run_mAP=$(sed -n '8 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
mAP_model_type=$(sed -n '9 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
whether_run_float=$(sed -n '10 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
analysis_mode=$(sed -n '11 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
whether_run_update=$(sed -n '12 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
concat_input_change=$(sed -n '13 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
whether_limit_shift=$(sed -n '14 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
shift_upper_limit=$(sed -n '15 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
shift_lower_limit=$(sed -n '16 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
img_target_size=$(sed -n '17 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
keep_aspect_ratio=$(sed -n '18 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
channel_num=$(sed -n '19 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
list_of_customized_name=$(sed -n '20 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
whether_run_iee=$(sed -n '21 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
iee_output_path=$(sed -n '22 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)
iee_npu_name=$(sed -n '23 p' /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt)


rm /home/wbt/work/AI_Math_Mod-code_refactoring/input_params.txt

weight_analysis_folder_path="$input_case_folder/weight_analysis_output"
weight_json_file_path="$weight_analysis_folder_path/weight_analysis.json"

datapath_analysis_folder_path="$input_case_folder/datapath_analysis_output"
datapath_json_file_path="$datapath_analysis_folder_path/datapath_analysis.json"

fl_output_folder_path="$input_case_folder/float_inference_output"

fp_output_folder_path="$input_case_folder/fp_inference_output"

model_update_folder_path="$input_case_folder/model_update_output"
updated_onnx="$model_update_folder_path/update.onnx"
updated_json="$model_update_folder_path/update.json"

float_detection_path="$input_case_folder/float_detection_output"
fp_detection_path="$input_case_folder/fp_detection_output"

mAP_result_path="$input_case_folder/mAP_result"

log_folder_path="$input_case_folder/log"

fl_detection_path="$input_case_folder/float_detection_output"
draw_output_path="$input_case_folder/drawed_images"

createFolderIfNo $input_case_folder


if [ $whether_run_float == "True" ] || [ $whether_run_float == 'true' ]
then
    delFolder $input_case_folder
    mkdir $input_case_folder
fi 
if [ $whether_run_mAP == "True" ] || [ $whether_run_mAP == 'true' ]
then
    delFolder $draw_output_path
    mkdir $draw_output_path
fi 
delFolder $log_folder_path
mkdir $log_folder_path

if [ -e $hardware_config_file ]
then
    cp $hardware_config_file $input_case_folder/
else
    echo "$hardware_config_file doesn't exist."
    exit
fi

echo "whether_run_update: $whether_run_update"
echo "concat_input_change: $concat_input_change"
echo "analysis_mode: $analysis_mode"

python3 interface_preprocess.py -m img_preprocess -ic $input_case_folder -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num 

if [ $analysis_mode == "per_layer" ]
then
    if [ $whether_run_float == "True" ] || [ $whether_run_float == 'true' ]
    then
        printf "\n\n\n"
        echo "=================================="
        echo "------ start weight_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m weight_analysis  -ic $input_case_folder -am $analysis_mode -io $input_onnx_file -l $log_level -lf "$log_folder_path/weight_analysis.log"
        chk_flow $? "weight_analysis fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start datapath_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m datapath_analysis -ic $input_case_folder -io $input_onnx_file -am $analysis_mode -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -lf "$log_folder_path/datapath_analysis.log" -ca $concat_input_change -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "datapath_analysis fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start model_update"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m model_update      -ic $input_case_folder -io $input_onnx_file -wj $weight_json_file_path -dj $datapath_json_file_path -l $log_level -uo $updated_onnx -uj $updated_json -c $hardware_config_file -ii $input_img_folder -m2 $img_preprocess_method -ru $whether_run_update -ca $concat_input_change -ls $whether_limit_shift -sul $shift_upper_limit -sll $shift_lower_limit -its $img_target_size -am $analysis_mode -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "model_update fails!"
	
	

        printf "\n\n\n"
        echo "=================================="
        echo "------ start float_inference"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m float_inference   -ic $input_case_folder -io $input_onnx_file -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -pl $output_per_layer -lf "$log_folder_path/float_inference.log" -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "float_inference fails!"
    fi
    
elif [ $analysis_mode == "per_channel_weight" ]
then
    echo "whether_run_float: $whether_run_float"
    if [ $whether_run_float == "True" ] || [ $whether_run_float == 'true' ]
    then
        printf "\n\n\n"
        echo "=================================="
        echo "------ start weight_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m weight_analysis  -ic $input_case_folder -io $input_onnx_file -l $log_level -lf "$log_folder_path/weight_analysis.log" -am $analysis_mode
        chk_flow $? "weight_analysis fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start datapath_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m datapath_analysis -ic $input_case_folder -io $input_onnx_file -am $analysis_mode -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -lf "$log_folder_path/datapath_analysis.log" -ca $concat_input_change -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "datapath_analysis fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start model_update"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m model_update      -ic $input_case_folder -io $input_onnx_file -wj $weight_json_file_path -dj $datapath_json_file_path -am $analysis_mode -l $log_level -uo $updated_onnx -uj $updated_json -c $hardware_config_file -ii $input_img_folder -am $analysis_mode -m2 $img_preprocess_method -ru $whether_run_update -ca $concat_input_change -ls $whether_limit_shift -sul $shift_upper_limit -sll $shift_lower_limit -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "model_update fails!"
	
        printf "\n\n\n"
        echo "=================================="
        echo "------ start float_inference"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m float_inference   -ic $input_case_folder -io $input_onnx_file -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -pl $output_per_layer -lf "$log_folder_path/float_inference.log" -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "float_inference fails!"
    fi

else
        printf "\n\n\n"
        echo "=================================="
        echo "------ start datapath_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m datapath_analysis -ic $input_case_folder -io $input_onnx_file -am $analysis_mode -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -lf "$log_folder_path/datapath_analysis.log" -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "datapath_analysis fails!"   

        printf "\n\n\n"
        echo "=================================="
        echo "------ start float_inference"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m float_inference   -ic $input_case_folder -io $input_onnx_file -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -pl $output_per_layer -lf "$log_folder_path/float_inference.log" -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
        chk_flow $? "float_inference fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start sw_model_update"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m sw_model_update  -ic $input_case_folder -io $input_onnx_file  -dj $datapath_json_file_path -c $hardware_config_file -l $log_level -uo $updated_onnx
        chk_flow $? "sw_model_update fails!" 


        printf "\n\n\n"
        echo "=================================="
        echo "------ start sw_weight_analysis"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m sw_weight_analysis  -ic $input_case_folder -uo $updated_onnx -l $log_level -lf "$log_folder_path/weight_analysis.log"
        chk_flow $? "weight_analysis fails!"

        printf "\n\n\n"
        echo "=================================="
        echo "------ start create_json"
        echo "=================================="
        printf "\n"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 interface.py -m create_json  -ic $input_case_folder -wj $weight_json_file_path -dj $datapath_json_file_path -l $log_level -uj $updated_json
        chk_flow $? "create_json fails!" 

fi

printf "\n\n\n"
echo "=================================="
echo "------ start fp_inference"
echo "=================================="
printf "\n"
cd /home/wbt/work/AI_Math_Mod-code_refactoring  && python3 interface.py -m fp_inference      -ic $input_case_folder -uo $updated_onnx -am $analysis_mode -uj $updated_json -c $hardware_config_file -ii $input_img_folder -m2 $img_preprocess_method -l $log_level -pl $output_per_layer -lf "$log_folder_path/fp_inference.log" -its $img_target_size -kar $keep_aspect_ratio -cn $channel_num -lcn $list_of_customized_name
chk_flow $? "fp_inference fails!"

printf "\n\n\n"
echo "=================================="
echo "------ start model_evaluate"
echo "=================================="
printf "\n"
cd /home/wbt/work/AI_Math_Mod-code_refactoring  && python3 interface.py -m model_evaluate    -ic $input_case_folder -ofl $fl_output_folder_path -ofp $fp_output_folder_path/float_res -l $log_level -lf "$log_folder_path/model_evaluator.log" -uo $updated_onnx
chk_flow $? "model_evaluate fails!"

printf "\n\n\n"
echo "=================================="
echo "------ start calculating mAP"
echo "=================================="
printf "\n"
echo "mAP_model_type: $mAP_model_type"

if [ $mAP_model_type == 'classification' ]
then
    class_file_path="./imagenet_label.json"
    ground_truth_path="./imagenet_groundtruth.json"
    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 classification.py --image_dir $input_img_folder --input_dir "$fp_output_folder_path/float_res"  --output_dir $fp_detection_path  --label $class_file_path --save_path "$draw_output_path/fp" --ground_truth $ground_truth_path
        chk_flow $? "fp detection fails!"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 classification.py --image_dir $input_img_folder --input_dir "$fl_output_folder_path"  --output_dir $fl_detection_path  --label $class_file_path --save_path "$draw_output_path/fl" --ground_truth $ground_truth_path
        chk_flow $? "fp detection fails!"
    fi
fi


if [ $mAP_model_type == 'yolo_v3' ] || [ $mAP_model_type == 'tiny_yolo_v3' ]
then
    if [ $mAP_model_type == 'yolo_v3' ] 
    then
        anchor_file_path="./model_data/yolo_anchors.txt"
        class_file_path="./model_data/coco_classes.txt"
    else
        anchor_file_path="./model_data/tiny_yolo_anchors.txt"
        class_file_path="./model_data/coco_classes.txt"
    fi
    echo "anchor_file_path: $anchor_file_path"
    echo "class_file_path: $class_file_path"
    echo "mAP_model_type: $mAP_model_type"
    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/detection_result && python3 yolo_output2standard.py -i $input_img_folder -o $fl_output_folder_path -d $float_detection_path  -a $anchor_file_path -c $class_file_path -op $mAP_model_type
        chk_flow $? "float detection fails!"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/detection_result && python3 yolo_output2standard.py -i $input_img_folder -o "$fp_output_folder_path/float_res" -d "$fp_detection_path"  -a $anchor_file_path -c $class_file_path -op $mAP_model_type
        chk_flow $? "fp detection fails!"
    fi
fi

if [ $mAP_model_type == 'ssd_keras' ] || [ $mAP_model_type == 'ssd_caffe' ]
then
    if [ $mAP_model_type == 'ssd_keras' ]
    then
        class_file_path="./kerasssd_class.txt" 
    else
        class_file_path="./caffessd_class.txt"
    fi
    
    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 ssd_postprocess.py --image_dir $input_img_folder --input_dir $fl_output_folder_path --output_dir $float_detection_path  --class_dir $class_file_path
        chk_flow $? "float detection fails!"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 ssd_postprocess.py --image_dir $input_img_folder --input_dir "$fp_output_folder_path/float_res"  --output_dir $fp_detection_path  --class_dir $class_file_path
        chk_flow $? "fp detection fails!"
    fi
fi

if [ $mAP_model_type == 'yolov4_pytorch' ]
then

    class_file_path="./coco_classes.txt"

    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 yolov4_postprocess.py --image_dir $input_img_folder --input_dir $fl_output_folder_path --output_dir $float_detection_path  --class_dir $class_file_path
        chk_flow $? "float detection fails!"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 yolov4_postprocess.py --image_dir $input_img_folder --input_dir "$fp_output_folder_path/float_res"  --output_dir $fp_detection_path  --class_dir $class_file_path
        chk_flow $? "fp detection fails!"
    fi
fi

if [ $mAP_model_type == 'yolov3_pytorch' ]
then

    class_file_path="./coco_classes.txt"

    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess

        python3 yolov3_postprocess.py --input_case_folder $input_case_folder --image_dir $input_img_folder --input_dir $fl_output_folder_path --output_dir $float_detection_path  --class_dir $class_file_path

        chk_flow $? "float detection fails!"

        python3 yolov3_postprocess.py --input_case_folder $input_case_folder --image_dir $input_img_folder --input_dir "$fp_output_folder_path/float_res"  --output_dir $fp_detection_path  --class_dir $class_file_path

        chk_flow $? "fp detection fails!"
    fi
fi

if [ $mAP_model_type == 'yolov5_pytorch' ]
then

    class_file_path="./coco_classes.txt"
    # class_file_path="./yolov5n-mask.txt"

    if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;}
    then
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 yolov5_postprocess.py --input_case_folder $input_case_folder --image_dir $input_img_folder --input_dir $fl_output_folder_path --output_dir $float_detection_path  --class_dir $class_file_path
        chk_flow $? "float detection fails!"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/postprocess && python3 yolov5_postprocess.py --input_case_folder $input_case_folder --image_dir $input_img_folder --input_dir "$fp_output_folder_path/float_res"  --output_dir $fp_detection_path  --class_dir $class_file_path
        chk_flow $? "fp detection fails!"
    fi
fi


if { [ $output_per_layer == 'False' ] || [ $output_per_layer == 'false' ] ;} && { [ $whether_run_mAP == 'True' ] || [ $whether_run_mAP == 'true' ] ;} && { [ $mAP_model_type != 'classification' ] || [ $mAP_model_type != 'classification' ] ;}
then

    if [ $mAP_model_type == 'yolov4_pytorch' ] || [ $mAP_model_type == 'yolov5_pytorch' ]
    then
        mkdir $mAP_result_path
        echo "floating inference mAP evaluation"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/weight_analysis/yolo_mAP_evaluation_framework/Object-Detection-Metrics && python3 pascalvoc.py -gt "groundtruths_coco2017val" -np -gtformat xyrb -detformat xyrb -det $float_detection_path > "$mAP_result_path/float_mAP.txt"
        chk_flow $? "float mAP fails!"

        echo "fp inference mAP evaluation"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/weight_analysis/yolo_mAP_evaluation_framework/Object-Detection-Metrics && python3 pascalvoc.py -gt "groundtruths_coco2017val" -np -gtformat xyrb -detformat xyrb -det $fp_detection_path > "$mAP_result_path/fp_mAP.txt"
        chk_flow $? "fp mAP fails!"
    else
        mkdir $mAP_result_path
        echo "floating inference mAP evaluation"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/weight_analysis/yolo_mAP_evaluation_framework/Object-Detection-Metrics && python3 pascalvoc.py -np -gtformat xyrb -detformat xyrb -det $float_detection_path > "$mAP_result_path/float_mAP.txt"
        chk_flow $? "float mAP fails!"

        echo "fp inference mAP evaluation"
        cd /home/wbt/work/AI_Math_Mod-code_refactoring/weight_analysis/yolo_mAP_evaluation_framework/Object-Detection-Metrics && python3 pascalvoc.py -np -gtformat xyrb -detformat xyrb -det $fp_detection_path > "$mAP_result_path/fp_mAP.txt"
        chk_flow $? "fp mAP fails!"
    fi
    cd /home/wbt/work/AI_Math_Mod-code_refactoring/weight_analysis/yolo_mAP_evaluation_framework/Object-Detection-Metrics && python3 draw.py  --image_dir $input_img_folder --fp_dir $fp_detection_path --fl_dir $fl_detection_path --output_dir $draw_output_path
fi
echo "fp inference mAP evaluation"
echo "$whether_run_iee"
echo "fp inference mAP evaluation"
if { [ $whether_run_iee == 'True' ] || [ $whether_run_iee == 'true' ] ;}
then
    echo "fp  mAP evaluation"
    cd /home/wbt/work/W_BT/testcases/iee && python3 run_iee.py --onnx_file "$input_case_folder/model_update_output/update.onnx" --img_file "$input_case_folder/img_npy/input_int" --scale_file "$input_case_folder/model_update_output/update.json" --output_path $iee_output_path --npu_name $iee_npu_name
    
    cd /home/wbt/work/AI_Math_Mod-code_refactoring && python3 iee_compare.py --fp_folder "$input_case_folder/fp_inference_output/int_res" --iee_folder "$iee_output_path/inference_files/data_out/int_res" --fp_folder_fl "$input_case_folder/fp_inference_output/float_res" --iee_folder_fl "$iee_output_path/inference_files/data_out/float_res" --file_name $input_case_folder

fi