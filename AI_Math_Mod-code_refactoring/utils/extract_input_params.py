import json
import argparse
import os


def main_(args):
    input_file_path     = args.input_file
    output_file_path    = args.output_file

    input_file  = open(input_file_path)
    output_file = open(output_file_path, 'w+')
    

    dict_ = json.loads(input_file.read())
    print(dict_)

    output_file.write(dict_["input_case_folder"] + "\n") #1
    output_file.write(dict_["input_onnx_file"] + "\n") #2
    output_file.write(dict_["input_img_folder"] + "\n") #3
    output_file.write(dict_["hardware_config_file"] + "\n") #4
    output_file.write(dict_["img_preprocess_method"] + "\n") #5
    output_file.write(str(dict_["output_per_layer"]) + "\n") #6
    output_file.write(dict_["log_level"] + "\n") #7
    output_file.write(str(dict_["whether_run_mAP"]) + "\n") #8
    output_file.write(dict_["mAP_model_type"] + "\n") #9
    output_file.write(str(dict_["whether_run_float"]) + "\n") #10
    output_file.write(dict_["analysis_mode"] + "\n") #11
    output_file.write(str(dict_["whether_run_update"]) + "\n") #12
    output_file.write(str(dict_["concat_input_change"]) + "\n") #13
    output_file.write(str(dict_["whether_limit_shift"]) + "\n") #14
    output_file.write(str(dict_["shift_upper_limit"]) + "\n") #15
    output_file.write(str(dict_["shift_lower_limit"]) + "\n") #16
    output_file.write(",".join('%s' %id for id in dict_["img_target_size"]) + "\n") #17
    output_file.write(str(dict_["keep_aspect_ratio"]) + "\n") #18
    output_file.write(str(dict_["channel_num"]) + "\n") #19
    output_file.write(",".join('%s' %id for id in dict_["list_of_customized_name"]) + "\n") #20
    output_file.write(str(dict_["whether_run_iee"]) + "\n") #18
    output_file.write(str(dict_["iee_output_path"]) + "\n") #19
    output_file.write(str(dict_["iee_npu_name"]) + "\n") #11
    input_file.close()
    output_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="extract the result of ip evaluator"
        )

    argparser.add_argument(
        '-i',
        '--input_file',
        help="input file path of input_params.json"
        )
    
    argparser.add_argument(
        '-o',
        '--output_file',
        help="output file path of input_params.txt"
        )

    args = argparser.parse_args()

    main_(args)
