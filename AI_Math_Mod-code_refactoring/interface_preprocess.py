from preprocessing.img_to_txt import ImgPreprocessor
from utils.utils import str2bool, check_folder, check_path
import logging, os, shutil, argparse, sys, time, colorsys

def main_(args):
    log_level = logging.INFO
    if args.log_level == "debug":
        log_level = logging.DEBUG
    elif args.log_level == "info":
        log_level = logging.INFO
    elif args.log_level == "warning":
        log_level = logging.WARNING
    elif args.log_level == "error":
        log_level = logging.ERROR
    else:
        log_level = logging.CRITICAL
    
    logging.basicConfig(format='%(message)s', level=log_level)

    log_file = args.log_file

    if args.mode == "img_preprocess":
        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder)

        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        print("channel_num: ", channel_num)
        keep_aspect_ratio = args.keep_aspect_ratio
        logging.critical("img preprocess method: {}".format(method))
        ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        data_max = ImgPreprocessor().get_datapath_input(method)
        output_file_path = os.path.join(input_case_folder, 'data_max.txt')
        output_file = open(output_file_path, 'w+')
        output_file.write(str(data_max) + "\n")
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="This is the interface of AI_Math_Mod proj\n \
            1. [mode] weight_analysis:  \npython3 interface.py -m weight_analysis   -ic input_case_folder -io input_onnx_file -l log_level\n\n\
            2. [mode] datapath_analysis:\npython3 interface.py -m datapath_analysis -ic input_case_folder -io input_onnx_file -ii input_img_folder -m2 img_preprocess_method -l log_level\n\n\
            3. [mode] model_update:     \npython3 interface.py -m model_update      -ic input_case_folder -io input_onnx_file -wj weight_analysis_json -dj datapath_analysis_json -l log_level -uo updated_onnx -uj updated_json -c $hardware_config_file -ii $input_img_folder\n\n\
            4. [mode] float_inference:  \npython3 interface.py -m float_inference   -ic input_case_folder -io input_onnx_file -ii input_img_folder -m2 img_preprocess_method -l log_level -pl output_per_layer\n\n\
            5. [mode] fp_inference:     \npython3 interface.py -m fp_inference      -ic input_case_folder -uo updated_onnx -uj updated_json -c hardware_config_file -ii input_img_folder -m2 img_preprocess_method -l log_level -pl output_per_layer\n\n\
            6. [mode] model_evaluate:   \npython3 interface.py -m model_evaluate    -ic input_case_folder -ofl fl_output_folder -ofp fp_output_folder -l log_level\n\n\
            ",
        formatter_class=argparse.RawTextHelpFormatter
        )


    argparser.add_argument(
        '-ic',
        '--input_case_folder',
        help="path of input_case_folder, will save all the useful result during the whole process",
        default="",
        )

    argparser.add_argument(
        '-io',
        '--input_onnx_file',
        help="path of input_onnx_file, the onnx file to be analysed",
        default="",
        )

    argparser.add_argument(
        '-ii',
        '--input_img_folder',
        help="path of input_img_folder, the imgs to be analysed and infereced",
        default="",
        )

    argparser.add_argument(
        '-c',
        '--hardware_config_file',
        help="path of hardware_config_file,  e.g. bitwidth constraint",
        default="./config/hardware.json",
    )

    argparser.add_argument(
        '-wj',
        '--weight_analysis_json',
        help="path of weight_analysis_json",
    )

    argparser.add_argument(
        '-dj',
        '--datapath_analysis_json',
        help="path of datapath_analysis_json",
    )

    argparser.add_argument(
        '-uo',
        '--updated_onnx_path',
        help="path of updated_onnx_path",
    )

    argparser.add_argument(
        '-uj',
        '--updated_json_path',
        help="path of updated_json_path",
    )

    argparser.add_argument(
        '-pl',
        '--output_per_layer',
        type=str2bool,
        help="whether output per layer when inference",
    )

    argparser.add_argument(
        '-ofl',
        '--fl_output_folder',
        help="path of float inference output",
    )

    argparser.add_argument(
        '-ofp',
        '--fp_output_folder',
        help="path of fp inference output",
    )

    argparser.add_argument(
        '-m2',
        '--img_preprocess_method',
        help="img_preprocess_method",
        default="yolo",
    )

    argparser.add_argument(
        '-m',
        '--mode',
        help="mode, options: weight_analysis, datapath_analysis, model_update, float_inference, fp_inference, model_evaluate",
        default="weight_analysis",
    )

    argparser.add_argument(
        '-l',
        '--log_level',
        help="log_level, options: info, warning, error, critical",
        default="critical",
    )

    argparser.add_argument(
        '-lf',
        '--log_file',
        help="the path of logging file",
        default="",
    )

    argparser.add_argument(	
        '-am',	
        '--analysis_mode',	
        help="the mode of model analysis",	
        default="per_layer",	
    )
    
    argparser.add_argument(	
        '-ru',	
        '--whether_run_update',	
        help="open or close for model update mudule",
        type=str2bool,
        default=True,	
    )
    
    argparser.add_argument(	
        '-ca',	
        '--concat_input_change',	
        help="option to decide whether to change radix of input layers for concat layer",
        type=str2bool,
        default=True,	
    )

    argparser.add_argument(	
        '-ls',	
        '--whether_limit_shift',	
        help="option to decide whether to limit datapath shift < 35",
        type=str2bool,
        default=True,	
    )

    argparser.add_argument(	
        '-sul',	
        '--shift_upper_limit',	
        help="option to decide datapath shift upper limit",
        default=35,	
    )

    argparser.add_argument(	
        '-sll',	
        '--shift_lower_limit',	
        help="option to decide datapath shift lower limit",
        default=-4,	
    )
    
    argparser.add_argument(	
        '-its',	
        '--img_target_size',	
        help="parameters to decide input size of image",
        default=[416,416],	
    )
    
    argparser.add_argument(
        '-kar',
        '--keep_aspect_ratio',
        help="image resize method, usually True for yolo and False for others",
        type=str2bool,
        default=True,
    )
        
    argparser.add_argument(
        '-cn',
        '--channel_num',
        help="format of image channels",
        default="RGB",
    )

    argparser.add_argument(
        '-lcn',
        '--list_of_customized_name',
        help="list of customized name",
        default=[],
    )

    args = argparser.parse_args()
    main_(args)
