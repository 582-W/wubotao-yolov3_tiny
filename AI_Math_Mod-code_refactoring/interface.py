from inference.fp_inferencer import FixedPointInferencer
from inference.fl_inferencer import FloatInferencer
from model_update.model_update import ModelUpdater
from ModelEvaluator.core import evaluate
#from preprocessing.img_to_txt import ImgPreprocessor
from utils.utils import str2bool, check_folder, check_path
from weight_analysis.weight_analyser import WeightAnalyser
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

    if args.mode == "weight_analysis":
        onnx_file_path = args.input_onnx_file
        check_path(onnx_file_path)

        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)
        
        weight_output_folder_path = os.path.join(input_case_folder, 'weight_analysis_output')
        check_folder(weight_output_folder_path)
        
        _, weight_analysis_json_path = WeightAnalyser(onnx_file_path, weight_output_folder_path, analysis_mode = args.analysis_mode).analyse_weight()
        #_, weight_analysis_json_path = WeightAnalyser(onnx_file_path, weight_output_folder_path, analysis_mode = 'per_layer', conv_percent = 1.0, bn_percent = 1.0, log_level=log_level, log_file = log_file)
     
    elif args.mode == "datapath_analysis":
        onnx_file_path = args.input_onnx_file
        check_path(onnx_file_path)
        
        concat_change = args.concat_input_change

        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder, whether_delete=False)

        datapath_output_folder_path = os.path.join(input_case_folder, 'datapath_analysis_output')
        check_folder(datapath_output_folder_path)

        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        print("channel_num: ", channel_num)
        keep_aspect_ratio = args.keep_aspect_ratio
        list_of_customized_name = args.list_of_customized_name
        print("list_of_customized_name: ", list_of_customized_name)
        logging.critical("img preprocess method: {}".format(method))
        #ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        datapath_analyser = FloatInferencer(onnx_file_path, image_npy_folder, input_case_folder, list_of_customized_name=list_of_customized_name, whether_analyse_datapath=True)
        _, datapath_analysis_json_path = datapath_analyser.float_inference(datapath_output_folder_path,img_preprocess_method=method,concat_change=concat_change, log_level=log_level, log_file = log_file)

    elif args.mode == "model_update":
        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        updated_onnx_path = args.updated_onnx_path

        analysis_mode = args.analysis_mode
        
        run_update = args.whether_run_update

        concat_change = args.concat_input_change

        limit_shift = args.whether_limit_shift
        shift_upper_limit=args.shift_upper_limit
        shift_lower_limit=args.shift_lower_limit

        config_file_path = args.hardware_config_file
        check_path(config_file_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder, whether_delete=False)
        
        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)
        
        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        keep_aspect_ratio = args.keep_aspect_ratio
        list_of_customized_name = args.list_of_customized_name
        logging.critical("img preprocess method: {}".format(method))
        #ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)

        onnx_file_path = args.input_onnx_file
        check_path(onnx_file_path)

        model_update_output_folder_path = os.path.join(input_case_folder, 'model_update_output')
        check_folder(model_update_output_folder_path)

        weight_analysis_json_path = args.weight_analysis_json
        datapath_analysis_json_path = args.datapath_analysis_json
        print("list_of_customized_name: ", list_of_customized_name)
        modelupdater = ModelUpdater(onnx_file_path,input_case_folder,image_npy_folder,weight_analysis_json_path,datapath_analysis_json_path,config_file_path, \
                                    model_update_output_folder_path,analysis_mode,run_update,method,concat_change, list_of_customized_name)
        modelupdater.model_update(limit_shift, shift_upper_limit, shift_lower_limit)
        
    elif args.mode == "sw_model_update":
        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        updated_onnx_path = args.updated_onnx_path

        onnx_file_path = args.input_onnx_file
        check_path(onnx_file_path)
        
        config_file_path = args.hardware_config_file
        check_path(config_file_path)

        model_update_output_folder_path = os.path.join(input_case_folder, 'model_update_output')
        check_folder(model_update_output_folder_path)

        datapath_analysis_json_path = args.datapath_analysis_json
        ModelUpdater_scaled_weight(onnx_file_path, datapath_analysis_json_path, model_update_output_folder_path,updated_onnx_path, config_file_path, group_size = 1)

    elif args.mode == "create_json":
        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        updated_json_path = args.updated_json_path

        model_update_output_folder_path = os.path.join(input_case_folder, 'model_update_output')
        check_folder(model_update_output_folder_path, whether_delete=False)

        weight_analysis_json_path = args.weight_analysis_json
        datapath_analysis_json_path = args.datapath_analysis_json
        ModelUpdater_create_json(model_update_output_folder_path, weight_analysis_json_path, datapath_analysis_json_path, updated_json_path)

    elif args.mode == "float_inference":
        input_onnx_file = args.input_onnx_file
        check_path(input_onnx_file)

        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        fl_output_folder_path = os.path.join(input_case_folder, 'float_inference_output')
        check_folder(fl_output_folder_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder, whether_delete=False)

        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)

        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        keep_aspect_ratio = args.keep_aspect_ratio
        list_of_customized_name = args.list_of_customized_name
        logging.critical("img preprocess method: {}".format(method))
        #ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        flinferencer = FloatInferencer(input_onnx_file, image_npy_folder, input_case_folder, list_of_customized_name=list_of_customized_name, whether_analyse_datapath=False)
        flinferencer.float_inference(fl_output_folder_path, output_per_layer = args.output_per_layer,log_level = log_level, log_file = log_file)

    elif args.mode == "fp_inference":
        updated_onnx_path = args.updated_onnx_path
        check_path(updated_onnx_path)

        updated_json_path = args.updated_json_path
        check_path(updated_json_path)

        config_file_path = args.hardware_config_file
        check_path(config_file_path)

        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        fp_output_folder_path = os.path.join(input_case_folder, 'fp_inference_output')
        check_folder(fp_output_folder_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder, whether_delete=False)

        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)

        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        keep_aspect_ratio = args.keep_aspect_ratio
        list_of_customized_name = args.list_of_customized_name
        logging.critical("img preprocess method: {}".format(method))
        # img_preprocess(input_image_folder_path, image_npy_folder, method, target_size)
        #ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        my_dfp_inference = FixedPointInferencer(updated_onnx_path, input_image_folder_path, input_case_folder, list_of_customized_name, config_file_path)
        print("list_of_customized_name: ", list_of_customized_name)
        my_dfp_inference.fix_point_inference(image_npy_folder, fp_output_folder_path, updated_json_path, 
          log_level = log_level, analysis_mode = args.analysis_mode, 
          output_per_layer = args.output_per_layer, is_hardware = False, accelerate_option="img2col", 
          log_file = log_file, img_preprocess_method=method)

        # def fix_point_inference(self, output_folder_path, update_json_file_path, 
        #   log_level = logging.INFO, analysis_mode = 'per_layer', 
        #   output_per_layer = False, is_hardware = False, accelerate_option="img2col", 
        #   log_file="", img_preprocess_method="yolo"):

        # dfp_inference(updated_onnx_path, image_npy_folder, fp_output_folder_path, config_file_path, updated_json_path, log_level = log_level, analysis_mode = args.analysis_mode, output_per_layer = args.output_per_layer, is_hardware = False, log_file = log_file, img_preprocess_method=method)

        # ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        # dfp_inference(updated_onnx_path, image_npy_folder, fp_output_folder_path, config_file_path, updated_json_path, log_level = log_level, analysis_mode = args.analysis_mode, output_per_layer = args.output_per_layer, is_hardware = False, log_file = log_file, img_preprocess_method=method)


    elif args.mode == "sw_fp_inference":
        updated_onnx_path = args.updated_onnx_path
        check_path(updated_onnx_path)

        updated_json_path = args.updated_json_path
        check_path(updated_json_path)

        config_file_path = args.hardware_config_file
        check_path(config_file_path)

        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        fp_output_folder_path = os.path.join(input_case_folder, 'fp_inference_output')
        check_folder(fp_output_folder_path)

        image_npy_folder = os.path.join(input_case_folder, 'img_npy')
        check_folder(image_npy_folder, whether_delete=False)

        input_image_folder_path = args.input_img_folder
        check_path(input_image_folder_path)

        method = args.img_preprocess_method
        target_size = args.img_target_size
        channel_num = args.channel_num
        keep_aspect_ratio = args.keep_aspect_ratio
        logging.critical("img preprocess method: {}".format(method))
        #ImgPreprocessor().preprocess(input_image_folder_path, image_npy_folder, method, target_size, channel_num = channel_num, output_format = "npy", keep_aspect_ratio = keep_aspect_ratio)
        dfp_inference(updated_onnx_path, image_npy_folder, fp_output_folder_path, config_file_path, updated_json_path, log_level = log_level, analysis_mode = 'per_channel', output_per_layer = args.output_per_layer, is_hardware = False, log_file = log_file, img_preprocess_method=method)
    

    elif args.mode == "model_evaluate":
        input_case_folder = args.input_case_folder
        check_folder(input_case_folder, whether_delete=False)

        evaluator_output_folder_path = os.path.join(input_case_folder, 'evaluator_output')
        check_folder(evaluator_output_folder_path)

        fl_output_folder_path = args.fl_output_folder
        check_path(evaluator_output_folder_path)

        fp_output_folder_path = args.fp_output_folder
        check_path(fp_output_folder_path)

        updated_onnx_path = args.updated_onnx_path
        check_path(updated_onnx_path)

        if not fp_output_folder_path.endswith("float_res"):
            fp_output_folder_path = os.path.join(fp_output_folder_path, "float_res")

        evaluate(fl_output_folder_path, fp_output_folder_path, evaluator_output_folder_path, onnx_file_path = updated_onnx_path, log_level = log_level, log_file = log_file)


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
