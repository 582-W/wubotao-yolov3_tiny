import numpy as np
import tensorflow as tf
from keras import backend as K

import argparse
import shutil
import colorsys
import os
import sys
import time
import logging

from PIL import Image, ImageFont, ImageDraw
sys.path.append("..")
sys.path.append("../..")
from detection_result.yolo3.model import yolo_eval

# current_path = os.path.abspath(sys.argv[0])
current_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_path)
#classes_path = current_path + "/model_data/coco_classes.txt"


def yolo_output2standard(imag_path, onnx_output_path, detection_result_path, classes_file, anchors_file, options = "yolo_v3", draw_bbox = False, log_level = logging.INFO):
   
    #logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=log_level)

    if not os.path.exists(classes_file):
        logging.error("The classes_file does not exist,please input the right path!")
    
    with open(classes_file) as f:
        class_names = f.readlines()

    class_names = [c.strip() for c in class_names]

    for i in range(len(class_names)):	
        name = class_names[i]	
        if len(name.split(" ")) > 1:	
            name = "_".join(name.split(" "))	
            class_names[i] = name

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    if not os.path.exists(anchors_file):
        logging.error("The anchors_file does not exist, please input the right path!")

    with open(anchors_file) as f:
        anchors = f.readline()

    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

    img_ids = []
    for filename in os.listdir(imag_path):
        portion = os.path.split(filename)
        img_id = portion[1].split('.')[0]
        exten_name = portion[1].split('.')[-1]
        img_ids.append(img_id)
    
    filenamelist = []
    for filename in os.listdir(onnx_output_path):
        if img_ids[0] in filename:
            filenamelist.append(filename)
    
    filenamelist = sorted(filenamelist)  ####
    logging.debug("filenamelist:", filenamelist)
    
    if options == "yolo_v3": 
        logging.debug("len(filenamelist):", len(filenamelist))
        assert len(filenamelist) == 3, "Please check the num of yolov3 onnx output per image, it is not 3!"
        index = filenamelist[0].index(img_ids[0])
        pre_name1 = filenamelist[0][:index]
        pre_name2 = filenamelist[1][:index]
        pre_name3 = filenamelist[2][:index]
        post_name = filenamelist[0][index+len(img_ids[0]):]

    if options == "tiny_yolo_v3":
        logging.debug("len(filenamelist):", len(filenamelist))
        assert len(filenamelist) == 2, "Please check the yolov3_tiny onnx output per image, it is not 2!"
        index = filenamelist[0].index(img_ids[0])
        pre_name1 = filenamelist[0][:index]
        pre_name2 = filenamelist[1][:index]
        post_name = filenamelist[0][index+len(img_ids[0]):]
        
    logging.info("The img_id to detect are : {}\n total {}".format(img_ids, len(img_ids)))
    
    if os.path.exists(detection_result_path):
        shutil.rmtree(detection_result_path)
    os.makedirs(detection_result_path)

    # result如果之前存放的有文件，全部清除-c $classes_file -a $anchors_file
    for i in os.listdir(detection_result_path):
        path_file = os.path.join(detection_result_path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)

    t1 = time.time()
    for img_id in img_ids:
        time_loop_start = time.time()
        # time1 = time.time()
        if options == "yolo_v3":
            file1 = os.path.join(onnx_output_path, pre_name1 + img_id + post_name)
            file2 = os.path.join(onnx_output_path, pre_name2 + img_id + post_name)
            file3 = os.path.join(onnx_output_path, pre_name3 + img_id + post_name)

            data1 = np.load(file1)
            data2 = np.load(file2)
            data3 = np.load(file3)
            data1 = data1.transpose(0,2,3,1)
            data2 = data2.transpose(0,2,3,1)
            data3 = data3.transpose(0,2,3,1)

            data1 = K.constant(data1)
            data2 = K.constant(data2)
            data3 = K.constant(data3)
        
            yolo_model_output = [data1,data2,data3]

        if options == "tiny_yolo_v3":
            file1 = os.path.join(onnx_output_path, pre_name1 + img_id + post_name)
            file2 = os.path.join(onnx_output_path, pre_name2 + img_id + post_name)
            logging.debug("file1:", file1)
            logging.debug("file2:", file2)
            data1 = np.load(file1)
            data2 = np.load(file2)
            data1 = data1.transpose(0,2,3,1)
            data2 = data2.transpose(0,2,3,1)
            
            data1 = K.constant(data1)
            data2 = K.constant(data2)
            
            yolo_model_output = [data1,data2]

        image_file = os.path.join(imag_path, img_id + "." + exten_name)
        logging.debug("image file:", image_file)
            
        # print("img_file:", image_file)
        image = Image.open(image_file)
        input_image_shape = image.size

        input_image_shape = (input_image_shape[1], input_image_shape[0])
        score_threshold = 0.3
        iou_threshold = 0.45
        out_boxes, out_scores, out_classes = yolo_eval(yolo_model_output, anchors,
                    len(class_names), input_image_shape,
                    score_threshold = score_threshold, iou_threshold = iou_threshold)

        
        out_boxes = K.eval(out_boxes)
        out_scores = K.eval(out_scores)
        out_classes = K.eval(out_classes)

        list_file = open(detection_result_path + '/%s.txt'%(img_id), 'w')
        
        ###  draw bbox
        logging.info('Found {} boxes for {}.{}'.format(len(out_boxes), img_id, exten_name)) # 提示用于找到几个bbox
        if draw_bbox:
            font = ImageFont.truetype(font=current_path + '/font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.3).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 800

        # 保存框检测出的框的个数
        #file.write('find  '+str(len(out_boxes))+' target(s) \n')

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.5f}'.format(predicted_class, score)
            score = '{:.5f}'.format(score)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 写入检测位置            
            #file.write(predicted_class+'  score: '+str(score)+' \nlocation: top: '+str(top)+'、 bottom: '+str(bottom)+'、 left: '+str(left)+'、 right: '+str(right)+'\n')
            #file.write(predicted_class +''+ str(score) +  str(top) + str(bottom)+ str(left)+ str(right)+'\n')
            #file.write('{} {} {} {} {} {}\n'.format(predicted_class,score,left,top,right,bottom))
            
            logging.info('{} {} {} {} {}'.format(label, left, top, right, bottom))
            list_file.write('{} {} {} {} {} {}\n'.format(predicted_class,score,left,top,right,bottom))
            
            if draw_bbox:
                image = image.convert('RGB')
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline= colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill= colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        
        if draw_bbox:
            # display in Jupyter notebook
            # display(image)
            image.show()

        list_file.close()
        
        # print("detection result use time {}".format(time.time()-time5))
        logging.info("{}.{} detection result has saved to {}/{}.txt!\n".format(img_id, exten_name, detection_result_path, img_id))
        
        time_loop_end = time.time()
        logging.info("img: {}.{} detection use {}.\n".format(img_id, exten_name, time_loop_end - time_loop_start))
        K.clear_session()
        

    total_time = time.time() - t1
    logging.critical("total time is {} in yolo_output2standard.\n".format(total_time))
    logging.critical("Done!\n")
    

def main_(args):
    imag_path = args.img_path
    assert os.path.exists(imag_path), "Please enter valid img_path"

    onnx_output_path = args.onnx_output_path
    assert os.path.exists(onnx_output_path), "Please enter valid onnx_output_path"


    detection_result_path = args.detection_result_path

    if os.path.exists(detection_result_path):
        shutil.rmtree(detection_result_path)
    os.mkdir(detection_result_path)

    classes_file = args.classes_file
    assert os.path.exists(classes_file), "Please enter valid classes_file path"

    anchors_file = args.anchors_file
    assert os.path.exists(anchors_file), "Please enter valid anchors_file path"

    options = args.options

    draw_bbox = args.draw_bbox
    yolo_output2standard(imag_path, onnx_output_path, detection_result_path, classes_file, anchors_file, options, draw_bbox)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    logging.critical("Changing the YOLOv3 onnx output results to standard detection and saving to file\n")
    #img_path = "/home/data2/img2"
    #onnx_output_path = "/home/data2/YOLOv3_output"
    #detection_result_path = "/home/data2/detections"
    #yolo_output2standard(img_path, onnx_output_path, detection_result_path)

    argparser = argparse.ArgumentParser(
        description="Changing the YOLOv3 onnx output results to standard detection and save to file"
        )

    argparser.add_argument(
        '-i',
        '--img_path',
        help="path of input image file folder containing the paths of all the imgs",
        default="",
        )

    argparser.add_argument(
        '-o',
        '--onnx_output_path',
        help="path of input .npy file folder containing the YOLOv3 onnx output results",
        default="",
        )

    argparser.add_argument(
        '-d',
        '--detection_result_path',
        help="path of output .txt file folder containing the detection results",
        default="",
        )

    argparser.add_argument(
        '-c',
        '--classes_file',
        help="path of the classes file that the dataset class type",
        default="",
    )

    argparser.add_argument(
        '-a',
        '--anchors_file',
        help="path of the anchors file that the object detection network",
        default="",
    )
    

    argparser.add_argument(
        '-dr',
        '--draw_bbox',
        help="draw the bbox of the detection result",
        type=str2bool,
        default=False,
    )
    argparser.add_argument(
        '-op',
        '--options',
        help="the optinon of the network model, such as, yolo, yolo_tiny",
        default="yolo",

    )
    args = argparser.parse_args()
    main_(args)
### python yolo_output2standard.py --img_path "/data1/img.jpg" --onnx_output_path "...txt" 
# --detection_result_path "" --draw_result True
