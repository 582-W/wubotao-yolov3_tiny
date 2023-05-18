import numpy as np
import math
import argparse
import os,shutil
from PIL import Image,ImageDraw
import json
import cv2
import json
def sofmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs
def load_gd(path):
    with open(path) as f:
        data = json.load(f)
    return data
def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)
def Plot_labels_cv2(img, labels,probs, save_path, color_fp=(0,255,0)):

    img = np.copy(img)
    for i in range(5):
        class_name = labels[i]
        confidence = int(probs[i]*10000)
        string = class_name + str(' ')+str(confidence*0.0001)
        # print(string)
        img = cv2.putText(img, string, (5, 30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fp, 2)
        # img = cv2.putText(img, str('0.')+str(confidence), (250, 30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fp, 2)
    cv2.imwrite(save_path, img)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--image_dir")
    parser.add_argument("--save_path")
    parser.add_argument("--input_dir", help="dir of input array file(.npy)")
    parser.add_argument("--output_dir", help="dir of postprocess output")
    parser.add_argument("--ground_truth", help="dir of postprocess output")
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    txt_folder = args.output_dir + '/result'
    os.mkdir(txt_folder)
    
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)

    label = load_labels(args.label)

    image_list = []
    out = {}
    out['top-1 accuracy'] = 0
    out['classification result'] = {}    
    for file in os.listdir(args.input_dir):
        img_name = str(file).split('img_')[1].replace('.npy', '')
        if img_name not in image_list:
            image_list.append(img_name)
    for img in image_list:
        for file in os.listdir(args.input_dir):
            if img in file:
                ai = np.load(os.path.join(args.input_dir, file))
                ai = ai.transpose(0,2,3,1).reshape(-1)
                soft = sofmax(ai)
                pred = np.argsort(soft)[::-1][:5]
                prob = soft[pred]
                clas = label[pred]
                out['classification result'][img]=np.array(pred[0]).tolist()
                f = open(os.path.join(txt_folder, img + '.txt'), mode='w')
                for i in range(5):
                    class_name = clas[i]
                    confidence = prob[i]
                    f.write((class_name + ' ' + str(confidence) + '\n'))
                f.close()
                for file1 in os.listdir(args.image_dir):
                    if img in file1:
                        img1 = cv2.imread(os.path.join(args.image_dir, file1))
                        save_path1 = os.path.join(args.save_path, img + '.jpg')
                        Plot_labels_cv2(img1, clas,prob, save_path1, color_fp=(0,255,0))

    gd = load_gd(args.ground_truth)
    # print(gd)
    img_num = len(out['classification result'])
    correct = 0
    for key in out['classification result'].keys():
        if gd[key] == out['classification result'][key] :
            correct = correct + 1
    acc = correct/img_num
    out['top-1 accuracy'] = acc
    json_str = json.dumps(out,sort_keys=False,indent=4)
    with open(os.path.join(args.output_dir,'output.json'), 'w') as json_file:
        json_file.write(json_str)
            

