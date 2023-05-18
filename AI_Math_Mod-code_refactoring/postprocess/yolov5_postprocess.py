import numpy as np
import math
import argparse
import os,shutil
from PIL import Image,ImageDraw
def py_cpu_nms(dets, thresh):
    """
    nms
    :param dets: ndarray [x1,y1,x2,y2,score]
    :param thresh: int
    :return: list[index]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    order = dets[:, 4].argsort()[::-1]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]  # 不包括第0个
    return keep


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=1000):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            assert 1 == 0 , "only support single lable"
        else:  # best class only
            # conf, j = x[:, 5:].max(1, keepdim=True)
            conf = np.expand_dims(np.max(x[:, 5:], axis=1), axis=0).T
            j = np.expand_dims(np.argmax(x[:, 5:], axis=1), axis=0).T.astype(np.float32)
            # print(conf,j)
            x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]


        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        scores = np.expand_dims(scores, axis=0).T
        dets=np.concatenate((boxes, scores), axis=1)
        i=py_cpu_nms(dets, iou_thres)
        if np.array(i).shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class yolov5_postprocess():
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80):  # detection layer
        super().__init__()
        self.training=False
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = 2  # number of detection layers
        self.na = 3  # number of anchors
        self.grid = [np.array(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [np.array(0) for _ in range(self.nl)] # init anchor grid
        self.inplace = True # use in-place ops (e.g. slice assignment)
        self.stride=np.array([16.0, 32.0])
        self.anchors=np.array([[[ 0.62500,  0.87500],
         [ 1.43750,  1.68750],
         [ 2.31250,  3.62500]],

        [[ 2.53125,  2.56250],
         [ 4.21875,  5.28125],
         [ 10.75000,  9.96875]]])

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid_1, self.anchor_grid_1 = self._make_grid(nx, ny, i)

                y = sigmoid(x[i])
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid_1) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid_1 # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid_1) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid_1  # wh
                    y = np.concatenate((xy, wh, y[..., 4:]), axis=-1)
                z.append(y.reshape(bs, -1, self.no))

        return x if self.training else (np.concatenate(z, axis=1), x)

    def _make_grid(self, nx=20, ny=20, i=0):

        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid = np.expand_dims(np.expand_dims(np.stack((xv, yv), axis=2),axis=0).repeat(self.na,axis=0),axis=0).astype(np.float32)
        anchor_grid = (np.array(self.anchors[i]) * self.stride[i]).reshape((1, self.na, 1, 1, 2)).repeat(ny,axis=2).repeat(nx,axis=3).astype(np.float32)

        return grid, anchor_grid

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)2
      # np.array (faster grouped)
    boxes=np.array(boxes)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", help="height of model input", type=int, default=300)
    parser.add_argument("--img_width", help="width of model input ", type=int, default=300)
    parser.add_argument("--image_dir", help="dir of test image")
    parser.add_argument("--input_dir", help="dir of input array file(.npy)")
    parser.add_argument("--output_dir", help="dir of postprocess output")
    parser.add_argument("--class_dir", help="dir of txt file about object class")
    parser.add_argument("--input_case_folder", help="dir of input_case_folder")
    args = parser.parse_args()

    image_npy_dir=os.path.join(args.input_case_folder,"img_npy")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)
    img_height = args.img_height
    img_width = args.img_width

    # obtain object's classes from text file
    classes = []
    f = open(args.class_dir)
    for line in f:
        line = line.replace('\n', '')
        classes.append(line)
    f.close()

    n_classes = len(classes)

    image_list = []
    for file in os.listdir(args.input_dir):
        img_name = str(file).split('img_')[1].replace('.npy', '')
        if img_name not in image_list:
            image_list.append(img_name)

    for img in image_list:
        for file in os.listdir(args.input_dir):
            if img in file:
                temp = np.load(os.path.join(args.input_dir, file))
                if temp.shape[2] == 40 or temp.shape[3] == 40:
                    input1 = temp
                elif temp.shape[2] == 20 or temp.shape[3] == 20:
                    input2 = temp
                if temp.shape[2] == 26 or temp.shape[3] == 26:
                    input1 = temp
                elif temp.shape[2] == 13 or temp.shape[3] == 13:
                    input2 = temp
        re = yolov5_postprocess().forward([input1,input2])
        pred = non_max_suppression(re[0], conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        for file in os.listdir(args.image_dir):
            if img in file:
                image = Image.open(os.path.join(args.image_dir, file))
                img_array = np.array(image)
        
        for file in os.listdir(image_npy_dir):
            if img in file:
                image_npy = np.load(os.path.join(image_npy_dir, file))

        pred[0][:, :4] = np.round(scale_coords(image_npy.shape[2:], pred[0][:, :4], img_array.shape))
        
        f = open(os.path.join(args.output_dir, img + '.txt'), mode='w')
        for y_pred in pred[0]:
            class_id = int(y_pred[5])
            confidence = y_pred[4]
            class_name = classes[class_id]
            xmin = max(int(y_pred[0] ), 0)
            ymin = max(int(y_pred[1] ), 0)
            xmax = min(int(y_pred[2] ), image.width - 1)
            ymax = min(int(y_pred[3] ), image.height - 1)
            f.write((class_name + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(
                xmax) + ' ' + str(ymax) + '\n'))
        f.close()




