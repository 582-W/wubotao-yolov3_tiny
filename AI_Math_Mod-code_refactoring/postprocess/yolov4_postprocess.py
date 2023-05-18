import numpy as np
import math
import argparse
import os,shutil
from PIL import Image,ImageDraw

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_region_boxes(boxes_and_confs):
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = np.concatenate(boxes_list, axis=1)
    confs = np.concatenate(confs_list, axis=1)


    return [boxes, confs]

def yolo_forward_dynamic(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    output_np=output
    bxy_list_np = []
    bwh_list_np = []
    det_confs_list_np = []
    cls_confs_list_np = []
    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list_np.append(output_np[:, begin: begin + 2])
        bwh_list_np.append(output_np[:, begin + 2: begin + 4])
        det_confs_list_np.append(output_np[:, begin + 4: begin + 5])
        cls_confs_list_np.append(output_np[:, begin + 5: end])

    bxy_np = np.concatenate(bxy_list_np, axis=1)

    bwh_np = np.concatenate(bwh_list_np, axis=1)
    # Shape: [batch, num_anchors, H, W]
    det_confs_np = np.concatenate(det_confs_list_np, axis=1)


    det_confs_np = det_confs_np.reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3])


    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs_np = np.concatenate(cls_confs_list_np, axis=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs_np = cls_confs_np.reshape(output_np.shape[0], num_anchors, num_classes, output_np.shape[2] * output_np.shape[3])

    cls_confs_np = cls_confs_np.transpose(0, 1, 3, 2).reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3],
                                                      num_classes)


    bxy_np = sigmoid(bxy_np) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh_np = np.exp(bwh_np)
    det_confs_np = sigmoid(det_confs_np)
    cls_confs_np = sigmoid(cls_confs_np)

    grid_x = np.expand_dims(np.expand_dims(
        np.expand_dims(np.linspace(0, output_np.shape[3] - 1, output_np.shape[3]), axis=0).repeat(output_np.shape[2], 0), axis=0),
                            axis=0)
    grid_y = np.expand_dims(np.expand_dims(
        np.expand_dims(np.linspace(0, output_np.shape[2] - 1, output_np.shape[2]), axis=1).repeat(output_np.shape[3], 1), axis=0),
                            axis=0)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []


    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx_np = bxy_np[:, ii: ii + 1] + grid_x  # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by_np = bxy_np[:, ii + 1: ii + 2] + grid_y  # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw_np = bwh_np[:, ii: ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh_np = bwh_np[:, ii + 1: ii + 2] * anchor_h[i]

        bx_list.append(bx_np)
        by_list.append(by_np)
        bw_list.append(bw_np)
        bh_list.append(bh_np)

        ########################################
        #   Figure out bboxes from slices     #
        ########################################

        # Shape: [batch, num_anchors, H, W]
    bx_np = np.concatenate(bx_list, axis=1)
    # Shape: [batch, num_anchors, H, W]
    by_np = np.concatenate(by_list, axis=1)
    # Shape: [batch, num_anchors, H, W]
    bw_np = np.concatenate(bw_list, axis=1)
    # Shape: [batch, num_anchors, H, W]
    bh_np = np.concatenate(bh_list, axis=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw_np = np.concatenate((bx_np, bw_np), axis=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh_np = np.concatenate((by_np, bh_np), axis=1)

    # normalize coordinates to [0, 1]
    bx_bw_np /= output_np.shape[3]
    by_bh_np /= output_np.shape[2]



    # Shape: [batch, num_anchors * H * W, 1]
    bx_np = bx_bw_np[:, :num_anchors].reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3], 1)
    by_np = by_bh_np[:, :num_anchors].reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3], 1)
    bw_np = bx_bw_np[:, num_anchors:].reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3], 1)
    bh_np = by_bh_np[:, num_anchors:].reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3], 1)

    bx1_np = bx_np - bw_np * 0.5
    by1_np = by_np - bh_np * 0.5
    bx2_np = bx1_np + bw_np
    by2_np = by1_np + bh_np

    boxes_np = np.concatenate((bx1_np, by1_np, bx2_np, by2_np), axis=2).reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3],
                                                        1, 4)


    det_confs_np = det_confs_np.reshape(output_np.shape[0], num_anchors * output_np.shape[2] * output_np.shape[3], 1)
    confs_np = cls_confs_np * det_confs_np

    return boxes_np, confs_np



class YoloLayer( ):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward_dynamic(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)



class Yolov4_postprocess():
    def __init__(self, n_classes):
        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)
    def forward(self, x2, x10, x18):

        y1 = self.yolo1.forward(x2)
        y2 = self.yolo2.forward(x10)
        y3 = self.yolo3.forward(x18)
        y = get_region_boxes([y1, y2, y3])
        return y


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)
def post_processing(conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]



    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)


    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)



    return bboxes_batch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", help="height of model input ", type=int, default=300)
    parser.add_argument("--img_width", help="width of model input ", type=int, default=300)
    parser.add_argument("--image_dir", help="dir of test image")
    parser.add_argument("--input_dir", help="dir of input array file(.npy)")
    parser.add_argument("--output_dir", help="dir of postprocess output")
    parser.add_argument("--class_dir", help="dir of txt file about object class")
    args = parser.parse_args()

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
                if temp.shape == (1, 255, 52, 52):
                    x2 = temp
                elif temp.shape == (1, 255, 26, 26):
                    x10 = temp
                elif temp.shape == (1, 255, 13, 13):
                    x18 = temp
        output = Yolov4_postprocess(n_classes=80).forward(x2, x10, x18)
        boxes = post_processing(0.4, 0.6, output)

        for file in os.listdir(args.image_dir):
            if img in file:
                image = Image.open(os.path.join(args.image_dir, file))


        f = open(os.path.join(args.output_dir, img + '.txt'), mode='w')
        for y_pred in boxes[0]:
            class_id = int(y_pred[6])
            confidence = y_pred[5]
            class_name = classes[class_id]
            xmin = max(int(y_pred[0] * image.width ), 0)
            ymin = max(int(y_pred[1] * image.height), 0)
            xmax = min(int(y_pred[2] * image.width ), image.width - 1)
            ymax = min(int(y_pred[3] * image.height ), image.height - 1)
            f.write((class_name + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(
                xmax) + ' ' + str(ymax) + '\n'))
        f.close()