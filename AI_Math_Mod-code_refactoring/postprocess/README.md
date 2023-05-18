# postprocess of ssd model
This is the description file of ssd postprocess. As long as you  get the output of twelve convolution layers,You can conduct the inference using this code. The details are explained below. For specific calculation process,please refer to the code.

## AnchorBoxes
Create an output tensor containing anchor box coordinates and variances based on the input tensor and the passed arguments.

Input shape:
    4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
    or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

Output shape:
    5D tensor of shape `(batch, height, width, n_boxes, 8)`. Value of n_boxes is determined by `aspect_ratios` and `two_boxes_for_ar1`,it has been kept at 4 in this case.The last axis contains the four anchor box coordinates and the four variance values for each box. 

Arguments:
    
    img_height (int): The height of the input images.
    
    img_width (int): The width of the input images.
    
    this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
        as a fraction of the shorter side of the input image.
    
    next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
        `self.two_boxes_for_ar1 == True`.
    
    aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
        generated for this layer.
    
    two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
        If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
        using the scaling factor for the respective layer, the second one will be generated using
        geometric mean of said scaling factor and next bigger scaling factor.
    
    clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within image boundaries.
    
    variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate will be divided by
        its respective variance value.
    
    coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
        of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
        'corners' for the format `(xmin, ymin, xmax,  ymax)`, or 'minmax' for the format `(xmin, xmax, ymin, ymax)`.
    
    normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
        i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.


## Reshape
Change shape of input tensor.

1.Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`

   We want the classes isolated in the last axis to perform softmax on them.

2.Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`

   We want the four box coordinates isolated in the last axis to compute the smooth L1 loss.

3.Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`


## Concatenate
Concatenate the predictions from the different layers.

Because of the reshape operation above,Axis 0 (batch) and axis 2 (n_classes, 4 or 8, respectively) are identical for all layer predictions,so we want to concatenate along axis 1, the number of boxes per layer.

1.Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)

2.Output shape of `mbox_loc`: (batch, n_boxes_total, 4)

3.Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)

## Activation
For the class confidence predictions, we'll apply an activation layer using `softmax` as activation function.


## Concatenate
Concatenate the class and box predictions and the anchors to one large predictions vector.

Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)


## DecodeDetections
Convert model prediction output back to a format that contains only the positive box predictions.

Arguments:
    
    y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
        of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
        boxes predicted by the model per image and the last axis contains
        `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
    
    confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
        positive class in order to be considered for the non-maximum suppression stage for the respective class.
        A lower value will result in a larger part of the selection process being done by the non-maximum suppression
        stage, while a larger value will result in a larger part of the selection process happening in the confidence
        thresholding stage.
    
    iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
        with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
        to the box score.
    
    top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
        non-maximum suppression stage.
    
    input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
        for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
        `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
    
    normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
        and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
        relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
        Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
        coordinates. Requires `img_height` and `img_width` if set to `True`.
    
    img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
    
    img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
        Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
        to the boxes. If 'exclude', the border pixels do not belong to the boxes.
        If 'half', then one of each of the two horizontal and vertical borders belong
        to the boxex, but not the other.
    

## Difference between the portprocess of keras and caffe ssd model 

Generally,there is almost no difference between the postprocess of keras model and caffe model. In anchorbox layer(called "priors" in Caffe implementation),they just use different names or formats to represent mathematical variables with the same meaning.

1.The value of aspect_ratios in keras model is `[2,3,1,0.5,1/3]` and in caffe model it is`[2,3,1]`.However caffe model use an operation called `flip` to add reciprocal to the list.

2.`min_size` and `max_size` in caffe model correspond to `this_scale*min(image_height,image_width)` and `next_scale*min(image_height,image_width)` in keras model
