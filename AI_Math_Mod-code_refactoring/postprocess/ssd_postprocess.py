import numpy as np
import math
import argparse
import os,shutil
from PIL import Image,ImageDraw


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] + d # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1

def anchorbox(input_tensor,
                this_scale,
                next_scale,
                aspect_ratios,
                this_steps,
                this_offsets,
                img_height=300,
                img_width=300,
                two_boxes_for_ar1=True,
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                coords='centroids',
                normalize_coords=True):
        
    if (1 in aspect_ratios) and two_boxes_for_ar1:
        n_boxes = len(aspect_ratios) + 1
    else:
        n_boxes = len(aspect_ratios)
    # Compute box width and height for each aspect ratio
    # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
    size = min(img_height, img_width)
        # Compute the box widths and and heights for all aspect ratios
    wh_list = []
    for ar in aspect_ratios:
        if (ar == 1):
            # Compute the regular anchor box for aspect ratio 1.
            box_height = box_width = this_scale * size
            wh_list.append((box_width, box_height))
            if two_boxes_for_ar1:
                # Compute one slightly larger version using the geometric mean of this scale value and the next.
                box_height = box_width = np.sqrt(this_scale * next_scale) * size
                wh_list.append((box_width, box_height))
        else:
            box_height = this_scale * size / np.sqrt(ar)
            box_width = this_scale * size * np.sqrt(ar)
            wh_list.append((box_width, box_height))
    wh_list = np.array(wh_list)


    #batch_size, feature_map_height, feature_map_width, feature_map_channels = input_tensor._keras_shape
    batch_size, feature_map_height, feature_map_width, feature_map_channels = input_tensor.shape


    # Compute the grid of box center points. They are identical for all aspect ratios.

    # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
    if (this_steps is None):
        step_height = img_height / feature_map_height
        step_width = img_width / feature_map_width
    else:
        if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
            step_height = this_steps[0]
            step_width = this_steps[1]
        elif isinstance(this_steps, (int, float)):
            step_height = this_steps
            step_width = this_steps
    # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
    if (this_offsets is None):
        offset_height = 0.5
        offset_width = 0.5
    else:
        if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
            offset_height = this_offsets[0]
            offset_width = this_offsets[1]
        elif isinstance(this_offsets, (int, float)):
            offset_height = this_offsets
            offset_width = this_offsets
    # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
    cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
    # print(cy)
    # print('########################################')
    cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
    # print(cx)
    # print('########################################')
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    # print(cx_grid)
    # print('########################################')
    # print(cy_grid)
    # print('########################################')
    cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
    cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down
    # print(cx_grid)
    # print('########################################')
    # print(cy_grid)
    # print('########################################')

    # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
    # where the last dimension will contain `(cx, cy, w, h)`
    boxes_tensor = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))

    boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
    boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
    boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
    boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h
    # print(boxes_tensor)
    # print('########################################')

    # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
    boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')
    # print(boxes_tensor)
    # print('########################################')

    # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
    if clip_boxes:
        x_coords = boxes_tensor[:,:,:,[0, 2]]
        x_coords[x_coords >= img_width] = img_width - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:,:,:,[0, 2]] = x_coords
        y_coords = boxes_tensor[:,:,:,[1, 3]]
        y_coords[y_coords >= img_height] = img_height - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:,:,:,[1, 3]] = y_coords

    # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
    if normalize_coords:
        boxes_tensor[:, :, :, [0, 2]] /= img_width
        boxes_tensor[:, :, :, [1, 3]] /= img_height

    # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
    if coords == 'centroids':
        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
    elif coords == 'minmax':
        # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')
    # print(boxes_tensor)

    # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
    # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
    variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
    variances_tensor += variances # Long live broadcasting
    # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
    boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

    # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
    # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
    boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
    #boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(input_tensor)[0], 1, 1, 1, 1))

    return boxes_tensor

def compute_output_shape(input_arr,target_shape):
    input_shape = input_arr.shape
    if None in input_shape[1:]:
        # input shape (partially) unknown? replace -1's with None's
        return ((input_shape[0],) +
                tuple(s if s != -1 else None for s in target_shape))
    else:
        # input shape known? then we can compute the output shape
        target_shape = list(target_shape)
        msg = 'total size of new array must be unchanged'
        known, unknown = 1, None
        for index, dim in enumerate(target_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape[1:], dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            target_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        
        return (input_shape[0],) +tuple(target_shape)

def softmax(input_tensor):
    output_tensor = input_tensor.copy()
    shape = output_tensor.shape
    for k in range(shape[0]):
        for i in range(shape[1]):
            denominator = 0
            for j in range(shape[2]):
                denominator += math.exp(output_tensor[k][i][j])
            for j in range(shape[2]):
                output_tensor[k][i][j] = math.exp(output_tensor[k][i][j])/denominator
    return output_tensor

def decode_detections(y_pred,
                      confidence_thresh,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

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

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))
    '''
    cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
    cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
    w = np.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6] # w = exp(w_pred * variance_w) * w_anchor
    h = np.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5] # h = exp(h_pred * variance_h) * h_anchor

    # Convert 'centroids' to 'corners'.
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h

    # If the model predicts box coordinates relative to the image dimensions and they are supposed
    # to be converted back to absolute coordinates, do that.
    def normalized_coords():
        xmin1 = np.expand_dims(xmin * img_width, axis=-1)
        ymin1 = np.expand_dims(ymin * img_height, axis=-1)
        xmax1 = np.expand_dims(xmax * img_width, axis=-1)
        ymax1 = np.expand_dims(ymax * img_height, axis=-1)
        return xmin1, ymin1, xmax1, ymax1
    def non_normalized_coords():
        return np.expand_dims(xmin, axis=-1), np.expand_dims(ymin, axis=-1), np.expand_dims(xmax, axis=-1), np.expand_dims(ymax, axis=-1)

    if normalize_coords:
        xmin = np.expand_dims(xmin * img_width, axis=-1)
        ymin = np.expand_dims(ymin * img_height, axis=-1)
        xmax = np.expand_dims(xmax * img_width, axis=-1)
        ymax = np.expand_dims(ymax * img_height, axis=-1)
    else:
        xmin, ymin, xmax, ymax = non_normalized_coords

    # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
    y_pred = np.concatenate(([y_pred[...,:-12], xmin, ymin, xmax, ymax]), axis=-1)
    y_pred_decoded_raw = y_pred
    '''
    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates
    
    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded

def _greedy_nms(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)           

def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)`, 'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format
            `(xmin, ymin, xmax, ymax)`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'. In 'outer_product' mode, returns an
            `(m,n)` matrix with the IoU overlaps for all possible combinations of the `m` boxes in `boxes1` with the
            `n` boxes in `boxes2`. In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2`
            must be boadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`. 0 means there is no overlap between two given
        boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("All boxes must consist of 4 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(boxes1.shape[1], boxes2.shape[1]))
    if not mode in {'outer_product', 'element-wise'}: raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # Compute the IoU.

    # Compute the interesection areas.

    intersection_areas = intersection_area_(boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims((boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d), axis=1), reps=(1,n))
        boxes2_areas = np.tile(np.expand_dims((boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d), axis=0), reps=(m,1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:,xmax] - boxes1[:,xmin] + d) * (boxes1[:,ymax] - boxes1[:,ymin] + d)
        boxes2_areas = (boxes2[:,xmax] - boxes2[:,xmin] + d) * (boxes2[:,ymax] - boxes2[:,ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas

def intersection_area_(boxes1, boxes2, coords='corners', mode='outer_product', border_pixels='half'):
    '''
    The same as 'intersection_area()' but for internal use, i.e. without all the safety checks.
    '''

    m = boxes1.shape[0] # The number of boxes in `boxes1`
    n = boxes2.shape[0] # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:,[xmin,ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmin,ymin]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:,[xmax,ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:,[xmax,ymax]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,:,0] * side_lengths[:,:,1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:,[xmin,ymin]], boxes2[:,[xmin,ymin]])
        max_xy = np.minimum(boxes1[:,[xmax,ymax]], boxes2[:,[xmax,ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:,0] * side_lengths[:,1]


def postprocess(conv4_3_norm_mbox_conf,
                fc7_mbox_conf,
                conv6_2_mbox_conf,
                conv7_2_mbox_conf,
                conv8_2_mbox_conf,
                conv9_2_mbox_conf,
                conv4_3_norm_mbox_loc,
                fc7_mbox_loc,
                conv6_2_mbox_loc,
                conv7_2_mbox_loc,
                conv8_2_mbox_loc,
                conv9_2_mbox_loc,
                image_size,
                n_classes,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                coords='centroids',
                normalize_coords=True,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200):
    
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    # conv4_3_norm_mbox_loc = conv4_3_norm_mbox_loc
    conv4_3_norm_mbox_conf_reshape = conv4_3_norm_mbox_conf.reshape(compute_output_shape(conv4_3_norm_mbox_conf,(-1,n_classes)))
    fc7_mbox_conf_reshape = fc7_mbox_conf.reshape(compute_output_shape(fc7_mbox_conf,(-1,n_classes)))
    conv6_2_mbox_conf_reshape = conv6_2_mbox_conf.reshape(compute_output_shape(conv6_2_mbox_conf,(-1,n_classes)))
    conv7_2_mbox_conf_reshape = conv7_2_mbox_conf.reshape(compute_output_shape(conv7_2_mbox_conf,(-1,n_classes)))
    conv8_2_mbox_conf_reshape = conv8_2_mbox_conf.reshape(compute_output_shape(conv8_2_mbox_conf,(-1,n_classes)))
    conv9_2_mbox_conf_reshape = conv9_2_mbox_conf.reshape(compute_output_shape(conv9_2_mbox_conf,(-1,n_classes)))

    conv4_3_norm_mbox_loc_reshape = conv4_3_norm_mbox_loc.reshape(compute_output_shape(conv4_3_norm_mbox_loc,(-1,4)))
    fc7_mbox_loc_reshape = fc7_mbox_loc.reshape(compute_output_shape(fc7_mbox_loc,(-1,4)))
    conv6_2_mbox_loc_reshape = conv6_2_mbox_loc.reshape(compute_output_shape(conv6_2_mbox_loc,(-1,4)))
    conv7_2_mbox_loc_reshape = conv7_2_mbox_loc.reshape(compute_output_shape(conv7_2_mbox_loc,(-1,4)))
    conv8_2_mbox_loc_reshape = conv8_2_mbox_loc.reshape(compute_output_shape(conv8_2_mbox_loc,(-1,4)))
    conv9_2_mbox_loc_reshape = conv9_2_mbox_loc.reshape(compute_output_shape(conv9_2_mbox_loc,(-1,4)))

    conv4_3_norm_mbox_priorbox = anchorbox(conv4_3_norm_mbox_loc,this_scale = scales[0],next_scale = scales[1],\
        aspect_ratios = aspect_ratios_per_layer[0],this_steps=steps[0],this_offsets=offsets[0])
    fc7_mbox_priorbox = anchorbox(fc7_mbox_loc,this_scale = scales[1],next_scale = scales[2],\
        aspect_ratios = aspect_ratios_per_layer[1],this_steps=steps[1],this_offsets=offsets[1])
    conv6_2_mbox_priorbox = anchorbox(conv6_2_mbox_loc,this_scale = scales[2],next_scale = scales[3],\
        aspect_ratios = aspect_ratios_per_layer[2],this_steps=steps[2],this_offsets=offsets[2])
    conv7_2_mbox_priorbox = anchorbox(conv7_2_mbox_loc,this_scale = scales[3],next_scale = scales[4],\
        aspect_ratios = aspect_ratios_per_layer[3],this_steps=steps[3],this_offsets=offsets[3])
    conv8_2_mbox_priorbox = anchorbox(conv8_2_mbox_loc,this_scale = scales[4],next_scale = scales[5],\
        aspect_ratios = aspect_ratios_per_layer[4],this_steps=steps[4],this_offsets=offsets[4])
    conv9_2_mbox_priorbox = anchorbox(conv9_2_mbox_loc,this_scale = scales[5],next_scale = scales[6],\
        aspect_ratios = aspect_ratios_per_layer[5],this_steps=steps[5],this_offsets=offsets[5])

    conv4_3_norm_mbox_priorbox_reshape = conv4_3_norm_mbox_priorbox.reshape(compute_output_shape(conv4_3_norm_mbox_priorbox,(-1,8)))
    fc7_mbox_priorbox_reshape = fc7_mbox_priorbox.reshape(compute_output_shape(fc7_mbox_priorbox,(-1,8)))
    conv6_2_mbox_priorbox_reshape = conv6_2_mbox_priorbox.reshape(compute_output_shape(conv6_2_mbox_priorbox,(-1,8)))
    conv7_2_mbox_priorbox_reshape = conv7_2_mbox_priorbox.reshape(compute_output_shape(conv7_2_mbox_priorbox,(-1,8)))
    conv8_2_mbox_priorbox_reshape = conv8_2_mbox_priorbox.reshape(compute_output_shape(conv8_2_mbox_priorbox,(-1,8)))
    conv9_2_mbox_priorbox_reshape = conv9_2_mbox_priorbox.reshape(compute_output_shape(conv9_2_mbox_priorbox,(-1,8)))

    mbox_conf = np.concatenate((conv4_3_norm_mbox_conf_reshape,fc7_mbox_conf_reshape,conv6_2_mbox_conf_reshape,\
        conv7_2_mbox_conf_reshape,conv8_2_mbox_conf_reshape,conv9_2_mbox_conf_reshape), axis = 1)
    mbox_loc = np.concatenate((conv4_3_norm_mbox_loc_reshape,fc7_mbox_loc_reshape,conv6_2_mbox_loc_reshape,\
        conv7_2_mbox_loc_reshape,conv8_2_mbox_loc_reshape,conv9_2_mbox_loc_reshape), axis = 1)
    mbox_priorbox = np.concatenate((conv4_3_norm_mbox_priorbox_reshape,fc7_mbox_priorbox_reshape,conv6_2_mbox_priorbox_reshape,\
        conv7_2_mbox_priorbox_reshape,conv8_2_mbox_priorbox_reshape,conv9_2_mbox_priorbox_reshape), axis = 1)
    mbox_priorbox = np.repeat(mbox_priorbox,mbox_conf.shape[0], axis = 0)
    mbox_conf_softmax = softmax(mbox_conf)
    predictions = np.concatenate((mbox_conf_softmax,mbox_loc,mbox_priorbox), axis = 2)

    y_pred = decode_detections(predictions,
                      confidence_thresh=confidence_thresh,
                      iou_threshold=iou_threshold,
                      top_k=top_k,
                      input_coords=coords,
                      normalize_coords=normalize_coords,
                      img_height=img_height,
                      img_width=img_width)

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", help="height of model input ",type=int, default=300)
    parser.add_argument("--img_width", help="width of model input ",type=int, default=300)
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
        line = line.replace('\n','')
        classes.append(line)
    f.close()
    if 'background' not in classes:
        classes.insert(0,'background')
    n_classes = len(classes)


    image_list = []
    for file in os.listdir(args.input_dir):
        img_name = str(file).split('img_')[1].replace('.npy','')
        if img_name not in image_list:
            image_list.append(img_name)

    for img in image_list:
        for file in os.listdir(args.input_dir):
            if img in file:
                temp = np.load(os.path.join(args.input_dir, file))
                if temp.shape == (1,38,38,4*n_classes):
                   conv4_3_norm_mbox_conf = temp
                elif temp.shape == (1,19,19,6*n_classes):
                   fc7_mbox_conf = temp
                elif temp.shape == (1,10,10,6*n_classes):
                   conv6_2_mbox_conf = temp
                elif temp.shape == (1,5,5,6*n_classes):
                   conv7_2_mbox_conf = temp
                elif temp.shape == (1,3,3,4*n_classes):
                   conv8_2_mbox_conf = temp
                elif temp.shape == (1,1,1,4*n_classes):
                   conv9_2_mbox_conf = temp
                elif temp.shape == (1,38,38,16):
                   conv4_3_norm_mbox_loc = temp
                elif temp.shape == (1,19,19,24):
                   fc7_mbox_loc = temp
                elif temp.shape == (1,10,10,24):
                   conv6_2_mbox_loc = temp
                elif temp.shape == (1,5,5,24):
                   conv7_2_mbox_loc = temp
                elif temp.shape == (1,3,3,16):
                   conv8_2_mbox_loc = temp
                elif temp.shape == (1,1,1,16):
                   conv9_2_mbox_loc = temp
        y_postprocess = postprocess(conv4_3_norm_mbox_conf,
                fc7_mbox_conf,
                conv6_2_mbox_conf,
                conv7_2_mbox_conf,
                conv8_2_mbox_conf,
                conv9_2_mbox_conf,
                conv4_3_norm_mbox_loc,
                fc7_mbox_loc,
                conv6_2_mbox_loc,
                conv7_2_mbox_loc,
                conv8_2_mbox_loc,
                conv9_2_mbox_loc,
                image_size = (img_height,img_width,3),
                n_classes = n_classes)
        for file in os.listdir(args.image_dir):
            if img in file:
                image = Image.open(os.path.join(args.image_dir,file))

        
        f = open(os.path.join(args.output_dir,img+'.txt'),mode='w')
        for y_pred in y_postprocess[0]:
            class_id = int(y_pred[0])
            confidence = y_pred[1]
            class_name = classes[class_id]
            xmin = max(int(y_pred[2] * image.width / img_width),0)
            ymin = max(int(y_pred[3] * image.height / img_height),0)
            xmax = min(int(y_pred[4] * image.width / img_width),image.width-1)
            ymax = min(int(y_pred[5] * image.height / img_height),image.height-1)
            f.write((class_name + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)+'\n'))
        f.close()
