# Introduction

The codes implement to change the outputs of yolov3 in onnx model which you can get by the **ModelInferencer** to the standard detection results as below, and save the detection results to TXT files so you can calculate the mAP of yolov3 model by the  **yolo_mAP_evaluation_framework** 



 bottle 0.14981 80 1 295 500
 
 bus 0.12601 36 13 404 316
 
 horse 0.12526 430 117 500 307
 
 pottedplant 0.14585 212 78 292 118
 
 tvmonitor 0.070565 388 89 500 196
 



You can run the python  **`yolo_output2standard.py`**  by changing the parameters as below.

**img_path**: the directory that stores the images to detect.

**onnx_output_path**: the directory that stores the outputs of yolov3 in onnx model.

**detection_result_path**: the directory to save the detection results using the TXT file.

**draw_bbox**: If you set **True**, you can get the bounding box for the image.
