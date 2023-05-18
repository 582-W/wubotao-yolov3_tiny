import numpy as np
import os
from PIL import Image
import shutil
import numbers
import cv2

class ImgPreprocessor(object):

    def __init__(self):
        super().__init__()
    
    
    def preprocess(self, input_image_folder_path:str, output_folder_path:str, method:str, 
                   target_size:str, channel_num:str, output_format = "npy", 
                   keep_aspect_ratio = True) -> None:
        
        '''
        Preprocess images needed for the network.

        # Arguments
            input_image_folder_path: Path of image datasets
            output_folder_path: The output path of preprocessing
            method: Type of network
                    Options: yolo_keras / ssd_keras / vgg_keras / mobilenet_keras / resnet_pytorch
            target_size: Input image size of network
            channel_num: Output format of image
                         Options: RGB / BGR / L
            output_format: The format of the output file
                           Options: txt / npy
                           Default: npy
            keep_aspect_ratio: Whether keep aspect ratio, false for yolo method
                               Options: True / False
                               Default: True

        # Raises
            ValueError: if channel_num is incorrect.
        '''
        # read input_size of network from input_params.json
        if type(target_size) == str:
            target_size = target_size.split(',')
            target_size = [int(i) for i in target_size]
        if len(target_size) != 2:
            raise ValueError('Expected `size` to be a tuple of 2 integers, '
                             'but got: {}'.format(target_size,))
        
        if os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
        os.mkdir(output_folder_path)
        img_path = self._read_directory(input_image_folder_path)
        
        for img_name in img_path:
            if method == 'yolo_keras':
                img = self._load_img(img_name, channel_num=channel_num, 
                                     target_size=target_size, 
                                     interpolation='bicubic', 
                                     keep_aspect_ratio=keep_aspect_ratio)
                img = self._img_to_array(img, channel_num = channel_num)
                img = np.expand_dims(img, axis=0)
                img = img.astype(float)
                if channel_num == 'RGB':
                    img /= 256.
                else:
                    raise ValueError('Unsupported channel num for yolo_keras!')
                img = img.transpose(0,3,1,2)
            elif method == 'resnet_pytorch':
                img = self._load_img_resnet_pytorch(img_name, channel_num=channel_num, 
                                     target_size=target_size)
                img = self._img_to_array(img, channel_num = channel_num)   
                img = np.expand_dims(img, axis=0)   
                img = img.astype(float)
                if channel_num == 'RGB':
                    img /= 255
                    mean = [0.485,0.456,0.406]
                    std = [0.229,0.224,0.225]
                    img[..., 0] = (img[..., 0]-mean[0])/std[0]
                    img[..., 1] = (img[..., 1]-mean[1])/std[1]
                    img[..., 2] = (img[..., 2]-mean[2])/std[2]
                else:
                    raise ValueError('Unsupported channel num for resnet_pytorch!')  
                img = img.transpose(0,3,1,2)             
            elif method == 'yolov4_pytorch':
                img = self._load_img_yolov4_pytorch(img_name, channel_num=channel_num,target_size=target_size)
                img = img.astype(float)
                if channel_num == 'RGB':
                    img = img.transpose(2, 0, 1)  # .unsqueeze(0)
                    img /= 255.
                    img = np.expand_dims(img, axis=0)
                else:
                    raise ValueError('Unsupported channel num for yolov4_pytorch!')
            elif method == 'yolov5_pytorch' or method == 'yolov3_pytorch':
                img = self._load_img_yolov5_pytorch(img_name, channel_num=channel_num,target_size=target_size)
                img = np.expand_dims(img, axis=0)
                img = img.astype(float)   
                img = img.astype(np.float32)
                img /= 255  # 0 - 255 to 0.0 - 1.0           
            elif method == 'yolo_keras':
                img = self._load_img(img_name, channel_num=channel_num, 
                                     target_size=target_size, 
                                     interpolation='bicubic', 
                                     keep_aspect_ratio=keep_aspect_ratio)
                img = self._img_to_array(img, channel_num = channel_num)
                img = np.expand_dims(img, axis=0)
                img = img.astype(float)
                if channel_num == 'RGB':
                    img /= 256.
                else:
                    raise ValueError('Unsupported channel num for yolo_keras!')
                img = img.transpose(0,3,1,2)
            elif method == 'ssd_keras': 
                img = self._load_img(img_name, channel_num=channel_num, 
                                     target_size=target_size, 
                                     keep_aspect_ratio=keep_aspect_ratio)
                img = self._img_to_array(img, channel_num = channel_num)
                img = np.expand_dims(img, axis=0) 
                img = img.astype(float)  
                if channel_num == 'BGR':
                    mean = [104, 117, 123]
                    img[..., 0] -= mean[0]
                    img[..., 1] -= mean[1]
                    img[..., 2] -= mean[2]
                    #img[:,:,:,[0, 2]] = img[:,:,:,[2, 0]]
                else:
                    raise ValueError('Unsupported channel num for ssd_keras!')
                img = img.transpose(0,3,1,2)
            elif method == 'vgg_keras':
                img = self._load_img(img_name, channel_num=channel_num, 
                                     target_size=target_size, 
                                     keep_aspect_ratio=keep_aspect_ratio)
                img = self._img_to_array(img, channel_num = channel_num)
                img = np.expand_dims(img, axis=0) 
                img = img.astype(float)  
                if channel_num == 'BGR':
                    #img = img[..., ::-1]
                    mean = [103.939, 116.779, 123.68]
                    img[..., 0] -= mean[0]
                    img[..., 1] -= mean[1]
                    img[..., 2] -= mean[2]
                else:
                    raise ValueError('Unsupported channel num for vgg_keras!')
                img = img.transpose(0,3,1,2)
            elif method == 'mobilenet_keras':
                img = self._load_img(img_name, channel_num=channel_num, 
                                     target_size=target_size, 
                                     keep_aspect_ratio=keep_aspect_ratio)
                img = self._img_to_array(img, channel_num = channel_num)
                img = np.expand_dims(img, axis=0)
                img = img.astype(float)
                if channel_num == 'RGB':
                    img /= 127.5
                    img -= 1
                else:
                    raise ValueError('Unsupported channel num for mobilenet_keras!')
                img = img.transpose(0,3,1,2)
            else:
                raise ValueError('Unsupported method!')
            image_name = os.path.splitext(os.path.split(img_name)[1])[0]

            if output_format == "npy":
                # np.random.seed(1) 
                # img = np.random.rand(1,1,1024,1024)
                # img = cv2.imread('/home/aojie/temp/RepLFSR_32/img/input_1024x1024.png',cv2.IMREAD_GRAYSCALE)/255.0 
                # img = np.expand_dims(img, axis=0)  
                # img = np.expand_dims(img, axis=0)
                # print(img[0][1][300])
                np.save('{}/{}.npy'.format(output_folder_path, image_name), img)
            elif output_format == "txt":
                np.savetxt('{}/{}.txt'.format(output_folder_path, image_name), img)
    
    
    def get_datapath_input(self, method:str):
        data_max = 1.
        if method == "yolo_keras":
            data_max = 1.
        elif method == "mobilenet_keras":
            data_max = 1.
        elif method == "vgg_keras":
            data_max = 151
        elif method == "ssd_keras":
            data_max = 151
        elif method == "resnet_pytorch":
            data_max = 3
        elif method == "yolov3_pytorch":
            data_max = 1
        elif method == "yolov4_pytorch":
            data_max = 1
        elif method == "yolov5_pytorch":
            data_max = 1
        return data_max
    
    
    def _read_directory(self, directory_name:str) -> list:
        file_path = []
        for filename in os.listdir(directory_name):
            filepath = os.path.join(directory_name, filename)
            file_path.append(filepath)
        return file_path
    
    
    def _load_img(self, path:str, channel_num:str, target_size=None, 
                  interpolation='nearest', keep_aspect_ratio = True) -> list:
        
        """Loads an image into PIL format.

        # Arguments
            path: Path to image file
            channel_num: Output format of image
                         Options: RGB / BGR / L
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        
        if Image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        _PIL_INTERPOLATION_METHODS = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
        }
        img = Image.open(path)
        iw, ih = img.size
        h, w = target_size
        if channel_num == 'L':
            if img.mode != 'L':
                img = img.convert('L')
        elif channel_num == 'RGB' or channel_num == 'BGR':
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        if target_size is not None:
            width_height_tuple = (w, h)
            if img.size != width_height_tuple:
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                if keep_aspect_ratio == True:
                    scale = min(w/iw, h/ih)
                    nw = int(iw*scale)
                    nh = int(ih*scale)
                    img = img.resize((nw,nh), resample)
                    new_image = Image.new('RGB', width_height_tuple, (128,128,128))
                    new_image.paste(img, ((w-nw)//2, (h-nh)//2))
                    img = new_image
                else:
                    img = img.resize(width_height_tuple, resample)
        return img
    
    def _load_img_resnet_pytorch(self, path, channel_num, target_size):
        """Loads an image into PIL format.

        # Arguments
            path: Path to image file
            channel_num: Output format of image
                         Options: RGB / BGR / L
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        img = Image.open(path).convert(channel_num)
        size = 256
        interpolation = Image.BILINEAR
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = img
            if w < h:
                ow = size
                oh = int(size * h / w)
                img = img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                img = img.resize((ow, oh), interpolation)
        else:
            img = img.resize(size[::-1], interpolation)

        if isinstance(target_size, numbers.Number):
            target_size = (int(target_size), int(target_size))
        w, h = img.size
        th, tw = target_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        img = img.crop((j, i, j + tw, i + th))

        return img
    def _load_img_yolov4_pytorch(self, path, channel_num, target_size):
        """Loads an image into cv format.

        # Arguments
            path: Path to image file
            channel_num: Output format of image
                         Options: RGB / BGR / L
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        h, w = target_size
        img = cv2.imread(path)
        img = np.array(img)
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_img_yolov5_pytorch(self, path, channel_num, target_size):
        """Loads an image into cv format.

        # Arguments
            path: Path to image file
            channel_num: Output format of image
                         Options: RGB / BGR / L
            target_size: Either `None` (default to original size)
                or tuple of ints `(img_height, img_width)`.

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if interpolation method is not supported.
        """
        im = cv2.imread(path)
        color=(114, 114, 114)
        stride=32
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(target_size[0] / shape[0], target_size[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]  # wh padding

        # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        # np.save("C:/Users/abcdef/Desktop/pytorch-YOLOv4-master/img/windows.npy", im)
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return im

    def _img_to_array(self, img:list, channel_num:str, data_format='channels_last'):
        
        """Converts a PIL Image instance to a Numpy array.

        # Arguments
            img: PIL Image instance.
            data_format: Image data format.

        # Returns
            A 3D Numpy array.

        # Raises
            ValueError: if invalid `img` or `data_format` is passed.
        """
        
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ', data_format)
        # Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but original PIL image has format (width, height, channel)
        x = np.asarray(img)
        if len(x.shape) == 3:
            if data_format == 'channels_first':
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if data_format == 'channels_first':
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: ', x.shape)
            
        if channel_num == 'BGR':
            x = x[..., ::-1]
        return x
    
    

        
