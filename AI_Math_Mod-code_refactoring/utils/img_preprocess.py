import sys,getopt
import numpy as np
import os
import json
from PIL import Image
import math
import shutil

def read_directory(directory_name):
    file_path = []
    for filename in os.listdir(directory_name):
        filepath = directory_name + "/" + filename
        file_path.append(filepath)
    return file_path
        


def img_process_core(image_name, target_size, keep_aspect_ratio, method):
    
    if method == "mobilenet" or method == "vgg":
        keep_aspect_ratio = False
    
    img = Image.open(image_name)
    iw, ih = img.size
    h, w = target_size
    if keep_aspect_ratio:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        if method == "mobilenet" or method == "vgg":
            img = img.resize((nw,nh), Image.NEAREST)
        else:
            img = img.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', tuple([w, h]), (128,128,128))
        new_image.paste(img, ((w-nw)//2, (h-nh)//2))
        img = new_image
    else:
        if method == "mobilenet" or method == "vgg":
            img = img.resize((w,h), Image.NEAREST)
        else:
            img = img.resize((w,h), Image.BICUBIC)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
    #convert the image pixels to a numpy array
    img = np.array(img)
    #add one more dimension
    img = np.expand_dims(img, axis=0)
   
    img = img.astype(float)
   
    #preprocess the image
    
    if method == "yolo":
        img /= 256.
    elif method == "vgg":
        img = img[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
    elif method == "mobilenet":
        img /= 127.5
        img -= 1.
    
    return img

# target_size = (h, w)
def img_preprocess(input_image_folder_path, image_npy_folder, method, target_size = (416, 416),  keep_aspect_ratio = True):
    if type(target_size) == str:
        target_size = target_size.split(',')
        target_size = [int(i) for i in target_size]

    if os.path.exists(image_npy_folder):
        shutil.rmtree(image_npy_folder)
    os.mkdir(image_npy_folder)
    img_path = read_directory(input_image_folder_path)

    for image_name in img_path:
        if method == 'ssd':
            img = preprocess(image_name, input_shape=(300, 300),subtract_mean=[123, 117, 104])
        else:
            img = img_process_core(image_name, target_size, keep_aspect_ratio, method)
        img_name = os.path.splitext(os.path.split(image_name)[1])[0]
        np.save('{}/{}.npy'.format(image_npy_folder, img_name), img)
    
    return img_path
 
def img_preprocess_multi(input_image_folder_path, image_npy_folder, method, multipro, target_size = (416, 416), keep_aspect_ratio = True):
    
    if os.path.exists(image_npy_folder):
        shutil.rmtree(image_npy_folder)
    os.mkdir(image_npy_folder)
    img_path = read_directory(input_image_folder_path)
    #print(img_path)
    for k in range(multipro):
        os.mkdir(image_npy_folder + '/' + str(k))
    k = 0
    if method == 'yolo':
        for image_name in img_path:
            if k == multipro:
                k = 0
            img = img_process_core(image_name, target_size, keep_aspect_ratio, method)
            img_name = os.path.splitext(os.path.split(image_name)[1])[0]
            np.save('{}/{}.npy'.format(image_npy_folder + '/' + str(k), img_name), img)
            k = k + 1
    if method == 'vgg':
        for image_name in img_path:
            if k == multipro:
                k = 0            
            img = img_process_core(image_name, target_size, keep_aspect_ratio, method)
            img_name = os.path.splitext(os.path.split(image_name)[1])[0]
            np.save('{}/{}.npy'.format(image_npy_folder + '/' + str(k), img_name), img)
            k = k + 1
    return img_path             


### SSD
def load_img(path, grayscale=False, target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
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
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def img_to_array(img, data_format='channels_last'):
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
    return x


def preprocess(img_path, input_shape,subtract_mean=[123, 117, 104]):
    """
    input_shape: (h, w)
    """
    #input_images = [] # Store resized versions of the images here.
    img = load_img(img_path,target_size=input_shape)
    img = img_to_array(img)
    img = img - np.array(subtract_mean)
    img[:,:,[0, 2]] = img[:,:,[2, 0]]
    input_images = []
    input_images.append(img)
    input_images = np.array(input_images)
    return input_images
