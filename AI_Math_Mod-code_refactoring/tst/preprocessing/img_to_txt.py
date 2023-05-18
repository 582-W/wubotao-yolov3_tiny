import unittest
import sys
import os
import shutil
from PIL import Image
import numpy as np
#sys.path.append("..")
sys.path.append("../..")
from preprocessing.img_to_txt import ImgPreprocessor

class TestImgPreprocessor(unittest.TestCase):
    def setUp(self):
        self.test_obj = ImgPreprocessor()
        self.case_path = "../case_files/case_preprocess"
        self.output_folder_path = "tmp"
        if os.path.exists(self.output_folder_path):
            shutil.rmtree(self.output_folder_path)
        os.mkdir(self.output_folder_path)
        
    def test_get_datapath_input(self):
        #imgPreprocessor = ImgPreprocessor()
        
        self.assertEqual(self.test_obj.get_datapath_input('yolo_keras'), 1.)
        self.assertEqual(self.test_obj.get_datapath_input('vgg_keras'), 151)
        self.assertEqual(self.test_obj.get_datapath_input('ssd_keras'), 151)
        self.assertEqual(self.test_obj.get_datapath_input('other'), 1.)
        
    def test_preprocess(self) -> None:
        #imgPreprocess = ImgPreprocessor()
        input_image_folder_path = os.path.join(self.case_path, 'input_image')
        input_1 = np.load(os.path.join(self.case_path, 'input_1.npy'))
        input_2 = np.load(os.path.join(self.case_path, 'input_2.npy'))
        input_3 = np.load(os.path.join(self.case_path, 'input_3.npy'))
        input_4 = np.load(os.path.join(self.case_path, 'input_4.npy'))
        
        self.test_obj.preprocess(input_image_folder_path, self.output_folder_path, 
            method = 'yolo_keras', target_size = '300, 300', channel_num = 'RGB', 
            output_format = 'npy', keep_aspect_ratio = True)
        output_npy_path = self.test_obj._read_directory(self.output_folder_path)
        self.assertTrue((np.load(output_npy_path[0]) == input_1).all())
        
        self.test_obj.preprocess(input_image_folder_path, self.output_folder_path, 
            method = 'ssd_keras', target_size = '300, 300', channel_num = 'BGR', 
            output_format = 'npy', keep_aspect_ratio = True)
        output_npy_path = self.test_obj._read_directory(self.output_folder_path)
        self.assertTrue((np.load(output_npy_path[0]) == input_2).all())
        
        self.test_obj.preprocess(input_image_folder_path, self.output_folder_path, 
            method = 'vgg_keras', target_size = '300, 300', channel_num = 'BGR', 
            output_format = 'npy', keep_aspect_ratio = True)
        output_npy_path = self.test_obj._read_directory(self.output_folder_path)
        self.assertTrue((np.load(output_npy_path[0]) == input_3).all())

        self.test_obj.preprocess(input_image_folder_path, self.output_folder_path, 
            method = 'mobilenet_keras', target_size = '300, 300', channel_num = 'RGB', 
            output_format = 'npy', keep_aspect_ratio = True)
        output_npy_path = self.test_obj._read_directory(self.output_folder_path)
        self.assertTrue((np.load(output_npy_path[0]) == input_4).all())
        
    
    def test_load_img(self) -> None:
        #imgPreprocessor = ImgPreprocessor()
        L_img_path = os.path.join(self.case_path, "ILSVRC2012_val_00002796.JPEG")
        RGB_img_path = os.path.join(self.case_path, "ILSVRC2012_val_00000016.JPEG")
        img_1 = np.asarray(Image.open(os.path.join(self.case_path, 'img_1.png')))
        img_2 = np.asarray(Image.open(os.path.join(self.case_path, 'img_2.png')))
        img_3 = np.asarray(Image.open(os.path.join(self.case_path, 'img_3.png')))
        img_4 = np.asarray(Image.open(os.path.join(self.case_path, 'img_4.png')))
        
        img = self.test_obj._load_img(L_img_path, channel_num = 'L', 
                                      target_size = [224, 224], interpolation = 'nearest', 
                                      keep_aspect_ratio = True)
        img = np.asarray(img)
        self.assertTrue((img == img_1).all())
        
        img = self.test_obj._load_img(L_img_path, channel_num = 'RGB', 
                                      target_size = [224, 224], interpolation = 'nearest', 
                                      keep_aspect_ratio = False)
        img = np.asarray(img)
        self.assertTrue((img == img_2).all())
        
        img = self.test_obj._load_img(RGB_img_path, channel_num = 'L', 
                                      target_size = [224, 224], interpolation = 'nearest', 
                                      keep_aspect_ratio = True)
        img = np.asarray(img)
        self.assertTrue((img == img_3).all())
        
        img = self.test_obj._load_img(RGB_img_path, channel_num = 'RGB', 
                                      target_size = [224, 224], interpolation = 'nearest', 
                                      keep_aspect_ratio = False)
        img = np.asarray(img)
        self.assertTrue((img == img_4).all())
        
    def test_img_to_array(self) -> None:
        #imgPreprocessor = ImgPreprocessor()
        img = Image.open(os.path.join(self.case_path, 'img_4.jpg'))
        nparray_1 = np.load(os.path.join(self.case_path, 'nparray_1.npy'))
        nparray_2 = np.load(os.path.join(self.case_path, 'nparray_2.npy'))
        
        self.assertTrue((self.test_obj._img_to_array(img, channel_num='BGR', 
                         data_format='channels_last') == nparray_1).all())
        self.assertTrue((self.test_obj._img_to_array(img, channel_num='RGB', 
                         data_format='channels_first') == nparray_2).all())
        


if __name__ == '__main__': 
    unittest.main()
    