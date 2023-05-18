import os
import onnx
import sys
import shutil
import unittest
sys.path.append("..")
sys.path.append("../..")
from inference.inferencer import Inferencer
from model_update.model_update import ModelUpdater

class TestModelUpdate(unittest.TestCase):
    ##usage python tst/modelupdate/modelupdate.py
    def setUp(self):
        self.case_path = "tst/case_files/case_modelupdate/"   ### case path
        self.model_ori = os.path.join(self.case_path, "model_ori.onnx")
        self.image_npy_folder = os.path.join(self.case_path, "img_npy")
        self.weight_analysis_json_path = os.path.join(self.case_path, "weight_analysis.json")
        self.datapath_analysis_json_path = os.path.join(self.case_path, "datapath_analysis.json")
        self.config_file_path = 'tst/case_files/hardware.json'
        self.model_update_output_folder_path = os.path.join(self.case_path, "output")
        self.whether_limit_shift = True
        self.shift_upper_limit = 35
        self.shift_lower_limit = -4



    def test_update(self):
        '''
        test modelupdate

        '''
        for analysis_mode in ['per_channel_weight','per_layer']:

            modelupdater = ModelUpdater(self.model_ori,self.image_npy_folder,self.weight_analysis_json_path,self.datapath_analysis_json_path,self.config_file_path, \
                                        self.model_update_output_folder_path,analysis_mode,run_update=True,img_preprocess_method='yolo_keras',concat_change=True)
            modelupdater.model_update(self.whether_limit_shift, self.shift_upper_limit, self.shift_lower_limit)
            model_test = Inferencer(os.path.join(self.model_update_output_folder_path, "update.onnx"))
            model_golden = Inferencer(os.path.join(self.case_path, f'update_{analysis_mode}.onnx'))
            
            for i in range(len(model_test.weights)):
                w_test = onnx.numpy_helper.to_array(model_test.weights[i])
                w_golden = onnx.numpy_helper.to_array(model_golden.weights[i])
                self.assertTrue((w_test == w_golden).all())
            
            shutil.rmtree(self.model_update_output_folder_path)
        

    
if __name__ == '__main__': 
    unittest.main()