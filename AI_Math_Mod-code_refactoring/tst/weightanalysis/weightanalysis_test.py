import unittest
import onnx
import numpy as np
import json
import operator
from .weight_analysis.weight_analyser import WeightAnalyser


class TestWeightAnalyser(unittest.TestCase):


    def setUp(self):
        self.model_file_path = "./data/unittest.onnx"
        self.result_json_path = "./data/res_json"
        self.model = onnx.load_model(self.model_file_path)
        self.weight_dict = {'scale10': 0, 'bias10': 1, 'mean10': 2, 'var10': 3, 'conv2d_1/kernel:0': 4}
        self.weightanalyser_per_layer = WeightAnalyser(self.model_file_path, self.result_json_path, "per_layer")
    

    def test_init(self):
        self.assertEqual(self.weightanalyser_per_layer.analysis_mode, "per_layer")
        self.assertEqual(self.weightanalyser_per_layer.CONV_PERCENT, 1.0)
        self.assertEqual(self.weightanalyser_per_layer.BN_PERCENT, 1.0)
        self.assertEqual(self.weightanalyser_per_layer.GROUP_SIZE, 1)


    def test_obtain_helper_dict(self):
        standard_answer = self.weight_dict
        test_res = self.weightanalyser_per_layer._obtain_helper_dict()
        self.assertEqual(test_res, standard_answer)
    

    def test_parse_conv_weight(self):
        conv_node = self.model.graph.node[0]
        weight_first_nine = [0.18700912594795227, 0.4113427698612213, 0.22223861515522003, 
                            0.4927622079849243, -0.22341439127922058, -0.45198026299476624, 
                            -0.014923283830285072, -0.8258866667747498, -0.11931180953979492]

        weight_last_nine = [0.5685935020446777, 0.6504067182540894, 0.024485375732183456, 
                            0.6721663475036621, 0.9477818608283997, 0.763622522354126, 
                            0.46941956877708435, 0.6983206272125244, 0.6281874179840088]
        
        w, w_kernel, w_bias  = self.weightanalyser_per_layer.parse_conv_weight(conv_node, self.weight_dict)
        self.assertEqual(weight_first_nine, w[:9])
        self.assertEqual(weight_last_nine, w[-9:])


    def test_parse_bn_parameters(self):
        bn_node = self.model.graph.node[1]
        A_list_first_three = [0.9999995000003763, 0.9999995000003763, 0.9999995000003763]
        B_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        test_A_list, test_B_list = self.weightanalyser_per_layer.parse_bn_parameters(bn_node, self.weight_dict)
        self.assertEqual(A_list_first_three, test_A_list[:3])
        self.assertEqual(B_list, test_B_list)


    def test_remove_outliers(self):
        test_data = [i for i in range(10)]
        test_res_std = [2, 3, 4, 5, 6]
        test_res = self.weightanalyser_per_layer._remove_outliers(test_data, 0.5)
        test_res = list(test_res)
        self.assertEqual(test_res_std, test_res)


    def test_analyse_weight(self):
        with open ("./data/standard.json", "r") as load_file:
            standard_dict = json.load(load_file)
        per_layer_test, file_path = self.weightanalyser_per_layer.analyse_weight()
        self.assertTrue(operator.eq(standard_dict["per_layer"], per_layer_test))

        self.weightanalyser_per_channel = WeightAnalyser(self.model_file_path, self.result_json_path, "per_channel_weight")
        per_channel_test, file_path_p = self.weightanalyser_per_channel.analyse_weight()
        self.assertTrue(operator.eq(standard_dict["per_channel_weight"], per_channel_test))


if __name__ == '__main__':
    unittest.main()

