import os
import numpy as np
import pdb

samples = 'samples/'
testcases = 'testcases/'

list1 = os.listdir(samples)
list2 = os.listdir(testcases)

for f1 in list1:
    for f2 in list2:
        if os.path.basename(f1) == os.path.basename(f2):
            print('\n\n')
            print('start to verify one mode')
            # verify float res
            s_path = samples + os.path.basename(f1) + "/float_inference_output/"
            t_path = testcases + os.path.basename(f2) + "/float_inference_output/"
            print(t_path)
            path_list=os.listdir(t_path)
            path_list.sort()
            for output in path_list:
                assert(output in os.listdir(s_path))
                a = np.load(os.path.join(s_path,output))
                b = np.load(os.path.join(t_path,output))
                if not (a==b).all():
                    print(output,'false')
                else:
                    print(output,'true')
            # verify dfp res
            s_path = samples + os.path.basename(f1) + "/fp_inference_output/float_res/"
            t_path = testcases + os.path.basename(f2) + "/fp_inference_output/float_res/"
            print(t_path)
            path_list=os.listdir(t_path)
            path_list.sort()
            for output in path_list:
                assert(output in os.listdir(s_path))
                a = np.load(os.path.join(s_path,output))
                b = np.load(os.path.join(t_path,output))
                if not (a==b).all():
                    print(output,'false')
                else:
                    print(output,'true')
            # remove update.onnx
            update_onnx_s = samples + os.path.basename(f1) + "/model_update_output/update.onnx"
            update_onnx_t = testcases + os.path.basename(f2) + "/model_update_output/update.onnx"
            if os.path.isfile(update_onnx_s):
                os.remove(update_onnx_s)
            if os.path.isfile(update_onnx_t):
                os.remove(update_onnx_t)

