import os
import numpy as np
import argparse
import shutil
from sklearn import metrics
import json

parser = argparse.ArgumentParser()
parser.add_argument("--fp_folder", help="dir of test image")
parser.add_argument("--iee_folder", help="dir of draw image output")
parser.add_argument("--fp_folder_fl", help="dir of test image")
parser.add_argument("--iee_folder_fl", help="dir of draw image output")
parser.add_argument("--file_name", help="dir of draw image output")
args = parser.parse_args()
# 定义两个文件夹的路径
folder1 = args.fp_folder
folder2 = args.iee_folder
folder3 = args.fp_folder_fl
folder4 = args.iee_folder_fl
file_name = args.file_name
data = {}
data['int'] = {}
# 遍历两个文件夹中的numpy文件
for file1 in os.listdir(folder1):
    if file1.endswith(".npy"):
        # 读取numpy文件
        array1 = np.load(os.path.join(folder1, file1)).reshape(-1)
        # 查找与file1文件同名的文件
        file2 = os.path.join(folder2, file1)
        if os.path.exists(file2):
            print(file1)
            data['int'][file1]={}
            # 读取numpy文件
            array2 = np.load(file2).transpose(0,3,1,2).reshape(-1)
            # 计算绝对值平# 计算绝对值平均误差
            # error = np.mean_absolute_error(array1, array2)
            MSE = metrics.mean_squared_error(array1, array2)
            
            # print(file1 + ": mean_squared_error is: ", MSE)
            MAE = metrics.mean_absolute_error(array1, array2)
            # print(file1 + ": mean_absolute_error is: ", MAE)
            MAPE = metrics.mean_absolute_percentage_error(array1, array2)
            # print(file1 + ": mean_absolute_percentage_error is: ", MAPE)
            # 计算相等元素的个数
            count = np.count_nonzero(np.equal(array1, array2))
            # print(file1 + ": The number of equal elements is: ", count)
            # print(file1 + ": The number of unequal elements is: ", array1.size - count)
            # 计算相等元素所占比例
            ratio = np.mean(np.equal(array1, array2))
            # print(file1 + ": The ratio of equal elements is: ", ratio*100,"%")
            data['int'][file1]['mean_squared_error'] = "{}".format(MSE)
            data['int'][file1]['mean_absolute_error'] = "{}".format(MAE)
            data['int'][file1]['mean_absolute_percentage_error'] = "{}".format(MAPE)
            data['int'][file1]['equal_elements'] = count
            data['int'][file1]['unequal_elements'] = array1.size - count
            data['int'][file1]['ratio_of_equal_elements'] = "{}%".format(ratio*100)
# 遍历两个文件夹中的numpy文件
data['flaot'] = {}
# 遍历两个文件夹中的numpy文件
for file1 in os.listdir(folder3):
    if file1.endswith(".npy"):
        # 读取numpy文件
        array1 = np.load(os.path.join(folder3, file1)).reshape(-1)
        # 查找与file1文件同名的文件
        file2 = os.path.join(folder4, file1)
        if os.path.exists(file2):
            print(file1)
            data['flaot'][file1]={}
            # 读取numpy文件
            array2 = np.load(file2).transpose(0,3,1,2).reshape(-1)
            # 计算绝对值平# 计算绝对值平均误差
            # error = np.mean_absolute_error(array1, array2)
            MSE = metrics.mean_squared_error(array1, array2)
            
            # print(file1 + ": mean_squared_error is: ", MSE)
            MAE = metrics.mean_absolute_error(array1, array2)
            # print(file1 + ": mean_absolute_error is: ", MAE)
            MAPE = metrics.mean_absolute_percentage_error(array1, array2)
            # print(file1 + ": mean_absolute_percentage_error is: ", MAPE)
            # 计算相等元素的个数
            count = np.count_nonzero(np.equal(array1, array2))
            # print(file1 + ": The number of equal elements is: ", count)
            # print(file1 + ": The number of unequal elements is: ", array1.size - count)
            # 计算相等元素所占比例
            ratio = np.mean(np.equal(array1, array2))
            # print(file1 + ": The ratio of equal elements is: ", ratio*100,"%")
            data['flaot'][file1]['mean_squared_error'] = "{}".format(MSE)
            data['flaot'][file1]['mean_absolute_error'] = "{}".format(MAE)
            data['flaot'][file1]['mean_absolute_percentage_error'] = "{}".format(MAPE)
            data['flaot'][file1]['equal_elements'] = count
            data['flaot'][file1]['unequal_elements'] = array1.size - count
            data['flaot'][file1]['ratio_of_equal_elements'] = "{}%".format(ratio*100)
print(data)
json_str = json.dumps(data,sort_keys=False,indent=4)
with open('{}/aimath-iee-compare.json'.format(file_name), 'w') as json_file:
    json_file.write(json_str)