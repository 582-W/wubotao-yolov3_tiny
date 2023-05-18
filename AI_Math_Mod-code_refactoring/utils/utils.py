import os
import argparse
import shutil

def read_directory(directory_name):
    file_path = []
    for filename in os.listdir(directory_name):
        filepath = directory_name + "/" + filename
        file_path.append(filepath)
    
    return file_path

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_folder(folder_path, whether_delete=True):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if whether_delete:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)

def check_path(file_path):
    assert os.path.exists(file_path), "{} doesn't exist".format(file_path)


def convert2float64(input_dict):
    for k, v in input_dict.items():
        if isinstance(v,dict):
            convert2float64(v)
        else:
            if isinstance(v,list):
                input_dict[k] = [float(item) for item in v]
            elif isinstance(v,int):
                pass
            else:
                input_dict[k] = float(v)

