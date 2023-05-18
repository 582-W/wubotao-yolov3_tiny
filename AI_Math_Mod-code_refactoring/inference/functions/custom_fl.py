import onnx
import numpy as np

def custom_node_fl(x: list, node_name: str) -> list:
    """custom_node float function 
    Args:
        x (list): input from last layer
        node_name (str) : the custom node name
    Returns:
        list: output 
    """
    node_name_fun_mapping = {
        "identity": custom_fun_1,
        "inputmeannormalization":custom_fun_2,
        "inputchannelswap":custom_fun_3
    }

    if node_name in node_name_fun_mapping.keys():
        custom_fun = node_name_fun_mapping[node_name]
    else:   
        raise ValueError("The Custom_node {}'s function must be defined first!".format(node_name))
    
    return custom_fun(x)

def custom_fun_1(x):
    '''
    custom node function
    Args:
        x (list): input from last layer
    '''
   
    return x

def custom_fun_2(x):
    '''
    custom node function
    Args:
        x (list): input from last layer
    '''
    # local variable
    subtract_mean=[123, 117, 104]
    return x - np.array(subtract_mean)

def custom_fun_3(x):
    '''
    custom node function
    Args:
        x (list): input from last layer
    '''
    return x
