import os
import json
from functools import reduce
import operator
import random
import numpy as np
import torch

def get_params(file_name):
    complete_path = make_full_params_path(file_name)
    # load params
    with open(complete_path) as json_file:
        params = json.load(json_file)
    return params

def set_params(file_name, params):
    complete_path = make_full_params_path(file_name)
    with open(complete_path, 'w') as json_file:
        json.dump(params, json_file)

def make_full_params_path(file_name):
    # creating full path string
    dir_name = os.path.dirname(os.path.realpath(__file__))
    file_name, extension = os.path.splitext(file_name)
    if extension == '':
        extension = ".json"
    file_name += extension
    file_name = os.path.join("params", file_name)
    complete_path = os.path.join(dir_name, file_name)

    return complete_path

def get_path(string_path, add_absolute=False):
    """ fromats the path to a format that is correct to python. Can also add its absolute path prefix.

    Returns:
        string: absolute path that is correct to python
    """
    modified_string_path = ""
    if add_absolute:
        modified_string_path = os.path.abspath(os.path.join(os.sep, *string_path.split("/")))
    else:
        modified_string_path = os.path.join(*string_path.split("/"))
    return modified_string_path

def set_random_seed(seed):
    """set all the seeds of the libraries that are susceptible of doing
    random stuff

    Args:
        seed (int)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_from_dict(d, map_tuple):
    """get the value of a specific key in a dictionary

    Args:
        d (dict): dictionary where we access the data
        map_tuple (tuple): the keys that lead to the wanted key

    Returns:
        [type]: [description]
    """
    return reduce(operator.getitem, map_tuple, d)

def set_in_dict(d, map_tuple, value):
    """set the value of a specific key in the dictionary to the specified value

    Args:
        d (dict): dictionary to modify
        map_tuple (tuple): the keys that lead to the wanted key
        value: new value
    """
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value
    
def get_attr(o, map_tuple):
    return reduce(getattr, map_tuple, o)
    