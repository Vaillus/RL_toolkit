import os
import json
from functools import reduce
import operator

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

def recursive_get(d, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_from_dict(d, map_tuple):
    return reduce(operator.getitem, map_tuple, d)

def set_in_dict(d, map_tuple, value):
    get_from_dict(d, map_tuple[:-1])[map_tuple[-1]] = value
    
    