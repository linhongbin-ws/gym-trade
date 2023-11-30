import os
import numpy as np



def get_csv_dir(root_dir:str):
    """get csv file names"""
    csv_list = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if file.endswith(".csv"):
                file_name = os.path.join(root, file)
                csv_list.append(file_name)
    return csv_list

def scale_arr(_input, old_min,old_max,new_min,new_max):
    _in = _input
    _in = np.divide(_input-old_min,old_max-old_min)
    _in = np.multiply(_in,new_max-new_min) + new_min
    return _in

def scale(input, old_min, old_max, new_min, new_max):
        out = (input-old_min)/(old_max-old_min)*(new_max-new_min) + new_min

