import tensorflow as tf
import os

def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
class utils:
    def __init__(self):
        pass
    def print_tensor_shape(self,x):
        print (x.get_shape().as_list())