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
    @staticmethod
    def print_tensor_shape(x):
        print (x.get_shape().as_list())