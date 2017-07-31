

class BaseConfig:
    adam_lr=1e-4

class ConvConfig():
    n_epochs=5
    nit_epoch=1000

class Experiment1(BaseConfig,ConvConfig):
    pass 



def get_config():
    return Experiment1