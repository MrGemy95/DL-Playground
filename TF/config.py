 

class BaseConfig:
    adam_lr=1e-4
    scalar_summary_tags = ['loss','acc']
    checkpoint_dir = './summaries/'
    summary_dir= './summaries/'
    max_to_keep=5
    load=True
class ConvConfig(BaseConfig):
    n_epochs=20
    nit_epoch=1000
    adam_lr=1e-4
    batch_size=32
class NeuralConfig(BaseConfig):
    n_epochs=20
    nit_epoch=1000
    adam_lr=1e-4
    batch_size=32
class LstmConfig(BaseConfig):
    n_steps=28
    state_size=28
    n_epochs=20
    nit_epoch=1000
    adam_lr=1e-4
    batch_size=32
class Experiment1(ConvConfig):
    pass 



def get_config():
    return Experiment1