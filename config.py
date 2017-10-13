 

class BaseConfig:
    lr=1e-4
    checkpoint_dir = './summaries/'
    summary_dir= './summaries/'
    max_to_keep=5
    max_images_to_visualize=5
    summary_image_shape=[28,56,1]
    load=True

class NeuralConfig(BaseConfig):
    n_epochs=20
    nit_epoch=1000
    lr=1e-4
    batch_size=32
    state_size=[784]

class ConvConfig(BaseConfig):
    n_epochs=20
    nit_epoch=1000
    lr=1e-4
    batch_size=32
    state_size=[28,28,1]

class LstmConfig(BaseConfig):
    n_steps=28
    state_size=28
    n_epochs=20
    nit_epoch=1000
    lr=1e-4
    batch_size=32

class AutoEncoderConfig(BaseConfig):
    state_size=[96,96,2]
    labels_size=[96,96]
    n_epochs=20
    nit_epoch=100
    lr=.01
    batch_size=32
    num_test=5
    train_ratio=.9
    states_path='./data/states.npy'
    dropout_rate=.5