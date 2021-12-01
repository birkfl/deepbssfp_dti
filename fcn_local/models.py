"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU, LeakyReLU, PReLU, ELU, GaussianNoise
from keras.models import Model, Sequential
from keras import regularizers, initializers
from keras.layers.core import Lambda
from keras import backend as K
import warnings
import logging


def dnn(n_inputs, n_outputs, use_batchnorm=False, n_hidden_layers=3, hlayer_size=128, hidden_activation="relu", dropout_rate=0.0, l2_reg=0.0, probabilistic=True, random_seed=1234, dropout_test=False, gauss_noise=0.0, output_activation='linear'):
    """Defines simple DNN model
    """
    
    if probabilistic and output_activation == "sigmoid":
        print("Warning: sigmoid output layer combined with probabilistic prediction is most likely not a good idea!")
    
    if type(hlayer_size) is not tuple: # same number of neurons for each layer: convert int to tuple
        hlayer_size = (hlayer_size,)
    if len(hlayer_size) != n_hidden_layers: # repeat number of neurons in case it is constant for each layer
        hlayer_size = hlayer_size * n_hidden_layers
    
    x_input = Input(shape=[n_inputs])
    model = Sequential()
    weight_init = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=random_seed)
    model.add(GaussianNoise(gauss_noise)) # default: 0 -> nothing happens...
    for jj in range(0, n_hidden_layers):
        if use_batchnorm: model.add(BatchNormalization())
        if hidden_activation == "selu":
            model.add(Dense(hlayer_size[jj], activation="selu", kernel_initializer=weight_init, kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(Dense(hlayer_size[jj], activation="linear", kernel_initializer=weight_init, kernel_regularizer=regularizers.l2(l2_reg)))
        if hidden_activation == "relu" or "activations.ReLU" in hidden_activation:
            model.add(ReLU())
        if hidden_activation == "lrelu" or "activations.LeakyReLU" in hidden_activation:
            model.add(LeakyReLU())
        if hidden_activation == "elu" or "activations.ELU" in hidden_activation:
            model.add(ELU())
        if hidden_activation == "prelu" or "activations.PReLU" in hidden_activation:
            model.add(PReLU())
        if dropout_rate != 0.0:
            if dropout_test:
                model.add(Lambda(lambda x: K.dropout(x, level=dropout_rate)))
                print('dropout at test time!')
            else:
                model.add(Dropout(dropout_rate))
    if use_batchnorm: model.add(BatchNormalization())
    if probabilistic:
        model.add(Dense(n_outputs * 2, activation=output_activation, kernel_initializer=weight_init))
    else:
        model.add(Dense(n_outputs, activation=output_activation, kernel_initializer=weight_init))

    model = Model(x_input, model(x_input))
    print(model.summary())

    
    return model

def predict_prob(model, x, batch_size=8192):
    """Make predictions given model and 2d data
    """

    ypred = model.predict(x, batch_size=batch_size, verbose=1)
    n_outs = int(ypred.shape[1] / 2)
    mean = ypred[:, 0:n_outs]
    sigma = np.exp(ypred[:, n_outs:])
    return mean, sigma

def predict_nonProb(model, x, batch_size=8192):
    """Make predictions given model and 2d data
     non-probabilistic -> all sigmas are set to zero! (dirty trick to maintain compatibility with other code)
    """

    ypred = model.predict(x, batch_size=batch_size, verbose=1)
    n_outs = int(ypred.shape[1] / 1)
    mean = ypred[:, 0:n_outs]
    sigma = np.zeros(mean.shape)
    return mean, sigma

def logging_setup(name, filepath, filemode="a", format='#%(asctime)s,%(msecs)d %(levelname)s \n%(message)s', dateformat='%a, %d %b %Y %H:%M:%S',level=logging.INFO):
    formatter = logging.Formatter(format, datefmt=dateformat)
    handler = logging.FileHandler(filepath, mode=filemode)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

    # logger = setup_logger('first_logger', 'first_logfile.log')
    # logger.info('This is just info message')
class LRFunction():
    def __init__(self,halve_n=100, halve_at=[100]):
        self.halve_n = halve_n
        self.halve_at = halve_at

    def halve_lr_every_n(self, episode, lr):
        if episode % self.halve_n == 0 and episode > 0:
            print(f"Epoch {episode}: LR went down to {lr/2}")
            return lr/2
        else:
            return lr

    def halve_lr_at_episodes(self, episode, lr):
        if episode in self.halve_at:
            print(f"Epoch {episode}: LR went down to {lr/2}")
            return lr/2
        else:
            return lr

class FinalEpisode(EarlyStopping):
    def __init__(self,
                 logger,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=1,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(FinalEpisode, self).__init__()
        self.logger = logger
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_val_loss_idx = 0
        self.best_val_loss = float("inf")

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            print("FinalEpisodeCallback: current is None")
            return

        if logs["val_loss"] < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            self.best_val_loss_idx = epoch
            message = f"Best validation value: {self.best_val_loss} at epoch {self.best_val_loss_idx +1} (1-indexed)"
            print(message)
            self.logger.info(message)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("")
                print("===== Early Stopping criterion met! =====")
                print(f"Current Val_Loss ({current}) did not exceed best Val_Loss ({self.best}) for {self.patience} epochs.")
                print(f"Early Stopping at epoch {epoch+1}.\n")
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            self.logger.info(f'Epoch {(self.stopped_epoch + 1)}: early stopping with best val_loss being {self.best_val_loss} at {(self.best_val_loss_idx+1)} (1-indexed)\n')

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value