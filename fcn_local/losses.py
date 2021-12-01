"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de
"""

import keras
import keras.backend as K
import numpy as np
from sklearn.metrics import mean_squared_error



def gaussian_nll(ytrue, ypreds):
    """Keras implementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5 * K.sum(K.square((ytrue - mu) / (K.exp(logsigma)+1e-20)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return K.mean(-log_likelihood)

def gaussian_nll_metric(ytrue, ypreds):
    n_dims = int(int(ypreds.shape[1]) / 2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5 * np.sum(np.square((ytrue - mu) / np.exp(logsigma)), axis=1)
    sigma_trace = -np.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return np.mean(-log_likelihood)

def custom_mse(ytrue, ypreds):
    loss = K.square(ytrue -ypreds) # (batch_size,number outputs)
    loss = loss * (1 + ytrue) # (batch_size,number outputs)
    loss = K.sum(loss,axis=1) # (batch_size,)
    return loss
