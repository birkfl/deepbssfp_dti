"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de
"""

from fcn_local.data_3T_bSSFP import load_data_bSSFP
from fcn_local.models import dnn, predict_prob, predict_nonProb, FinalEpisode, LRFunction, logging_setup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fcn_local.losses import gaussian_nll, gaussian_nll_metric, custom_mse
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, TensorBoard, EarlyStopping, LearningRateScheduler
from scipy.io import savemat, loadmat
import joblib # from sklearn.externals import joblib
import configparser
import matplotlib.pyplot as plt
import logging
import optparse
import numpy as np
import os
import sys

# Command line options for running
optParser = optparse.OptionParser()
optParser.add_option("--hidden_size",action="store", type="string", dest="hidden_layer_size", default=300)
optParser.add_option("--hidden_count",action="store", type="int", dest="hidden_layer_count", default=3)
optParser.add_option("-t","--test_set",action="store", type="float", dest="test_set", default=0)
optParser.add_option("-n","--name",action="store",type="string",dest="name",default="default_name")
optParser.add_option("--output_activation",action="store",type="string",dest="output_activation",default="linear") # optional: sigmoid
optParser.add_option("-a","--activation",action="store",type="string",dest="activation",default="relu")
optParser.add_option("-e","--nr_epochs",action="store",type="int",dest="epochs", default=5000)
optParser.add_option("-b","--batch_size",action="store",type="int",dest="batch_size", default=64)
optParser.add_option("-r","--regularization",action="store",type="float",dest="regularization", default=0.0)
optParser.add_option("-d","--dropout",action="store",type="float",dest="dropout", default=0.0)
optParser.add_option("-g", "--gauss_noise",action="store",type="float",dest="gauss_noise", default=0.0)
optParser.add_option("--standardize",action="store_false",dest="dummy") # dummy for maintaining compatibility with older version
optParser.add_option("--no_fig_save",action="store_false",dest="fig_save", default=True)
optParser.add_option("--batchnorm", action="store_true",dest="batch_norm", default=False)
optParser.add_option("--folder", action="store",type="string",dest="scan_folder", default="TF_ready")
optParser.add_option("-l","--lr_schedule", action="store", type="int", dest="lr_schedule", default=None)
optParser.add_option("--learning_rate", action="store", type="float", dest="learning_rate", default=1e-4)
optParser.add_option("--random_seed", action="store", type="int", dest="random_seed", default=1234)
optParser.add_option("--GPU",action="store_true",dest="useGPU", default=False)
optParser.add_option("--nonprob",action="store_true",dest="nonprob", default=False)
optParser.add_option("-i","--input",action="store",dest="mat_in", default=None)
optParser.add_option("-o","--output",action="store",dest="mat_out", default=None)
optParser.add_option("--save_each_epoch",action="store_true",dest="save_each_epoch", default=False)
optParser.add_option("--bSSFP", action="store_true", dest="bSSFP", default=True)
optParser.add_option("--custom",action="store_true",dest="custom_mse", default=False)
optParser.add_option("--nosi",action="store",type="int",dest="input_scaling", default=0) # input_scaling = 0 (default): No scaling for input data
optParser.add_option("--ssi",action="store",type="int",dest="input_scaling") # input_scaling = 1: Standardize input data
optParser.add_option("--nsi",action="store",type="int",dest="input_scaling") # input_scaling = 2: Normalize input data
optParser.add_option("--nost",action="store",type="int",dest="target_scaling", default=0) # target_scaling = 0 (default): No scaling for target data
optParser.add_option("--sst",action="store",type="int",dest="target_scaling") # target_scaling = 1: Standardize target data
optParser.add_option("--nst",action="store",type="int",dest="target_scaling") # target_scaling = 2: Normalize target data
opts, _ = optParser.parse_args()

# Paths and additional parameters that can be set manually
early_stopping_delta = 0.0
patience = 100
validation_split_percentage = 0.2
learning_rate = opts.learning_rate
base_path = os.path.dirname(os.path.abspath(__file__))

print("base_path=", base_path)

directory = f"{base_path}/logs/{opts.name}/"
lognames = ["config.log", "training.log", "evaluation.log"]

print(f"Training {opts.name}")

if not os.path.exists(directory):
    os.makedirs(directory)

if opts.useGPU:
    # needed for running on GPU
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran) (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   # Use CPU as GPU is slower for tasks with little complexity?



scan_dir =  f'{base_path}/{opts.scan_folder}/'
ckpt_path = f"{directory}voxel_dnn_prob.h5"

np.random.seed(opts.random_seed)

if opts.bSSFP:
    xtr, ytr, xtest, ytest = load_data_bSSFP(scan_dir,test_dataset_nr=int(opts.test_set), random_seed=opts.random_seed)
    # xtr, ytr, xtest, ytest = load_data_bSSFP(scan_dir,test_dataset_nr=int(opts.test_set), random_seed=opts.random_seed)
else:    
    ### Loading and splitting data
    if opts.test_set % 1 == 0: # opts.test_set is integer number -> use entire dataset as test set
        xtr, ytr, x_test_2d, y_test_2d, xtrs4d, ytrs4d, x_test_4d, y_test_4d = load_data(scan_dir, test_dataset_nr=int(opts.test_set), random_seed=opts.random_seed)
    else: # opts.test_set is float -> use randomly hold-out voxels of all datasets as test set
        xtr, ytr, x_test_2d, y_test_2d, xtrs4d, ytrs4d, x_test_4d, y_test_4d, train_idxs, test_idxs = load_data(scan_dir, test_split=opts.test_set, random_seed=opts.random_seed)


# calculate input scaler and save
# parameters are saved in scaler_input.save in log dir to be used later for test data
if opts.input_scaling == 1:
    input_scaler_filename = f"{directory}scaler_input.save"
    scaler_input = StandardScaler(copy=False)
    scaler_input.fit_transform(xtr)
    scaler_input.transform(xtest) 

    mean_xtr = np.mean(xtr,axis=0)
    length_meanCx = len(mean_xtr); print("Number of means for xtr:",length_meanCx)
    print("xtr standardized:",mean_xtr)
    
    joblib.dump(scaler_input, input_scaler_filename) 
elif opts.input_scaling == 2:
    input_scaler_filename = f"{directory}scaler_input.save"
    scaler_input = MinMaxScaler(feature_range=(0,1), copy=False)
    scaler_input.fit_transform(xtr)
    scaler_input.transform(xtest)
    joblib.dump(scaler_input, input_scaler_filename) 

    max_xtr = np.max(xtr)
    min_xtr = np.min(xtr)
    minmax_str = "After Scaling:\nInput Min: %d\nInput Max: %d\n" % (min_xtr,max_xtr)
    print(minmax_str)

# calculate target scaler and save
if opts.target_scaling == 1:
    target_scaler_filename = f"{directory}scaler_target.save"
    scaler_target = StandardScaler(copy=False)
    scaler_target.fit_transform(ytr)
    scaler_target.transform(ytest) 

    MD_mean = np.mean(ytr[:,0]); print("Mean of MD:",MD_mean)
    FA_mean = np.mean(ytr[:,1]); print("Mean of FA:",FA_mean)
    Azi_mean = np.mean(ytr[:,2]); print("Mean of Azi:",Azi_mean)
    Inc_mean = np.mean(ytr[:,3]); print("Mean of Inc:",Inc_mean)
    joblib.dump(scaler_target, target_scaler_filename)
elif opts.target_scaling == 2:
    target_scaler_filename = f"{directory}scaler_target.save"
    scaler_target = MinMaxScaler(feature_range=(0,1), copy=False)
    scaler_target.fit_transform(ytr)
    scaler_target.transform(ytest)
    joblib.dump(scaler_target, target_scaler_filename)

    max_ytr = np.max(ytr)
    min_ytr = np.min(ytr)
    minmax_Ystr = "After Scaling:\nTarget Min: %d\nTarget Max: %d\n" % (min_ytr,max_ytr)
    print(minmax_Ystr)

xtr, xval, ytr, yval = train_test_split(xtr, ytr, test_size=validation_split_percentage, random_state=opts.random_seed)
n_inputs = xtr.shape[1]
n_outputs = ytr.shape[1]

### Logging setup
config_logger = logging_setup(lognames[0],f"{directory}{lognames[0]}")
train_logger  = logging_setup(lognames[1],f"{directory}{lognames[1]}")
eval_logger   = logging_setup(lognames[2],f"{directory}{lognames[2]}")

config_logger.info(f"[{opts.name}]\n")
train_logger.info(f"{opts.name}\n")
eval_logger.info(f"{opts.name}\n")
config_logger.info(f"[config]\n"
    # f"Normalize: {False}\n"
    f"Input scaling: {opts.input_scaling}\n"
    f"Target scaling: {opts.target_scaling}\n"
    f"Dropout: {opts.dropout}\n"
    f"L2-Regularization: {opts.regularization}\n"
    f"Number of hidden layers: {opts.hidden_layer_count}\n"
    f"Hidden layer sizes: {opts.hidden_layer_size}\n"
    f"Activations: {opts.activation}\n"
    f"Batch size: {opts.batch_size}\n"
    f"Learning rate: {learning_rate}\n"
    f"Patience: {patience}\n"
    f"Early stopping: {early_stopping_delta}\n"
    f"Validation split percentage: {validation_split_percentage}\n"
    f"Random seed: {opts.random_seed}\n"
    f"Folder: {opts.scan_folder}\n"
    f"non-probabilistic: {opts.nonprob}\n"
    f"Test set: {opts.test_set}\n"
    f"LR schedule: {opts.lr_schedule}\n"
    f"Gaussian noise: {opts.gauss_noise}\n"
    f"number outputs: {n_outputs}\n"
    f"output activation: {opts.output_activation}\n")

### Model creation

# switch loss function for probabilistic / non-prob. NN
# non-prob: MSE, prob: log-likelihood with sigma as free param
if opts.custom_mse:
    lossFcn = custom_mse
else:
    if opts.nonprob:
        lossFcn = 'mean_squared_error'
    else:
        lossFcn = gaussian_nll


hidden_layers_tup = tuple(map(int, opts.hidden_layer_size.split(",")))

model = dnn(n_inputs, n_outputs,
            use_batchnorm=opts.batch_norm,
            n_hidden_layers=opts.hidden_layer_count, 
            hlayer_size=hidden_layers_tup, 
            hidden_activation=opts.activation,
            dropout_rate=opts.dropout,
            l2_reg=opts.regularization,
            probabilistic=not opts.nonprob,
            random_seed=opts.random_seed,
            gauss_noise=opts.gauss_noise,
            output_activation = opts.output_activation
            )
model.compile(loss=lossFcn, optimizer=Adam(lr=learning_rate))

### Callbacks for loading, training visualization and training end when the performance stops improving
if os.path.exists(ckpt_path):
    print(f"Existing weights found {ckpt_path}. Loading and continuing training.")
    model.load_weights(ckpt_path)

callback_list = []
if opts.save_each_epoch:
    ckpt_path = f"{directory}" + "weights.{epoch:04d}.hdf5"
    print("save weights to", ckpt_path)
    mccb = ModelCheckpoint(ckpt_path, save_best_only=False, save_weights_only=True)
else:
    mccb = ModelCheckpoint(ckpt_path, save_best_only=True, save_weights_only=True)

callback_list.append(mccb)
tb = TensorBoard(log_dir=directory, profile_batch = 0, histogram_freq=0, 
            #batch_size=opts.batch_size, 
            write_graph=True, 
            write_grads=False, 
            write_images=False)
callback_list.append(tb)
es = FinalEpisode(
            monitor="val_loss",
            logger=train_logger,
            min_delta=early_stopping_delta,
            patience=patience,
            verbose=1,
            mode="min")
callback_list.append(es)
if opts.lr_schedule is not None:
    lrf = LRFunction(opts.lr_schedule)
    lr_cb = LearningRateScheduler(lrf.halve_lr_every_n)
    callback_list.append(lr_cb)

### Training procedure
model.fit(xtr, ytr, 
        validation_data=(xval, yval), 
        epochs=opts.epochs, 
        batch_size=opts.batch_size, 
        shuffle=True, 
        callbacks=callback_list)

### Makes sure the best weights are used instead of the final ones
model.load_weights(ckpt_path)

### save also full model as h5
savemodelpath = f'{directory}/fullmodel.h5'
model.save(savemodelpath)
print("model saved to", savemodelpath)

### Predict outputs data (MD, FA, Azimuth, Elevation)
ytest_mean, ytest_sigma = predict_prob(model,xtest) 

# ### Calculation of different metrics
mae = np.mean(np.abs(ytest-ytest_mean))
mse = np.mean((ytest - ytest_mean)**2)
median_ae = np.median(np.abs(ytest-ytest_mean))
ae = np.abs(ytest-ytest_mean)
gll = gaussian_nll_metric(ytest, np.hstack((ytest_mean, ytest_sigma)))

### Save metrics into log file
eval_logger.info(f"MAE: {mae}\nMSE: {mse}\nMedianAE: {median_ae}\nAE: {ae}\nGNLL: {gll}\n")
