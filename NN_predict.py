"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de

FG: NN architecture is read from config.log in NN log folder
weights are loaded from H5 file voxel_dnn_prob.h5
if scalers are needed, they are loaded from scaler_input.save or scaler_target.save

if strange errors appear, check if there is duplictae entry in config.log and if 
yes, delete the superfluous ones until only one explicit entry remains
"""

from fcn_local.models import dnn, predict_prob, predict_nonProb
from keras.optimizers import Adam
from fcn_local.losses import gaussian_nll
# from sklearn.externals import joblib
import joblib
import configparser

import numpy as np
import os



def NN_predict_fb(NNname, inputStack, useGPU=False, mc_dropout=0):
    '''
    input:
        NNname:     name of folder in logs/ containing weights (h5 file) and architecture
                    (config.log) and maybe input scaler
    
        inputStack: input array
    
        useGPU:     flag whether to use GPU for prediction (not recommended)
    
        mc_dropout: number of MC dropout repetitions (default 0: no MC dropout)
    
    return:
        pred:             NN prediction results
        sigma:            uncertainties
        mc_dropout_results: dict containing results of MC forward passes, only if mc_dropout > 0
    '''

    if useGPU:
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
    

    base_path = os.path.dirname(os.path.abspath(__file__)) # maybe dangerous...
    directory = f"{base_path}/logs/{NNname}/"
    #lognames = ["config.log", "training.log", "evaluation.log"]

    print(f"Predicting with {NNname}")
    
    if not os.path.exists(directory):
        print(f"NN does not exist in {directory}!")

    # read network architecture from log file
    config = configparser.ConfigParser(strict=False)
    config.read(f"{directory}config.log")
    input_scaling = int(config.get("config", "Input scaling"))
    target_scaling = int(config.get("config","Target scaling"))
    # use_standardization_targets = config.getboolean("config", "Target standardization")
    dropout = float(config.get("config", "Dropout"))
    regularization = float(config.get("config", "L2-Regularization"))
    hidden_layer_count = int(config.get("config", "Number of hidden layers"))
    
    try: # in case number of neurons per layer is not constant
        hidden_layer_size = tuple(map(int, config.get("config", "Hidden layer sizes").split(",")))
    except:
        hidden_layer_size = int(config.get("config", "Hidden layer sizes"))
    
    activation = config.get("config", "Activations")
    #batch_size = int(config.get("config", "Batch size"))
    learning_rate = float(config.get("config", "Learning rate"))
    #patience = int(config.get("config", "Patience"))
    #early_stopping_delta = float(config.get("config", "Early stopping"))
    #validation_split_percentage = float(config.get("config", "Validation split percentage"))
    random_seed = int(config.get("config", "Random seed"))
    
    # being careful with some of the params, because they were added in different
    # versions to the config.log, and still it should be backward compatible
    try:
        output_activation = str(config.get("config", "output activation"))
    except:
        output_activation = 'linear'
        print('could not find output activation in config.log, assuming linear')
    #scan_folder = config.get("config", "Folder")
    #test_set = float(config.get("config", "Test set"))
    #lr_schedule = config.get("config", "LR schedule")
    try:
        nonprob = config.getboolean("config", "non-probabilistic")
    except:
        nonprob = False
        print("Warning: no info found whether NN is probabilistic, assuming no")
    try:
        n_outputs = config.get("config", "number outputs")
    except:
        n_outputs = 6 # DANGEROUS! just guessing
        print("Number of outputs not determined in config.log!")

    if mc_dropout > 0 and dropout == 0:
        print('Warning: MC dropout is applied to a NN that was not trained with dropout!')

    n_inputs = inputStack.shape[1] 

    ckpt_path = f"{directory}voxel_dnn_prob.h5"

    #np.random.seed(opts.random_seed) # determinism not desirable in case of MC dropout


    ### Model creation
    if nonprob:
        lossFcn = 'mean_squared_error'
        predFcn = predict_nonProb
    else:
        lossFcn = gaussian_nll
        predFcn = predict_prob
    
    model = dnn(int(n_inputs), int(n_outputs),
                use_batchnorm=False,
                n_hidden_layers=hidden_layer_count, 
                hlayer_size=hidden_layer_size, 
                hidden_activation=activation,
                dropout_rate=dropout,
                dropout_test=(mc_dropout > 0),
                l2_reg=regularization,
                probabilistic=not nonprob,
                random_seed=random_seed,
                output_activation=output_activation)
    model.compile(loss=lossFcn, optimizer=Adam(lr=learning_rate))

    ### Makes sure the best weights are used instead of the final ones
    model.load_weights(ckpt_path)

    
    # preprocess input data
    input_test = inputStack
    if input_scaling != 0:
        input_scaler_filename = f"{directory}scaler_input.save"
        scaler_input = joblib.load(input_scaler_filename)
        shape = np.shape(input_test)
        reshaped = np.reshape(input_test,(-1,shape[-1]))
        scaler_input.transform(reshaped)
        input_test = np.reshape(reshaped, shape)

    if mc_dropout > 0: # use MC dropout
        predstack = np.zeros(np.shape(input_test)[0:3] + (int(n_outputs), mc_dropout))
        sigmastack = np.zeros(np.shape(input_test)[0:3] + (int(n_outputs), mc_dropout))
        for jj in range(mc_dropout):
            predstack[:,:,:,:,jj], sigmastack[:,:,:,:,jj] = predFcn(model, input_test)
        
        mc_dropout_results = {'preds': predstack, 'sigmas': sigmastack, 'mc_std': np.std(predstack, axis=4)}
        
        return np.mean(predstack,axis=4), np.mean(sigmastack,axis=4), mc_dropout_results    
        
    else: # no MC dropout
        # network prediction happens here
        pred, sigma = predFcn(model, input_test)
       
        # if target scaling is active, apply it
        if target_scaling != 0:
            target_scaler_filename = f"{directory}scaler_target.save"
            scaler_target = joblib.load(target_scaler_filename)
            print('using scaler for inverse target trafo') # Scaler is loaded above
            shape = np.shape(pred)
            shape_sigma = np.shape(sigma) # Should be the same as pred
            reshaped = np.reshape(pred,(-1,shape[-1]))
            reshaped_sigma = np.reshape(sigma,(-1,shape[-1]))
            scaler_target.inverse_transform(reshaped)
            scaler_target.inverse_transform(reshaped_sigma)
            pred = np.reshape(reshaped, shape)
            sigma = np.reshape(reshaped_sigma,shape_sigma)
    
        
        
        return pred, sigma, {}
 
    
 