"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de

This script is used by applyPythonNN.m to apply NNs directly from Matlab

command line arguments:
    -n      : Network name (name of log folder)
    -i      : path to input mat file (variable name of input stack is hard-coded to Z_uncorr)
    -o      : path to where results are saved
    --mc_dropout: number of MC dropout repetitions, default: 0 -> no mc_dropout
    --GPU   : whether to use GPU for prediction (not recommended, slower on my machine (even with crazy external graphics card...))
"""

from scipy.io import savemat, loadmat
from NN_predict import NN_predict_fb
import optparse

# Command line options for running
optParser = optparse.OptionParser()
optParser.add_option("-n","--name",action="store",type="string",dest="name",default="default_name")
optParser.add_option("-i","--input",action="store",dest="mat_in", default=None)
optParser.add_option("-o","--output",action="store",dest="mat_out", default=None)
optParser.add_option("--mc_dropout",action="store",type=int,dest="mc_dropout", default=0)
optParser.add_option("--GPU",action="store_true",dest="useGPU", default=False)
opts, _ = optParser.parse_args()

print(f"Predicting with {opts.name}")

### load inputs from matlab file
if opts.mat_in is not None:
    loadedMat = loadmat(opts.mat_in)
    bSSFP_input_test = loadedMat['bSSFP_input'] 
    
    # do prediction
    pred, sigma, mc_dropout_results = NN_predict_fb(opts.name, bSSFP_input_test, opts.useGPU, opts.mc_dropout)
    
    # save result to matlab file 
    if opts.mat_out is not None: 
        matpath = opts.mat_out
    else:
        print('No output path specified!')
    savemat(matpath, {'pred': pred, 'sigma': sigma, 'mc_dropout_results': mc_dropout_results})
    print('matlab output saved to', matpath)