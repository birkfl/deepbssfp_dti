"""
Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
No warranty of completeness

September 2021
deepCEST: felix.glang@tuebingen.mpg.de
deepbSSFP: florian.birk@tuebingen.mpg.de
"""

import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import sys
from time import time


def load_data_bSSFP(scan_dir,test_split=0.0,test_dataset_nr=0, random_seed=1234):
   
    scan_list = sorted(os.listdir(scan_dir))
    scan_paths = [os.path.join(scan_dir, csv) for csv in scan_list]

    train_scans = scan_paths[0]
    train_data = loadmat(train_scans)
    xtr = train_data['input_data']
    ytr = train_data['target_data']
    test_scan = scan_paths[1]
    test_data = loadmat(test_scan)
    xtest = test_data['input_data']
    ytest = test_data['target_data']
    return xtr, ytr, xtest, ytest