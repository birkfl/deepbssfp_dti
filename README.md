# Disclaimer

**PERMISSION IS HEREBY GRANTED, FREE OF CHARGE, TO ANY PERSON OBTAINING A COPY OF THE PROVIDED SOURCE CODE AND DATA. THE REPOSITORY IS PROVIDED "AS IS", WITHOUT WARRANTY FOR ANY KIND. IN CASE OF CODE USAGE, MODIFICATION, MERGING, PUBLISHING, DISTRIBUTION OR SUBLICENSING, PLEASE CITE THIS PUBLICATION AND CORRESPONDING AUTHORS.**

# Content
- [Introduction](#introduction)
- [Repo structure](#repo-structure)
- [Data acquisition](#data-acquisition)
- [Data access via keeper](#data-access-via-keeper)
- [Python installation](#python-installation)
- [Single NN training](#training-a-single-nn-using-matlab-and-python)
- [Single NN testing](#testing-a-single-nn-on-unseen-data-using-matlab-and-python)

# Introduction
## deepbSSFP

Using neural networks for high-resolution multi-parametric quantification of DTI metrics from phase-cycled bSSFP data. 

The idea to use multiple bSSFP image contrasts and neural networks for quantitative multi-parametric mapping, here relaxometry and field maps, was first presented in: 

* Heule R, Bause J, Pusterla O, Scheffler K. Multi‐parametric artificial neural network fitting of phase‐cycled balanced steady‐state free precession data. Magnetic Resonance in Medicine. https://doi.org/10.1002/mrm.28325

The combination of Matlab and Python code for neural network training with additional uncertainty quantification was initially introduced in the DeepCEST project:

* Glang F, Deshmane A, Prokudin S, et al. DeepCEST 3T: Robust MRI parameter determination and uncertainty quantification with neural networks—application to CEST imaging of the human brain at 3T. Magnetic Resonance in Medicine.  https://doi.org/10.1002/mrm.28117

and further adapted for the proposed deepbSSFP approach:

* Birk F, Glang F, Loktyushin A, Birkl C, Ehses P, Scheffler K, Heule R: High-resolution neural network-driven mapping of multiple diffusion metrics leveraging asymmetries in the balanced SSFP frequency profile. **(Peer-review)**

## Purpose 

The purpose of this project is to use phase-cycled balanced steady-state free precession (bSSFP) data with rich information content about microstructural tissue properties, comprised in asymmetric frequency profiles, to directly estimate multiple diffusion metrics. A probabilistic (see also DeepCEST paper from Glang et al.) multilayer perceptron (MLP) (feedforward neural network (NN)) was used for voxelwise simultaneous parameter quantification. 

# Repo structure

In this repository, code snippets for the NN training of downsampled 3 T data are provided. 

**Folder structure:** 
* data: Train and Testing data set (including input and target data, more information below) 
    > **download separately via keeper upon request. Not provided directly in this repo.** (see below)
* fcn_global_Matlab: Functions needed within Matlab scripts
* fcn_local: Functions needed within Python scripts.

Remaining files should be explained in the following sections. 

# Data acquisition

**Pipeline of the proposed project**: Data acquisition was performed for the same **six subjects** at **3 T** and **9.4 T**. After successful postprocessing of input and output data, the input data were **registered** and **downsampled** to the output data. The data preparation for the NN was performed in Matlab. 

* **Input (paper):** 12-point bSSFP phase-cycled data, measured at 3 T (1.3 mm^3 isotropic) and 9.4 T (0.8 mm^3 isotropic).

* **Target (paper):** Diffusion tensor imaging (DTI) data, measured at 3 T (1.4 x 1.4 x 3 mm^3). Target parameters are the fractional anisotropy (**FA**), mean diffusivity (**MD**), axial diffusivity (**AD**), radial diffusivity (**RD**), azimuthal angle (**phi**), inclination angle (**theta**).

> Please note, that the necessary _MAT-Files_ to prepare the NN data, are not provided in this repository or at keeper (see below). 

# Data access via keeper

Training and testing data can be downloaded via keeper upon request. Up to now, only final input and target arrays for NN training and testing of the downsampled 3 T data are provided, no high-resolution or downsampled 9.4 T.

1. Contact florian.birk@tuebingen.mpg.de to receive the necessary password. 
2. Click on https://keeper.mpdl.mpg.de/d/e75b381cb49641d08838/
3. Type in the password
4. Download "data" folder, with **A_Train.mat** and **B_Test.mat**
5. Copy the data folder to the file location of the cloned/downloaded repository (path of e.g. PATHS.log file).

> The `data` folder content and path is required by the Matlab and Python script to find the appropiate data for this project.

> Data structure prepared for the NN to train, should be a 2D array, each row represents one training sample, each column a feature of a training sample (e.g. array size: 500.000 x 216). Mat files should contain input and target data.

# Python installation:

1. Install Anaconda (https://www.anaconda.com/distribution/). 
2. Import `environment.yml` file from this git repo
    - open Anaconda Prompt (Windows: can be found in start menu)
    - navigate to deepbSSFP repo folder where environment.yml is located
    - In the last row of the `environment.yml`, define the path where to install your environment ("Anaconda installation path\envs\")
    - call: `conda env create -f environment.yml`
    - environment is created and named deepbSSFP

> **Please note that package versions can differ, compared to the proposed versions in the paper, for better compatibility.**

> If there are problems with the creation or installation of the environment, check one/all of the following:
    - Is the newest Version of conda installed (in case you already had Anaconda on your computer). If not please update by inserting `$ conda update conda`.
    - In your text editor (e.g. Visual Studio Code or Notepad++) check if the .yml file is in UTF8 format.
3. Configure PATHS.log (this is needed to make Matlab and Python find each other)
    - first line is the path to python executable related to the deepbSSFP environment
      per default, on Windows this something like `D:\Anaconda\envs\deepbssfp\python.exe`
      or wherever Anaconda installation directory is located.  
    - second line is absolute path of this git repo on the file system

**Most of the Matlab code need the `fcn_global_Matlab` folder added to Matlab path!** 

# Training a single NN using Matlab and Python
For the application of the provided code, it is assumed that Keras and Tensorflow were successfully installed (using Anaconda and the `environment.yml` file) and the `PATHS.log` file were adjusted. 

**Training as well as testing data were shuffled row-wise using the `shuffle_data.m` script.**

1. Check the data MAT-files in `data`:
    - `A_Train.mat`: Training data from four subjects (including input and target data). Each row represents one training sample. 
2. Open `Train_NN.m` in Matlab. This code allows to set main parameters for the NN training. The script can be modified to train other NN configurations. 
3. Run `Train_NN.m`. Starting the NN training, Python scripts will be executed (`NN_train.py`). Multiple operations are performed during NN training. 

> **General remarks:** training datasets are organized in subfolders in `/data/`, each subfolder corresponds to the training data of one particular NN. The subfolders can contain multiple .mat files corresponding to multiple subjects. If there are multiple subject datasets, per default the very first one (according to alphabetical order of file names) is used for NN training and the second as testing data. If you want to use another subject as test dataset, this can be specified (`test_set_index`, see later). If there is only one subject .mat file, this will be used for training as well as testing (still, validation data is a randomly selected hold-out subset of voxels from this subject dataset, so no risk of overfitting anyway).
> Each trained NN is saved in a subfolder in `/logs` and the name of the subfolder is used to identify the NNs. The folder contains
> * `config.log`: NN hyperparameters
> * `evaluation.log`: quality metrics of NN evaluated on test dataset after training
> * `training.log`: info about validation loss progression during training
> * `scaler_input.save`: parameters of input normalization/standardization if used
> * `voxel_dnn_prob.h5`: NN weights giving best validation loss so far (gets constantly overwritten during training, unless the option `--save_each_epoch` for `NN_train.py` is used - in this case, a new .h5 file is saved for each epoch) 

# Testing a single NN on unseen data using Matlab and Python

At the current state, testing the NN on unseen data can only be performed to a certain extent. The `Test_NN.m` script enables the testing of the application of the NN for unseen testing data (`data\B_Test.mat`). 

Masks or whole brain target data are not available for visual comparison but might be available in the future. 

For testing the NN on unseen data run the script `Test_NN.m` with the same `NNname` as defined in the training script (see `NN_train.m`).