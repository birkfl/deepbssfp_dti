%% Example of how to apply trained NN to unseen dataset 
% Modified testing script for simplification
% Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
% No warranty of completeness

% September 2021
% deepCEST: felix.glang@tuebingen.mpg.de
% deepbSSFP: florian.birk@tuebingen.mpg.de

clc
clear all
path0 = uigetdir('','Select base directory (PATHS.log location)');
NNname = 'NN_bSSFP_test';

%% Select testing .mat file
path_data = strcat(path0,'\data\');
% Load test data
cd(path_data)
load B_Test 

%% NN prediction
[pred, sigma] = applyPythonNN(NNname,input_data,path0);

%% Create folders if not available
path_result = strcat(path0,'\results');
if ~isfolder(path_result); mkdir(path_result); end
path_results_folder = strcat(path_result,'\',NNname);
if ~isfolder(path_results_folder); mkdir(path_results_folder); end
cd(path_results_folder)

%% Save pred and target as 3D matrix
cd(path_results_folder)
save Test_pred.mat pred sigma target_data input_data
