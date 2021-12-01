%% Example of how to train NN 
% Modified testing script for simplification
% Code is adopted from the deepCEST project: DOI: 10.1002/mrm.28117 and modified for deepbSSFP.
% No warranty of completeness

% September 2021
% deepCEST: felix.glang@tuebingen.mpg.de
% deepbSSFP: florian.birk@tuebingen.mpg.de
%--------------------------------------------
% Input: 12-phase-cycle bSSFP data 
% Input features result from:
% 1. 12 phase cycles
% 2. 9 voxels from axial nearest neighbor voxels
% 3. 2 (real & imaginary) components of the complex bSSFP data
% == 216 features: 12*9*2
%--------------------------------------------
%Target: 6 Diffusion parameters:
% 1. Column: Mean diffusivity (MD) in range 0 - 2*10^-3 (mm^2/s)
% 2. Column: Fractional anisotropy (FA) in range 0-1 (no unit)
% 3. Column: Axial diffusivity (AD) in range 0 - 2*10^-3 (mm^2/s) 
% 4. Column: Radal diffusivity (RD) in range 0 - 2*10^-3 (mm^2/s)
% 5. Column: Azimuthal angle (Azi) in range 0-90 (degree)
% 6. Column: Inclination angle (Inc) in range 0-90 (degree)
%--------------------------------------------

clc
clear all
%% Matlab wrapper to start NN training in python/keras
% generates and executes command line call for NN_train.py
path0 = pwd;

%% network configuration
NNname = 'NN_bSSFP_test'; % this will be the name of the subfolder in /logs
    
N_layers = 2;
N_neurons_per_layer = [20,20];
activation = 'relu'; % implemented: elu, relu, lrelu, selu, prelu
output_activation = 'linear'; % last layer: linear or sigmoid
probabilistic = 1; % 1: with uncertainties (max likelihood training), 0: standard NN without uncertainties

scaling_input = 1; % do standardization transform of inputs (X-mean(X))./std(X) internally in python? usually recommended!
scaling_tar = 2;
% custom_mse = 0; %Using customized mse loss function weighting loss = mse*(1+target)

L2reg = 0.0;
dropout = 0.0; % fraction of randomly disabled neurons
gauss_noise = 0.0; % if > 0, gaussian noise of this std will be added to all training inputs

nr_epochs = 2; % weights with best validation loss after this number of epochs are used
batch_size = 128;
learning_rate = 1e-4; % default: 1e-4

useGPU = 0; 

% if there is only one file use test_set_index=0, then the same will be
% used for training and testing 
test_set_index = 1; % e.g. 1: second file from the training_folder according to alphabetical order
bSSFP = 1; % 1: Use -mat file from bSSFP project, 0: Use Cest Training data

%% select training data folder
data_folder = '\data';
full_data_folder = strcat(path0,data_folder);

%% generate command line call for training python script, training not yet started
cd(path0)
[python_path, base_path] = get_paths_from_log(pwd); % read from PATHS.log
train_script_path = fullfile(base_path, 'NN_train.py');
cd(path0)

% put all parameters together to call NN_train.py
if probabilistic; probStr = ''; else; probStr = '--nonprob'; end
if useGPU; GPUstr = '--GPU'; else; GPUstr = ''; end
% if standardize_input; stdString = ''; else; stdString = '--no_std'; end

if length(N_neurons_per_layer) > 1 % convert array of neurons per layer to string
    neurons_per_layer_str = sprintf('%d,', N_neurons_per_layer);
    neurons_per_layer_str = neurons_per_layer_str(1:end-1); % remove last comma
else
    neurons_per_layer_str = N_neurons_per_layer;
end

% if custom_mse; lossStr = ''; else; lossStr = '--nocustom'; end
if scaling_input == 1; input_scale = sprintf('%s %d','--ssi ',scaling_input); elseif scaling_input == 2; input_scale = sprintf('%s %d','--nsi ',scaling_input); end
if scaling_tar == 1; tar_scale = sprintf('%s %d','--sst ',scaling_tar); elseif scaling_tar == 2; tar_scale = sprintf('%s %d','--nst ',scaling_tar); end

command = sprintf(['%s %s -n %s --hidden_count %d --hidden_size %s -a %s ' ...
                    '-r %f -d %f --folder %s -t %d --nr_epochs %d ' ...
                    '-b %d --learning_rate %f -g %d %s %s --output_activation %s '...
                    '--bSSFP %d %s %s'],... % --custom %d --nt %d '],...
                    python_path, train_script_path, NNname, N_layers,...
                    neurons_per_layer_str, activation, L2reg, dropout,...
                    data_folder, test_set_index, nr_epochs, batch_size,...
                    learning_rate, gauss_noise, probStr, GPUstr,...
                    output_activation,bSSFP,input_scale,tar_scale)
                


%% write command to text file in logs folder
logDir = fullfile(base_path, 'logs', NNname);
if ~isfolder(logDir); mkdir(logDir); end
FP = fopen(fullfile(logDir, 'command.txt'), 'w');
fprintf(FP, '%s', command);
fclose(FP);


%% Start training
system([command ' &']); % append & to force opening console window

%% start tensorboard for monitoring the training process
review_tb = strcat(base_path,'\logs\',NNname);
% opens another console window, where (in the best case) an URL is
% displayed that can be viewed with the browser
tb_command = sprintf('%s/Scripts/tensorboard.exe --logdir=%s --host localhost --port 8093', fileparts(python_path), review_tb);
% tb_command = sprintf('%s/Scripts/tensorboard.exe --logdir=%s/logs --host localhost --port 8092', fileparts(python_path), base_path);
system([tb_command ' &']);