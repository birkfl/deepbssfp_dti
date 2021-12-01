%% Shuffling of training and testing data
% No warranty of completeness

% September 2021
% florian.birk@tuebingen.mpg.de


% Shuffle input and training data randomly
cd('data')
disp('Loading Train file and shuffle....')
load A_Train.mat
random_int = randperm(size(input_data,1));
input_data = input_data(random_int,:);
target_data = target_data(random_int,:);

save A_Train.mat input_data target_data
clear all

disp('Loading Test file and shuffle....')
load B_Test.mat
random_int = randperm(size(input_data,1));
input_data = input_data(random_int,:);
target_data = target_data(random_int,:);
save B_Test.mat input_data target_data
clear all