function [pred,sigma,varagout] = applyPythonNN(NNname,inp4d,path0,varargin)
% call NN_matlab_apply.py to generate NN prediction, save results in temp
% folder on hard disk in order to load directly to Matlab workspace
% for non-prob NNs: sigma contains only zeros!
% varargout: structure containing information about MC dropout if used

% No warranty of completeness

% September 2021
% florian.birk@tuebingen.mpg.de 

    temp_folder = 'C:/Temp'; % where temporary input/output mat files are saved
    if ~exist(temp_folder, 'dir')
        mkdir(temp_folder);
    end

    % parse input arguments
    p = inputParser;
    addParameter(p, 'Segment', []);
    addParameter(p, 'mc_dropout', 0);
    parse(p, varargin{:});
    
    % save inputs to mat file
    input_path = fullfile(temp_folder, 'NNinput.mat');
    bSSFP_input = inp4d;
    save(input_path,'bSSFP_input'); % python looks for variable name bSSFP_input
    
    output_path = fullfile(temp_folder, 'NNoutput.mat');
    save(output_path)

    % read needed paths from PATHS.log
    [python_path, base_path] = get_paths_from_log(path0);
    apply_script_path = fullfile(base_path, 'NN_matlab_apply.py');
    
    % prepare python script call
    if p.Results.mc_dropout > 0 % option for mc dropout
        mc_string = sprintf('--mc_dropout %d', p.Results.mc_dropout);
    else
        mc_string = '';
    end
    command = sprintf('%s %s -n %s -i %s -o %s %s',...
                    python_path, apply_script_path, NNname, input_path, output_path, mc_string);
    
    % execute python script call and show output in Matlab console
    [status,cmdout] = system(command) 
    
    % load results to workspace -> passed directly to function output
    load(output_path,'pred','sigma', 'mc_dropout_results'); 
    
    % if Segment is given, apply it
    if ~isempty(p.Results.Segment) 
        pred = SEGMENT_ZSTACK(pred, p.Results.Segment, NaN);
        sigma = SEGMENT_ZSTACK(sigma, p.Results.Segment, NaN);
        try
            mc_dropout_results.mc_std = SEGMENT_ZSTACK(mc_dropout_results.mc_std, p.Results.Segment, NaN);
            
        end
    end
    
    varargout{1} = mc_dropout_results; % only return if requested
end

