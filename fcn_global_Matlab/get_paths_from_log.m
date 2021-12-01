function [python_path, train_script_path] = get_paths_from_log(path)
    % No warranty of completeness

    % September 2021
    % florian.birk@tuebingen.mpg.de

    cd(path)
    fid = fopen('PATHS.log');
    txt = textscan(fid,'%s','delimiter','\n');
    python_path = txt{1}{1};
    train_script_path = txt{1}{2};
end