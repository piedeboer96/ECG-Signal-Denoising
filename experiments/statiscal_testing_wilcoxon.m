% noise_removal_functions.m

function aligned_signal = vertical_align_by_mean(clean_signal, noisy_signal)
    % Function to vertically align signals based on mean difference
    mean_clean = mean(clean_signal);
    mean_noisy = mean(noisy_signal);
    mean_diff = mean_clean - mean_noisy;
    aligned_signal = noisy_signal + mean_diff;
end

function aligned_signal = vertical_align_reconstructed(clean_signal, rec_signal)
    % Function to vertically align reconstructed signals based on mean difference
    vertical_shift_rec = mean(clean_signal) - mean(rec_signal);
    aligned_signal = rec_signal + vertical_shift_rec;
    aligned_signal = double(aligned_signal'); % Transpose and convert to double
end

function y_r = load_and_align_signals(path_to_files, clean_signal)
    % Function to load and align signals from a given path
    files = dir(fullfile(path_to_files, '*.mat'));
    y_r = cell(1, numel(files));
    for i = 1:numel(files)
        file_name = fullfile(path_to_files, files(i).name);
        loaded_data = load(file_name);
        y_r{i} = loaded_data.sig_rec;
    end
    for i = 1:numel(y_r)
        y_r{i} = vertical_align_reconstructed(clean_signal, y_r{i});
    end
end

function [mean_values, std_dev_values] = compute_avg_and_std_dev_MAE(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    MAE_values = zeros(1, num_signals);
    for i = 1:num_signals
        MAE_values(i) = mean(abs(clean_signal - reconstructed_signals{i}));
    end
    mean_values = mean(MAE_values)
    std_dev_values = std(MAE_values);
end

function [mean_values, std_dev_values] = compute_avg_and_std_dev_RMSE(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    RMSE_values = zeros(1, num_signals);
    for i = 1:num_signals
        RMSE_values(i) = rmse(reconstructed_signals{i}, clean_signal);
    end
    mean_values = mean(RMSE_values);
    std_dev_values = std(RMSE_values);
end

function [mean_values, std_dev_values] = compute_avg_and_std_dev_PSNR(clean_signal, reconstructed_signals)
    num_signals = length(reconstructed_signals);
    PSNR_values = zeros(1, num_signals);
    for i = 1:num_signals
        PSNR_values(i) = psnr(reconstructed_signals{i}, clean_signal);
    end
    mean_values = mean(PSNR_values);
    std_dev_values = std(PSNR_values);
end

function mae_values = compute_MAE_values(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    MAE_values = zeros(1, num_signals);
    for i = 1:num_signals
        MAE_values(i) = mean(abs(clean_signal - reconstructed_signals{i}));
    end

    mae_values = MAE_values;
end

function rmse_values = compute_RMSE_values(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    RMSE_values = zeros(1, num_signals);
    for i = 1:num_signals
        RMSE_values(i) = rmse(reconstructed_signals{i}, clean_signal);
    end

    rmse_values = RMSE_values;
end

function psnr_values = compute_PSNR_values(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    PSNR_values = zeros(1, num_signals);
    for i = 1:num_signals
        PSNR_values(i) = psnr(reconstructed_signals{i}, clean_signal);
    end

    psnr_values = PSNR_values;
end

function y_LPF = low_pass_filter(x)
    % Function to use low pass filter
    fs = 128; % Sampling rate (Hz)
    fc = 40.60449; % Cutoff frequency (Hz)  - adapted for changed sampling frequency
    N = 14; % Filter length
    phi_d = 0.1047; % Phase delay (rad/Hz)
    fc_norm = fc / fs;
    beta = 0; % Kaiser window parameter
    fir_coeffs = fir1(N-1, fc_norm, 'low', kaiser(N, beta));
    delay = floor((N-1) / 2);
    x = x(:);
    y_LPF = filter(fir_coeffs, 1, [x; zeros(delay, 1)]);
    y_LPF = y_LPF(delay+1:end);
    y_LPF = y_LPF';
end

function y_LMS = lms_filter(x, d)
    % Function to use LMS filter
    x = x(:);
    d = d(:);
    lms = dsp.LMSFilter();
    [y_LMS, ~, ~] = lms(x, d);
    y_LMS = y_LMS';
end

function y_hybrid = hybrid_filter_lms_lpf(x, d)
    % Function to use Hybrid Filter (LMS then LPF)
    y_LMS = lms_filter(x, d);
    y_hybrid = low_pass_filter(y_LMS);
end

function y_hybrid = hybrid_filter_lpf_lms(x, d)
    % Function to use Hybrid Filter (LPF then LMS)
    y_LPF = low_pass_filter(x);
    y_hybrid = lms_filter(y_LPF, d);
end

function y_wt = wavelet_denoise(x)
    % Function to use wavelet denoising
    y_wt = wdenoise(x, 'DenoisingMethod', 'Sure', 'Wavelet', 'sym6', 'ThresholdRule', 'Soft');
end

function y_MA = moving_average_filter(x)
    % Define the coefficients of the transfer function H(z)
    b = 1/4 * ones(1, 4);  % Numerator coefficients (1/4 * ones(1, 4))
    a = 1;                 % Denominator coefficients (1 - z^-1)

    % Apply filter
    y_MA = filter(b, a, x);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
d = load('noisy_samples/ardb_sig_HR.mat').sig_HR;

% Model 1 - Reconstructions
m1_snr_00 = 'reconstructions/model_1/ardb_ma_snr_00';
m1_snr_05 = 'reconstructions/model_1/ardb_ma_snr_05';
m1_snr_10 = 'reconstructions/model_1/ardb_ma_snr_10';
m1_snr_15 = 'reconstructions/model_1/ardb_ma_snr_15';

% Model 2 - Reconstructions
m2_snr_00 = 'reconstructions/model_2/ardb_ma_snr_00';
m2_snr_05 = 'reconstructions/model_2/ardb_ma_snr_05';
m2_snr_10 = 'reconstructions/model_2/ardb_ma_snr_10';
m2_snr_15 = 'reconstructions/model_2/ardb_ma_snr_15';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aligning the reconstructed signals to the clean signal - Model 1
m1_y0_list = load_and_align_signals(m1_snr_00, d);
m1_y1_list = load_and_align_signals(m1_snr_05, d);
m1_y2_list = load_and_align_signals(m1_snr_10, d);
m1_y3_list = load_and_align_signals(m1_snr_15, d);

% Aligning the reconstructed signals to the clean signal - Model 2
m2_y0_list = load_and_align_signals(m2_snr_00, d);
m2_y1_list = load_and_align_signals(m2_snr_05, d);
m2_y2_list = load_and_align_signals(m2_snr_10, d);
m2_y3_list = load_and_align_signals(m2_snr_15, d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Obtain list wih MAE values in order to do statistical analyis
m1_snr_00_mae_values = compute_MAE_values(d, m1_y0_list)
m1_snr_05_mae_values = compute_MAE_values(d, m1_y1_list);
m1_snr_10_mae_values = compute_MAE_values(d, m1_y2_list);
m1_snr_15_mae_values = compute_MAE_values(d, m1_y3_list);

m2_snr_00_mae_values = compute_MAE_values(d, m2_y0_list)
m2_snr_05_mae_values = compute_MAE_values(d, m2_y1_list);
m2_snr_10_mae_values = compute_MAE_values(d, m2_y2_list);
m2_snr_15_mae_values = compute_MAE_values(d, m2_y3_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute AVG mae value at each SNR for model 1 and model 2 ... (and
% st.dev)

[m1_snr_00_AVG_MAE, m1_snr_00_STD_MAE] = compute_avg_and_std_dev_MAE(d, m1_y0_list);
[m1_snr_05_AVG_MAE, m1_snr_05_STD_MAE] = compute_avg_and_std_dev_MAE(d, m1_y1_list);
[m1_snr_10_AVG_MAE, m1_snr_10_STD_MAE] = compute_avg_and_std_dev_MAE(d, m1_y2_list);
[m1_snr_15_AVG_MAE, m1_snr_15_STD_MAE] = compute_avg_and_std_dev_MAE(d, m1_y3_list);

[m2_snr_00_AVG_MAE, m2_snr_00_STD_MAE] = compute_avg_and_std_dev_MAE(d, m2_y0_list);
[m2_snr_05_AVG_MAE, m2_snr_05_STD_MAE] = compute_avg_and_std_dev_MAE(d, m2_y1_list);
[m2_snr_10_AVG_MAE, m2_snr_10_STD_MAE] = compute_avg_and_std_dev_MAE(d, m2_y2_list);
[m2_snr_15_AVG_MAE, m2_snr_15_STD_MAE] = compute_avg_and_std_dev_MAE(d, m2_y3_list);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Statistical Testing 

% Initialize arrays to store p-values
p_values = zeros(4, 1);

% Compare MAE values for Model 1 and Model 2 at different SNR levels
snr_levels = [0, 5, 10, 15];

for i = 1:numel(snr_levels)
    snr = snr_levels(i);
    m1_mae_values = eval(['m1_snr_' num2str(snr, '%02d') '_mae_values'])
    m2_mae_values = eval(['m2_snr_' num2str(snr, '%02d') '_mae_values'])
    p = ranksum(m1_mae_values, m2_mae_values)
    [p_values(i), ~] = ranksum(m1_mae_values, m2_mae_values);
end

% Display the results in a table with p-values formatted to 3 decimals
p_values_formatted = arrayfun(@(x) sprintf('%.3f', x), p_values, 'UniformOutput', false);

% Combine AVG and STD values for Model 1
m1_avg_std_values = [m1_snr_00_AVG_MAE, m1_snr_05_AVG_MAE, m1_snr_10_AVG_MAE, m1_snr_15_AVG_MAE;
                     m1_snr_00_STD_MAE, m1_snr_05_STD_MAE, m1_snr_10_STD_MAE, m1_snr_15_STD_MAE]';

% Combine AVG and STD values for Model 2
m2_avg_std_values = [m2_snr_00_AVG_MAE, m2_snr_05_AVG_MAE, m2_snr_10_AVG_MAE, m2_snr_15_AVG_MAE;
                     m2_snr_00_STD_MAE, m2_snr_05_STD_MAE, m2_snr_10_STD_MAE, m2_snr_15_STD_MAE]';

% Format AVG and STD values to 3 decimals
m1_avg_std_values_formatted = arrayfun(@(x) sprintf('%.3f', x), m1_avg_std_values, 'UniformOutput', false);
m2_avg_std_values_formatted = arrayfun(@(x) sprintf('%.3f', x), m2_avg_std_values, 'UniformOutput', false);

% Combine p-values and AVG/STD values into one table
T_combined = table(snr_levels', p_values_formatted, m1_avg_std_values_formatted(:,1), m1_avg_std_values_formatted(:,2), ...
                   m2_avg_std_values_formatted(:,1), m2_avg_std_values_formatted(:,2), ...
                   'VariableNames', {'SNR', 'P_Value', 'Model1_AVG_MAE', 'Model1_STD_MAE', 'Model2_AVG_MAE', 'Model2_STD_MAE'});

disp(T_combined);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save table to .txt 

% Save the table to a .txt file
% writetable(T_combined, 'results_wilcoxon_ardb_comp_new.txt');