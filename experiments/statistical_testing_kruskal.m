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

function mean_MAE_values = compute_avg_MAE_values(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    MAE_values = zeros(1, num_signals);
    for i = 1:num_signals
        MAE_values(i) = mean(abs(clean_signal - reconstructed_signals{i}));
    end

    mean_MAE_values = MAE_values
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

% yes... 
% - there is a paper : https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-spr.2020.0104#:~:text=For%20base%2Dline%20wander%2C%20and,methods%20for%20composite%20noise%20removal.
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

% Clean Signal
d = load('noisy_samples/af_sig_HR.mat').sig_HR;

% Noise sample
% noise = load('noisy_samples/slices/em_slice_ind.mat');

% Model 1 - Reconstructions
m1_snr_00 = 'reconstructions/model_1/af_comp_snr_00';
m1_snr_05 = 'reconstructions/model_1/af_comp_snr_05';
m1_snr_10 = 'reconstructions/model_1/af_comp_snr_10';
m1_snr_15 = 'reconstructions/model_1/af_comp_snr_15';

% Model 2 - Reconstructions
m2_snr_00 = 'reconstructions/model_2/af_comp_snr_00';
m2_snr_05 = 'reconstructions/model_2/af_comp_snr_05';
m2_snr_10 = 'reconstructions/model_2/af_comp_snr_10';
m2_snr_15 = 'reconstructions/model_2/af_comp_snr_15';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
% Statistical Analysis
m1_snr_00_mae_values = compute_avg_MAE_values(d, m1_y0_list);
m1_snr_05_mae_values = compute_avg_MAE_values(d, m1_y1_list);
m1_snr_10_mae_values = compute_avg_MAE_values(d, m1_y2_list);
m1_snr_15_mae_values = compute_avg_MAE_values(d, m1_y3_list);

m2_snr_00_mae_values = compute_avg_MAE_values(d, m2_y0_list);
m2_snr_05_mae_values = compute_avg_MAE_values(d, m2_y1_list);
m2_snr_10_mae_values = compute_avg_MAE_values(d, m2_y2_list);
m2_snr_15_mae_values = compute_avg_MAE_values(d, m2_y3_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure;

subplot(2, 2, 1);
histogram(m1_snr_00_mae_values);
title('Histogram of MAE for SNR 0');
xlabel('MAE');
ylabel('Frequency');

subplot(2, 2, 2);
histogram(m1_snr_05_mae_values);
title('Histogram of MAE for SNR 5');
xlabel('MAE');
ylabel('Frequency');

subplot(2, 2, 3);
histogram(m1_snr_10_mae_values);
title('Histogram of MAE for SNR 10');
xlabel('MAE');
ylabel('Frequency');

subplot(2, 2, 4);
histogram(m1_snr_15_mae_values);
title('Histogram of MAE for SNR 15');
xlabel('MAE');
ylabel('Frequency');


% Combine all MAE values into one array
m1_all_mae_values = [m1_snr_00_mae_values(:); m1_snr_05_mae_values(:); m1_snr_10_mae_values(:); m1_snr_15_mae_values(:)];
m2_all_mae_values = [m2_snr_00_mae_values(:); m2_snr_05_mae_values(:); m2_snr_10_mae_values(:); m2_snr_15_mae_values(:)];

% Create a grouping variable
group_m1 = [repmat({'0'}, length(m1_snr_00_mae_values), 1); 
         repmat({'5'}, length(m1_snr_05_mae_values), 1); 
         repmat({'10'}, length(m1_snr_10_mae_values), 1); 
         repmat({'15'}, length(m1_snr_15_mae_values), 1)];

% Create a grouping variable
group_m2 = [repmat({'0'}, length(m2_snr_00_mae_values), 1); 
         repmat({'5'}, length(m2_snr_05_mae_values), 1); 
         repmat({'10'}, length(m2_snr_10_mae_values), 1); 
         repmat({'15'}, length(m2_snr_15_mae_values), 1)];

% Perform Kruskal-Wallis test
[p1, tbl1, stats1] = kruskalwallis(m1_all_mae_values, group_m1);
[p2, tbl2, stats2] = kruskalwallis(m2_all_mae_values, group_m2);

% Display the results
disp(['p-value: ', num2str(p1)]);
if p1 < 0.05
    disp('There is a statistically significant difference between the groups.');
else
    disp('There is no statistically significant difference between the groups.');
end

% Display the results
disp(['p-value: ', num2str(p2)]);
if p2 < 0.05
    disp('There is a statistically significant difference between the groups.');
else
    disp('There is no statistically significant difference between the groups.');
end

% Create a figure with two vertically stacked subplots
figure;

% Subplot 1: Model 1
subplot(2,1,1);
boxplot(m1_all_mae_values, group_m1);
title('Composite Noise AF - Model 1');
xlabel('SNR Levels');
ylabel('MAE');
grid on;
% Add p-value annotation to the plot
text(1.5, max(m1_all_mae_values)*0.95, ['p-value: ', num2str(round(p1, 3))], 'FontSize', 12, 'Color', 'red');

% Subplot 2: Model 2
subplot(2,1,2);
boxplot(m2_all_mae_values, group_m2);
title('Composite Noise AF - Model 2');
xlabel('SNR Levels');
ylabel('MAE');
grid on;
% Add p-value annotation to the plot
text(1.5, max(m2_all_mae_values)*0.95, ['p-value: ', num2str(round(p2, 3))], 'FontSize', 12, 'Color', 'red');

