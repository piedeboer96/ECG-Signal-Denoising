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
d = load('noisy_samples/ardb_sig_HR.mat').sig_HR;

% Noise sample
noise = load('noisy_samples/slices/em_slice_ind.mat');

% Noisy Signals at SNR 0, 5, 10 and 15
x0 = load('noisy_samples/samples/ardb_sig_SR_em_snr_00.mat').ardb_sig_SR_em_snr_00;
x1 = load('noisy_samples/samples/ardb_sig_SR_em_snr_05.mat').ardb_sig_SR_em_snr_05;
x2 = load('noisy_samples/samples/ardb_sig_SR_em_snr_10.mat').ardb_sig_SR_em_snr_10;
x3 = load('noisy_samples/samples/ardb_sig_SR_em_snr_15.mat').ardb_sig_SR_em_snr_15;

% Model 1 - Reconstructions
m1_snr_00 = 'reconstructions/model_1/ardb_em_snr_00';
m1_snr_05 = 'reconstructions/model_1/ardb_em_snr_05';
m1_snr_10 = 'reconstructions/model_1/ardb_em_snr_10';
m1_snr_15 = 'reconstructions/model_1/ardb_em_snr_15';

% Model 2 - Reconstructions
m2_snr_00 = 'reconstructions/model_2/ardb_em_snr_00';
m2_snr_05 = 'reconstructions/model_2/ardb_em_snr_05';
m2_snr_10 = 'reconstructions/model_2/ardb_em_snr_10';
m2_snr_15 = 'reconstructions/model_2/ardb_em_snr_15';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Aligning the noisy signals vertically to the clean signal
x0 = vertical_align_by_mean(d, x0);
x1 = vertical_align_by_mean(d, x1);
x2 = vertical_align_by_mean(d, x2);
x3 = vertical_align_by_mean(d, x3);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Apply LMS Adaptive Filter
y0_LMS = lms_filter(x0,d);
y1_LMS = lms_filter(x1,d);
y2_LMS = lms_filter(x2,d);
y3_LMS = lms_filter(x3,d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model 1 - MAE and std.dev computation
[mae_m1_snr_00, std_m1_snr_00] = compute_avg_and_std_dev_MAE(d, m1_y0_list);
[mae_m1_snr_05, std_m1_snr_05] = compute_avg_and_std_dev_MAE(d, m1_y1_list);
[mae_m1_snr_10, std_m1_snr_10] = compute_avg_and_std_dev_MAE(d, m1_y2_list);
[mae_m1_snr_15, std_m1_snr_15] = compute_avg_and_std_dev_MAE(d, m1_y3_list);

% Model 2 - MAE and std. dev computation
[mae_m2_snr_00, std_m2_snr_00] = compute_avg_and_std_dev_MAE(d, m2_y0_list);
[mae_m2_snr_05, std_m2_snr_05] = compute_avg_and_std_dev_MAE(d, m2_y1_list);
[mae_m2_snr_10, std_m2_snr_10] = compute_avg_and_std_dev_MAE(d, m2_y2_list);
[mae_m2_snr_15, std_m2_snr_15] = compute_avg_and_std_dev_MAE(d, m2_y3_list);

% Hybrid (LPF -> LMS) - MAE computation
mae_lms_snr_00 = mean(abs(d - y0_LMS));
mae_lms_snr_05 = mean(abs(d - y1_LMS));
mae_lms_snr_10 = mean(abs(d - y2_LMS));
mae_lms_snr_15 = mean(abs(d - y3_LMS));

% Visualize the MAE results in grouped bar charts
snrs = [0, 5, 10, 15];
mae_model_1 = [mae_m1_snr_00, mae_m1_snr_05, mae_m1_snr_10, mae_m1_snr_15];
std_model_1 = [std_m1_snr_00, std_m1_snr_05, std_m1_snr_10, std_m1_snr_15];
mae_model_2 = [mae_m2_snr_00, mae_m2_snr_05, mae_m2_snr_10, mae_m2_snr_15];
std_model_2 = [std_m2_snr_00, std_m2_snr_05, std_m2_snr_10, std_m2_snr_15];
mae_LMS = [mae_lms_snr_00, mae_lms_snr_05, mae_lms_snr_10, mae_lms_snr_15];

figure;
hold on;
b = bar(snrs, [mae_LMS', mae_model_1', mae_model_2']);
% Adjust the position of the error bars to be centered on the Model 1 and Model 2 bars
nbars = size(b, 2);
x = nan(nbars, length(snrs));
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
% Plot the error bars
errorbar(x(2,:), mae_model_1, std_model_1, 'k', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 1 bars
errorbar(x(3,:), mae_model_2, std_model_2, 'r', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 2 bars

% Set x-axis ticks and labels
set(gca, 'XTick', snrs);
xlabel('SNR');
ylabel('Mean Absolute Error (MAE)');
title('Electrode Motion Noise Removal on ARDB');
legend('LMS', 'Model 1 (ARDB only)', 'Model 2 (AF Retrained');
grid on;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% FIRST PLOT: LMS model reconstructions at different SNR (overlayed) with legend
figure;


subplot(4,1,1);
hold on;
plot(d, 'LineWidth', 1.5, 'Color', '#013220');
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Original Signal');
grid on;

subplot(4,1,2);
hold on;
plot(y0_LMS, 'LineWidth', 1.5);
plot(y1_LMS, 'LineWidth', 1.5);
plot(y2_LMS, 'LineWidth', 1.5);
plot(y3_LMS, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('LMS Model Reconstructions at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% SECOND PLOT: Model 1 reconstructions at different SNR (overlayed) with legend
subplot(4,1,3);
hold on;
plot(m1_y0_list{1}, 'LineWidth', 1.5);
plot(m1_y1_list{1}, 'LineWidth', 1.5);
plot(m1_y2_list{1}, 'LineWidth', 1.5);
plot(m1_y3_list{1}, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Model 1 Reconstructions at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% THIRD PLOT: Model 2 reconstructions at different SNR (overlayed) with legend
subplot(4,1,4);
hold on;
plot(m2_y0_list{1}, 'LineWidth', 1.5);
plot(m2_y1_list{1}, 'LineWidth', 1.5);
plot(m2_y2_list{1}, 'LineWidth', 1.5);
plot(m2_y3_list{1}, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Model 2 Reconstructions at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

sgtitle('Electrode Motion Noise Removal on ARDB');





% We did investigate other metrics namely RMSE and PSNR
% However, these did not offer a lot of additional insights.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER METRICS ......  RMSE

% [rmse_m1_snr_00, std_m1_snr_00] = compute_avg_and_std_dev_RMSE(d, m1_y0_list);
% [rmse_m1_snr_05, std_m1_snr_05] = compute_avg_and_std_dev_RMSE(d, m1_y1_list);
% [rmse_m1_snr_10, std_m1_snr_10] = compute_avg_and_std_dev_RMSE(d, m1_y2_list);
% [rmse_m1_snr_15, std_m1_snr_15] = compute_avg_and_std_dev_RMSE(d, m1_y3_list);
% 
% % Model 2 - MAE and std. dev computation
% [rmse_m2_snr_00, std_m2_snr_00] = compute_avg_and_std_dev_RMSE(d, m2_y0_list);
% [rmse_m2_snr_05, std_m2_snr_05] = compute_avg_and_std_dev_RMSE(d, m2_y1_list);
% [rmse_m2_snr_10, std_m2_snr_10] = compute_avg_and_std_dev_RMSE(d, m2_y2_list);
% [rmse_m2_snr_15, std_m2_snr_15] = compute_avg_and_std_dev_RMSE(d, m2_y3_list);
% 
% % Hybrid (LPF -> LMS) - MAE computation
% rmse_lms_snr_00 = rmse(y0_LMS, d);
% rmse_lms_snr_05 = rmse(y1_LMS, d);
% rmse_lms_snr_10 = rmse(y2_LMS, d);
% rmse_lms_snr_15 = rmse(y3_LMS, d);
% 
% % Visualize the MAE results in grouped bar charts
% snrs = [0, 5, 10, 15];
% rmse_model_1 = [rmse_m1_snr_00, rmse_m1_snr_05, rmse_m1_snr_10, rmse_m1_snr_15];
% std_model_1 = [std_m1_snr_00, std_m1_snr_05, std_m1_snr_10, std_m1_snr_15];
% rmse_model_2 = [rmse_m2_snr_00, rmse_m2_snr_05, rmse_m2_snr_10, rmse_m2_snr_15];
% std_model_2 = [std_m2_snr_00, std_m2_snr_05, std_m2_snr_10, std_m2_snr_15];
% rmse_LMS = [rmse_lms_snr_00, rmse_lms_snr_05, rmse_lms_snr_10, rmse_lms_snr_15];
% 
% figure;
% hold on;
% b = bar(snrs, [rmse_LMS', rmse_model_1', rmse_model_2']);
% % Adjust the position of the error bars to be centered on the Model 1 and Model 2 bars
% nbars = size(b, 2);
% x = nan(nbars, length(snrs));
% for i = 1:nbars
%     x(i,:) = b(i).XEndPoints;
% end
% % Plot the error bars
% errorbar(x(2,:), rmse_model_1, std_model_1, 'k', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 1 bars
% errorbar(x(3,:), rmse_model_2, std_model_2, 'r', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 2 bars
% 
% % Set x-axis ticks and labels
% set(gca, 'XTick', snrs);
% xlabel('SNR');
% ylabel('Root Mean Square Error (RMSE)');
% title('Electrode Motion Noise Removal on ARDB');
% legend('LMS', 'Model 1 (ARDB only)', 'Model 2 (AF Retrained');
% grid on;
% hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER METRICS ......  PSNR

% [psnr_m1_snr_00, std_m1_snr_00] = compute_avg_and_std_dev_PSNR(d, m1_y0_list);
% [psnr_m1_snr_05, std_m1_snr_05] = compute_avg_and_std_dev_PSNR(d, m1_y1_list);
% [psnr_m1_snr_10, std_m1_snr_10] = compute_avg_and_std_dev_PSNR(d, m1_y2_list);
% [psnr_m1_snr_15, std_m1_snr_15] = compute_avg_and_std_dev_PSNR(d, m1_y3_list);
% 
% % Model 2 - MAE and std. dev computation
% [psnr_m2_snr_00, std_m2_snr_00] = compute_avg_and_std_dev_PSNR(d, m2_y0_list);
% [psnr_m2_snr_05, std_m2_snr_05] = compute_avg_and_std_dev_PSNR(d, m2_y1_list);
% [psnr_m2_snr_10, std_m2_snr_10] = compute_avg_and_std_dev_PSNR(d, m2_y2_list);
% [psnr_m2_snr_15, std_m2_snr_15] = compute_avg_and_std_dev_PSNR(d, m2_y3_list);
% 
% % Hybrid (LPF -> LMS) - MAE computation
% psnr_lms_snr_00 = psnr(y0_LMS, d);
% psnr_lms_snr_05 = psnr(y1_LMS, d);
% psnr_lms_snr_10 = psnr(y2_LMS, d);
% psnr_lms_snr_15 = psnr(y3_LMS, d);
% 
% % Visualize the MAE results in grouped bar charts
% snrs = [0, 5, 10, 15];
% psnr_model_1 = [psnr_m1_snr_00, psnr_m1_snr_05, psnr_m1_snr_10, psnr_m1_snr_15];
% std_model_1 = [std_m1_snr_00, std_m1_snr_05, std_m1_snr_10, std_m1_snr_15];
% psnr_model_2 = [psnr_m2_snr_00, psnr_m2_snr_05, psnr_m2_snr_10, psnr_m2_snr_15];
% std_model_2 = [std_m2_snr_00, std_m2_snr_05, std_m2_snr_10, std_m2_snr_15];
% psnr_LMS = [psnr_lms_snr_00, psnr_lms_snr_05, psnr_lms_snr_10, psnr_lms_snr_15];
% 
% figure;
% hold on;
% b = bar(snrs, [psnr_LMS', psnr_model_1', psnr_model_2']);
% % Adjust the position of the error bars to be centered on the Model 1 and Model 2 bars
% nbars = size(b, 2);
% x = nan(nbars, length(snrs));
% for i = 1:nbars
%     x(i,:) = b(i).XEndPoints;
% end
% % Plot the error bars
% errorbar(x(2,:), psnr_model_1, std_model_1, 'k', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 1 bars
% errorbar(x(3,:), psnr_model_2, std_model_2, 'r', 'linestyle', 'none', 'CapSize', 10); % Adding error bars to Model 2 bars
% 
% % Set x-axis ticks and labels
% set(gca, 'XTick', snrs);
% xlabel('SNR');
% ylabel('Peak Signal to Noise Ratio (PSNR)');
% title('Electrode Motion Noise Removal on ARDB');
% legend('LMS', 'Model 1 (ARDB only)', 'Model 2 (AF Retrained');
% grid on;
% hold off;