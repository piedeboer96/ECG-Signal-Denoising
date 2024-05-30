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
        MAE_values(i) = mean(abs(clean_signal' - reconstructed_signals{i}'));
    end
    mean_values = mean(MAE_values);
    std_dev_values = std(MAE_values);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
d = load('noisy_samples/ardb_sig_HR.mat').sig_HR;

% Noise sample
% noise = load('noisy_samples/slices/comp_slice_ind.mat').comp_slice_ind;

% Noisy Signals at SNR 0, 5, 10 and 15
x0 = load('noisy_samples/samples/ardb_sig_SR_comp_snr_00.mat').ardb_sig_SR_comp_snr_00;
x1 = load('noisy_samples/samples/ardb_sig_SR_comp_snr_05.mat').ardb_sig_SR_comp_snr_05;
x2 = load('noisy_samples/samples/ardb_sig_SR_comp_snr_10.mat').ardb_sig_SR_comp_snr_10;
x3 = load('noisy_samples/samples/ardb_sig_SR_comp_snr_15.mat').ardb_sig_SR_comp_snr_15;

% Model 1 - Reconstructions
m1_snr_00 = 'reconstructions/model_1/ardb_comp_snr_00';
m1_snr_05 = 'reconstructions/model_1/ardb_comp_snr_05';
m1_snr_10 = 'reconstructions/model_1/ardb_comp_snr_10';
m1_snr_15 = 'reconstructions/model_1/ardb_comp_snr_15';

% Model 2 - Reconstructions
m2_snr_00 = 'reconstructions/model_2/ardb_comp_snr_00';
m2_snr_05 = 'reconstructions/model_2/ardb_comp_snr_05';
m2_snr_10 = 'reconstructions/model_2/ardb_comp_snr_10';
m2_snr_15 = 'reconstructions/model_2/ardb_comp_snr_15';

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

% Apply hybrid_lpf_lms
y0_hybrid_lpf_lms = hybrid_filter_lpf_lms(x0, d);
y1_hybrid_lpf_lms = hybrid_filter_lpf_lms(x1, d);
y2_hybrid_lpf_lms = hybrid_filter_lpf_lms(x2, d);
y3_hybrid_lpf_lms = hybrid_filter_lpf_lms(x3, d);

y0_hybrid_lms_lpf = hybrid_filter_lms_lpf(x0, d);
y1_hybrid_lms_lpf = hybrid_filter_lms_lpf(x1, d);
y2_hybrid_lms_lpf = hybrid_filter_lms_lpf(x2, d);
y3_hybrid_lms_lpf = hybrid_filter_lms_lpf(x3, d);

% Apply hybrid_lms_lpf
y0_dwt = wavelet_denoise(x0);
y1_dwt = wavelet_denoise(x1);
y2_dwt = wavelet_denoise(x2);
y3_dwt = wavelet_denoise(x3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% Hybrid (LPF -> LMS) - MAE computation]
mae_hybrid_snr_00 = mae(d - y0_hybrid_lpf_lms);
mae_hybrid_snr_05 = mae(d - y1_hybrid_lpf_lms);
mae_hybrid_snr_10 = mae(d - y2_hybrid_lpf_lms);
mae_hybrid_snr_15 = mae(d - y3_hybrid_lpf_lms);

mae_hybrid2_snr_00 = mae(d - y0_hybrid_lms_lpf);
mae_hybrid2_snr_05 = mae(d - y1_hybrid_lms_lpf);
mae_hybrid2_snr_10 = mae(d - y2_hybrid_lms_lpf);
mae_hybrid2_snr_15 = mae(d - y3_hybrid_lms_lpf);

% Hybrid DWT SYM 5 - MAE computation
% mae_dwt_snr_00 = mae(d - y0_dwt);
% mae_dwt_snr_05 = mae(d - y1_dwt);
% mae_dwt_snr_10 = mae(d - y2_dwt);
% mae_dwt_snr_15 = mae(d - y3_dwt);

% TODO....
% Visualize the MAE results in grouped bar charts
snrs = [0, 5, 10, 15];
mae_model_1 = [mae_m1_snr_00, mae_m1_snr_05, mae_m1_snr_10, mae_m1_snr_15];
std_model_1 = [std_m1_snr_00, std_m1_snr_05, std_m1_snr_10, std_m1_snr_15];
mae_model_2 = [mae_m2_snr_00, mae_m2_snr_05, mae_m2_snr_10, mae_m2_snr_15];
std_model_2 = [std_m2_snr_00, std_m2_snr_05, std_m2_snr_10, std_m2_snr_15];
mae_hybrid2 = [mae_hybrid2_snr_00, mae_hybrid2_snr_05, mae_hybrid2_snr_10, mae_hybrid2_snr_15];


figure;
hold on;
b = bar(snrs, [mae_hybrid2', mae_model_1', mae_model_2']);
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
xlabel('Signal-to-noise ratio input (dB)');
ylabel('Mean Absolute Error');
title('Composite Noise Removal on ARDB');
legend('Hybrid [LPF -> LMS]', 'Model 1 (ARDB Trained)', 'Model 2 (AF Retrained');
grid on;
hold off;

% Save plot to PNG
saveas(gcf, 'results_ardb_comp_bars.png');


% the plot should be 3 rows and 1 column

% FIRST PLOT: LMS model reconstructions at different SNR (overlayed) with legend
% figure;
% 
% 
% subplot(4,1,1);
% hold on;
% plot(d, 'LineWidth', 1.5, 'Color', '#013220');
% hold off;
% xlabel('Sample');
% ylabel('Amplitude');
% title('Original Signal');
% grid on;
% 
% 
% 
% subplot(4,1,2);
% hold on;
% plot(y0_hybrid_lpf_lms, 'LineWidth', 1.5);
% plot(y1_hybrid_lpf_lms, 'LineWidth', 1.5);
% plot(y2_hybrid_lpf_lms, 'LineWidth', 1.5);
% plot(y3_hybrid_lpf_lms, 'LineWidth', 1.5);
% hold off;
% xlabel('Sample');
% ylabel('Amplitude');
% title('LMS Model Reconstructions at Different SNR');
% legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
% grid on;
% 
% % SECOND PLOT: Model 1 reconstructions at different SNR (overlayed) with legend
% subplot(4,1,3);
% hold on;
% plot(m1_y0_list{1}, 'LineWidth', 1.5);
% plot(m1_y1_list{1}, 'LineWidth', 1.5);
% plot(m1_y2_list{1}, 'LineWidth', 1.5);
% plot(m1_y3_list{1}, 'LineWidth', 1.5);
% hold off;
% xlabel('Sample');
% ylabel('Amplitude');
% title('Model 1 Reconstructions at Different SNR');
% legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
% grid on;
% 
% % THIRD PLOT: Model 2 reconstructions at different SNR (overlayed) with legend
% subplot(4,1,4);
% hold on;
% plot(m2_y0_list{1}, 'LineWidth', 1.5);
% plot(m2_y1_list{1}, 'LineWidth', 1.5);
% plot(m2_y2_list{1}, 'LineWidth', 1.5);
% plot(m2_y3_list{1}, 'LineWidth', 1.5);
% hold off;
% xlabel('Sample');
% ylabel('Amplitude');
% title('Model 2 Reconstructions at Different SNR');
% legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
% grid on;
% 
% sgtitle('Composite Noise Removal on ARDB');