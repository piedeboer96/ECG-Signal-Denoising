%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to vertically align signals based on mean difference
function aligned_signal = vertical_align_by_mean(clean_signal, noisy_signal)
    mean_clean = mean(clean_signal);
    mean_noisy = mean(noisy_signal);
    mean_diff = mean_clean - mean_noisy;
    aligned_signal = noisy_signal + mean_diff;
end

% Function to vertically align reconstructed signals based on mean difference
function aligned_signal = vertical_align_reconstructed(clean_signal, rec_signal)
    vertical_shift_rec = mean(clean_signal) - mean(rec_signal);
    aligned_signal = rec_signal + vertical_shift_rec;
    aligned_signal = double(aligned_signal'); % Transpose and convert to double
end

% Function to load and align signals from a given path
function y_r = load_and_align_signals(path_to_files, clean_signal)
    % Get list of all .mat files in the directory
    files = dir(fullfile(path_to_files, '*.mat'));
    
    % Initialize a cell array to store the loaded data
    y_r = cell(1, numel(files));
    
    % Load each file and store the data in the array
    for i = 1:numel(files)
        % Construct the full file name
        file_name = fullfile(path_to_files, files(i).name);
        % Load the file
        loaded_data = load(file_name);
        % Store the loaded data in the array
        y_r{i} = loaded_data.sig_rec;
    end
    
    % Align each signal to the clean signal
    for i = 1:numel(y_r)
        y_r{i} = vertical_align_reconstructed(clean_signal, y_r{i});
    end
end

% Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
function [mean_values, std_dev_values] = compute_avg_and_std_dev_MAE(clean_signal, reconstructed_signals)
    num_signals = length(reconstructed_signals);
    MAE_values = zeros(1, num_signals);
    for i = 1:num_signals
        MAE_values(i) = mean(abs(clean_signal' - reconstructed_signals{i}'));
    end
    mean_values = mean(MAE_values);
    std_dev_values = std(MAE_values);
end

% Function to use low pass filter
function y_LPF = low_pass_filter(x)
    % LOW_PASS_FILTER applies a low-pass filter to the input signal x and compensates for the delay.
    %
    % Inputs:
    %   x  - The noisy input signal
    %   fs - Sampling rate (Hz)
    %   fc - Cutoff frequency (Hz)
    %   N  - Filter length
    %
    % Output:
    %   y_LPF - The filtered and delay-compensated signal
        
    fs = 128; % Sampling rate (Hz)
    fc = 40.60449; % Cutoff frequency (Hz)  - adapted for changed sampling frequency
    N = 14; % Filter length
    phi_d = 0.1047; % Phase delay (rad/Hz)

    % Normalize the cutoff frequency
    fc_norm = fc / fs;

    % Design the FIR filter using the Kaiser window method
    beta = 0; % Kaiser window parameter
    fir_coeffs = fir1(N-1, fc_norm, 'low', kaiser(N, beta));

    % Calculate the delay
    delay = floor((N-1) / 2);

    % Ensure x is a column vector for consistency
    x = x(:);

    % Apply filter and compensate for the delay
    y_LPF = filter(fir_coeffs, 1, [x; zeros(delay, 1)]);
    y_LPF = y_LPF(delay+1:end);
    y_LPF = y_LPF';

end

% Function to use LMS filter
% function y_LMS = lms_filter(x, d)
%     % Ensure x and d are column vectors for consistency
%     x = x(:);
%     d = d(:);
% 
%     % Create LMS filter object
%     lms = dsp.LMSFilter();
% 
%     % Apply LMS filter
%     [y_LMS, ~, ~] = lms(x, d);
% 
%     % Convert the output to a row vector for consistency
%     y_LMS = y_LMS';
% end

function y_LMS = lms_filter(x, d)
    % Ensure x and d are column vectors for consistency
    x = x(:);
    d = d(:);

    % Create LMS filter object
    lms = dsp.LMSFilter();

    % Pre-pend zeros to the input signal
    pre_pend_length = 50; % Length of the zero buffer
    x_prepended = [zeros(pre_pend_length, 1); x];
    d_prepended = [zeros(pre_pend_length, 1); d];

    % Apply LMS filter
    [y_LMS_prepended, ~, ~] = lms(x_prepended, d_prepended);

    % Remove the prepended zeros from the output
    y_LMS = y_LMS_prepended(pre_pend_length + 1:end);

    % Convert the output to a row vector for consistency
    y_LMS = y_LMS';
end

% Function to use Hybrid Filter
function y_hybrid = hybrid_filter_lms_lpf(x, d)

    % Apply LMS filter
    y_LMS = lms_filter(x, d);

    % y_hybrid = y_LMS;
    % Apply low-pass filter
    y_hybrid = low_pass_filter(y_LMS);
end

% Function to use Hybrid Filter
function y_hybrid = hybrid_filter_lpf_lms(x, d)
    % HYBRID_FILTER applies LMS filter followed by a low-pass filter to the input signal x using the target signal d.
    %
    % Inputs:
    %   x - The noisy input signal
    %   d - The target signal
    %
    % Output:
    %   y_hybrid - The filtered signal

    % Apply LPF
    y_LPF = low_pass_filter(x);

    % Apply LMS
    y_hybrid = lms_filter(y_LPF,d);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
d = load('noisy_samples/af_sig_HR.mat').sig_HR;

% Noise sample
noise = load('noisy_samples/slices/comp_slice_ind.mat').comp_slice_ind;

% Noisy Signals at SNR 0, 5, 10 and 15
x0 = load('noisy_samples/samples/af_sig_SR_comp_snr_00.mat').af_sig_SR_comp_snr_00;
x1 = load('noisy_samples/samples/af_sig_SR_comp_snr_05.mat').af_sig_SR_comp_snr_05;
x2 = load('noisy_samples/samples/af_sig_SR_comp_snr_10.mat').af_sig_SR_comp_snr_10;
x3 = load('noisy_samples/samples/af_sig_SR_comp_snr_15.mat').af_sig_SR_comp_snr_15;

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hybrid (LPF -> LMS) - MAE computation]
mae_hybrid2_snr_00 = mae(d - y0_hybrid_lpf_lms);
mae_hybrid2_snr_05 = mae(d - y1_hybrid_lpf_lms);
mae_hybrid2_snr_10 = mae(d - y2_hybrid_lpf_lms);
mae_hybrid2_snr_15 = mae(d - y3_hybrid_lpf_lms);


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
xlabel('SNR');
ylabel('Mean Absolute Error (MAE)');
title('Composite Noise Removal - Artial Fibrillation  (AF)');
legend('Hybrid [LPF -> LMS]', 'Model 1 (ARDB Trained)', 'Model 2 (AF Retrained)');
grid on;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% % FIRST PLOT: LMS model reconstructions at different SNR (overlayed) with legend
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
% sgtitle('Composite Noise Removal on AF');



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
plot(y0_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y1_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y2_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y3_hybrid_lpf_lms, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('LMS Model Reconstructions at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15', 'Location', 'northeast');
grid on;

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

sgtitle('Composite Noise Removal on AF');
