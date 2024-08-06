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

function recon = multi_shot_recon(y_list)
    % Number of reconstructions
    M = length(y_list);
    
    % Initialize the reconstruction to zero
    recon = zeros(size(y_list{1}));
    
    % Sum all reconstructions in y_list
    for i = 1:M
        recon = recon + y_list{i};
    end
    
    % Average the sum to get the final reconstruction
    recon = recon / M;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
d = load('src/samples/clean_samples/ardb_sig_HR.mat').sig_HR;

% Noisy Signals at SNR 0, 5, 10 and 15
x0 = load('src/samples/noisy_samples/ardb_sig_SR_comp_snr_00.mat').ardb_sig_SR_comp_snr_00;
x1 = load('src/samples/noisy_samples/ardb_sig_SR_comp_snr_05.mat').ardb_sig_SR_comp_snr_05;
x2 = load('src/samples/noisy_samples/ardb_sig_SR_comp_snr_10.mat').ardb_sig_SR_comp_snr_10;
x3 = load('src/samples/noisy_samples/ardb_sig_SR_comp_snr_15.mat').ardb_sig_SR_comp_snr_15;

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

% Multi-shot reconstruction I
m1_y0_multi = multi_shot_recon(m1_y0_list);
m1_y1_multi = multi_shot_recon(m1_y1_list);
m1_y2_multi = multi_shot_recon(m1_y2_list);
m1_y3_multi = multi_shot_recon(m1_y3_list);

% Multi-shot reconstructions II
m2_y0_multi = multi_shot_recon(m2_y0_list);
m2_y1_multi = multi_shot_recon(m2_y1_list);
m2_y2_multi = multi_shot_recon(m2_y2_list);
m2_y3_multi = multi_shot_recon(m2_y3_list);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hybrid (LPF -> LMS) - MAE computation]
mae_hybrid_snr_00 = mae(d - y0_hybrid_lpf_lms);
mae_hybrid_snr_05 = mae(d - y1_hybrid_lpf_lms);
mae_hybrid_snr_10 = mae(d - y2_hybrid_lpf_lms);
mae_hybrid_snr_15 = mae(d - y3_hybrid_lpf_lms);

mae_hybrid2_snr_00 = mae(d - y0_hybrid_lms_lpf);
mae_hybrid2_snr_05 = mae(d - y1_hybrid_lms_lpf);
mae_hybrid2_snr_10 = mae(d - y2_hybrid_lms_lpf);
mae_hybrid2_snr_15 = mae(d - y3_hybrid_lms_lpf);

% Multi-Shot M1 - MAE
mae_multi_m1_snr_00 = mean(abs(d - m1_y0_multi));
mae_multi_m1_snr_05 = mean(abs(d - m1_y1_multi));
mae_multi_m1_snr_10 = mean(abs(d - m1_y2_multi));
mae_multi_m1_snr_15 = mean(abs(d - m1_y3_multi));

% Multi-Shot M2 - MAE
mae_multi_m2_snr_00 = mean(abs(d - m2_y0_multi));
mae_multi_m2_snr_05 = mean(abs(d - m2_y1_multi));
mae_multi_m2_snr_10 = mean(abs(d - m2_y2_multi));
mae_multi_m2_snr_15 = mean(abs(d - m2_y3_multi));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Visualize the MAE results in grouped bar charts
snrs = [0, 5, 10, 15];
mae_hybrid2 = [mae_hybrid2_snr_00, mae_hybrid2_snr_05, mae_hybrid2_snr_10, mae_hybrid2_snr_15];
mae_multi_m1 = [mae_multi_m1_snr_00, mae_multi_m1_snr_05, mae_multi_m1_snr_10, mae_multi_m1_snr_15];
mae_multi_m2 = [mae_multi_m2_snr_00, mae_multi_m2_snr_05, mae_multi_m2_snr_10, mae_multi_m2_snr_15];

figure;
hold on;
b = bar(snrs, [mae_hybrid2', mae_multi_m1', mae_multi_m2']);

% Adjust the position of the error bars to be centered on the Model 1 and Model 2 bars
nbars = size(b, 2);
x = nan(nbars, length(snrs));
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end

% Set x-axis ticks and labels
set(gca, 'XTick', snrs);
xlabel('Signal-to-noise ratio input (dB)');
ylabel('Mean Absolute Error');
title('Composite Noise Removal on ARDB');
legend('Hybrid [LPF -> LMS]', 'Model 1 (6-Shot)', 'Model 2 (6-Shot)');
grid on;
ylim([0,0.16])
hold off;

% Save the figure
saveas(gcf, 'NEW_ardb_comp.png');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
plot(y0_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y1_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y2_hybrid_lpf_lms, 'LineWidth', 1.5);
plot(y3_hybrid_lpf_lms, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Hybrid [LPF -> LMS] - Estimated Clean ECGs at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% SECOND PLOT: Model 1 reconstructions at different SNR (overlayed) with legend
subplot(4,1,3);
hold on;
plot(m1_y0_multi, 'LineWidth', 1.5);
plot(m1_y1_multi, 'LineWidth', 1.5);
plot(m1_y2_multi, 'LineWidth', 1.5);
plot(m1_y3_multi, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Model 1 (6-Shot) - Estimated Clean ECGs at Different SNR');
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
title('Model 2 (6-Shot) - Estimated Clean ECGs at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

sgtitle('Composite Noise Removal on ARDB');