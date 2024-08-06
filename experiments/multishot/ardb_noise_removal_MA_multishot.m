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

function y_MA = moving_average_filter(x)
    % Define the coefficients of the transfer function H(z)
    b = 1/4 * ones(1, 4);  % Numerator coefficients (1/4 * ones(1, 4))
    a = 1;                 % Denominator coefficients (1 - z^-1)

    % Apply filter
    y_MA = filter(b, a, x);
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
x0 = load('src/samples/noisy_samples/ardb_sig_SR_ma_snr_00.mat').ardb_sig_SR_ma_snr_00;
x1 = load('src/samples/noisy_samples/ardb_sig_SR_ma_snr_05.mat').ardb_sig_SR_ma_snr_05;
x2 = load('src/samples/noisy_samples/ardb_sig_SR_ma_snr_10.mat').ardb_sig_SR_ma_snr_10;
x3 = load('src/samples/noisy_samples/ardb_sig_SR_ma_snr_15.mat').ardb_sig_SR_ma_snr_15;

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

% Apply LFP
y0_LPF = low_pass_filter(x0);
y1_LPF = low_pass_filter(x1);
y2_LPF = low_pass_filter(x2);
y3_LPF = low_pass_filter(x3);

% Apply MA
y0_MA = moving_average_filter(x0);
y1_MA = moving_average_filter(x1);
y2_MA = moving_average_filter(x2);
y3_MA = moving_average_filter(x3);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Moving Average (LPF)
mae_ma_snr_00 = mean(abs(d - y0_MA));
mae_ma_snr_05 = mean(abs(d - y1_MA));
mae_ma_snr_10 = mean(abs(d - y2_MA));
mae_ma_snr_15 = mean(abs(d - y3_MA));

% Hybrid (LPF -> LMS) - MAE computation
mae_lpf_snr_00 = mean(abs(d - y0_LPF));
mae_lpf_snr_05 = mean(abs(d - y1_LPF));
mae_lpf_snr_10 = mean(abs(d - y2_LPF));
mae_lpf_snr_15 = mean(abs(d - y3_LPF));

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize the MAE results in grouped bar charts
snrs = [0, 5, 10, 15];

mae_MA = [mae_ma_snr_00, mae_ma_snr_05, mae_ma_snr_10, mae_ma_snr_15];
mae_LPF = [mae_lpf_snr_00, mae_lpf_snr_05, mae_lpf_snr_10, mae_lpf_snr_15];
mae_multi_m1 = [mae_multi_m1_snr_00, mae_multi_m1_snr_05, mae_multi_m1_snr_10, mae_multi_m1_snr_15];
mae_multi_m2 = [mae_multi_m2_snr_00, mae_multi_m2_snr_05, mae_multi_m2_snr_10, mae_multi_m2_snr_15];

figure;
hold on;

% Plot the bar chart
b = bar(snrs, [mae_MA', mae_LPF', mae_multi_m1', mae_multi_m2']);

% Manually specify the colors for each bar group using hexadecimal codes
colors = {'#0072BD', '#7E2F8E', '#D95319', '#EDB120'};  % Example colors: blue, orange, yellow, purple
for k = 1:length(b)
    b(k).FaceColor = colors{k};
end

% Set x-axis ticks and labels
set(gca, 'XTick', snrs, 'XTickLabel', snrs);
xlabel('Signal-to-noise ratio input (dB)');
ylabel('Mean Absolute Error');
title('Muscle Artifact Noise Removal on ARDB');
legend('Moving Average (N=4)', 'LPF Kaiser', 'Model 1 (6-Shot)', 'Model 2 (6-Shot)');
grid on;
hold off;

% Save the figure
saveas(gcf, 'NEW_ardb_ma.png');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recon_multi_m1 = {m1_y0_multi', m1_y1_multi', m1_y2_multi', m1_y3_multi'};
recon_multi_m2 = [m2_y0_multi', m2_y1_multi', m2_y2_multi', m2_y3_multi'];


% Plots of reconstructions at various input SNR for the different models 
figure;

subplot(5,1,1);
hold on;
plot(d, 'LineWidth', 1.5, 'Color', '#013220');
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Original Signal');
grid on;

subplot(5,1,2);
hold on;
plot(y0_MA, 'LineWidth', 1.5);
plot(y1_MA, 'LineWidth', 1.5);
plot(y2_MA, 'LineWidth', 1.5);
plot(y3_MA, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Moving Average - Estimated Clean ECGs at Different SNRs');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

subplot(5,1,3);
hold on;
plot(y0_LPF, 'LineWidth', 1.5);
plot(y1_LPF, 'LineWidth', 1.5);
plot(y2_LPF, 'LineWidth', 1.5);
plot(y3_LPF, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('LPF Kaiser - Estimated Clean ECGs at Different SNRs');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% THIRD PLOT: Model 1 reconstructions at different SNR (overlayed) with legend
subplot(5,1,4);
hold on;
plot(recon_multi_m1{1}, 'LineWidth', 1.5);
plot(recon_multi_m1{2}, 'LineWidth', 1.5);
plot(recon_multi_m1{3}, 'LineWidth', 1.5);
plot(recon_multi_m1{4}, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Model 1 (6-Shot) - Estimated Clean ECGs at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% FOURTH PLOT: Model 2 reconstructions at different SNR (overlayed) with legend
subplot(5,1,5);
hold on;
plot(recon_multi_m2(:, 1), 'LineWidth', 1.5);
plot(recon_multi_m2(:, 2), 'LineWidth', 1.5);
plot(recon_multi_m2(:, 3), 'LineWidth', 1.5);
plot(recon_multi_m2(:, 4), 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Model 2 (6-Shot) - Estimated Clean ECGs at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

sgtitle('Muscle Artifact Noise Removal on ARDB');

% TODO:
% Save the figure to a .png file
%saveas(gcf, 'reconstructions_ardb_ma.png');

