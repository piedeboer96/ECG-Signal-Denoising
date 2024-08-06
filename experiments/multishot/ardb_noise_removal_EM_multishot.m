
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

function y_LMS = lms_filter(x, d)
    % Function to use LMS filter
    x = x(:);
    d = d(:);
    lms = dsp.LMSFilter();
    [y_LMS, ~, ~] = lms(x, d);
    y_LMS = y_LMS';
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
x0 = load('src/samples/noisy_samples/ardb_sig_SR_em_snr_00.mat').ardb_sig_SR_em_snr_00;
x1 = load('src/samples/noisy_samples/ardb_sig_SR_em_snr_05.mat').ardb_sig_SR_em_snr_05;
x2 = load('src/samples/noisy_samples/ardb_sig_SR_em_snr_10.mat').ardb_sig_SR_em_snr_10;
x3 = load('src/samples/noisy_samples/ardb_sig_SR_em_snr_15.mat').ardb_sig_SR_em_snr_15;

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
% Hybrid (LPF -> LMS) - MAE computation
mae_lms_snr_00 = mean(abs(d - y0_LMS));
mae_lms_snr_05 = mean(abs(d - y1_LMS));
mae_lms_snr_10 = mean(abs(d - y2_LMS));
mae_lms_snr_15 = mean(abs(d - y3_LMS));

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Visualize the MAE results in grouped bar charts
snrs = [0, 5, 10, 15];

mae_LMS = [mae_lms_snr_00, mae_lms_snr_05, mae_lms_snr_10, mae_lms_snr_15];
mae_multi_m1 = [mae_multi_m1_snr_00, mae_multi_m1_snr_05, mae_multi_m1_snr_10, mae_multi_m1_snr_15];
mae_multi_m2 = [mae_multi_m2_snr_00, mae_multi_m2_snr_05, mae_multi_m2_snr_10, mae_multi_m2_snr_15];

figure;
hold on;

% Plot the bar chart
b = bar(snrs, [mae_LMS', mae_multi_m1', mae_multi_m2']);

% Manually specify the colors for each bar group using hexadecimal codes
colors = {'#0072BD', '#D95319', '#EDB120'};  % Example colors: blue, orange, yellow
for k = 1:length(b)
    b(k).FaceColor = colors{k};
end

% Set x-axis ticks and labels
set(gca, 'XTick', snrs);
xlabel('Signal-to-noise ratio input (dB)');
ylabel('Mean Absolute Error');
title('Electrode Motion Noise Removal on ARDB');
legend('LMS', 'Model 1 (6-Shot)', 'Model 2 (6-Shot)');
grid on;
ylim([0,0.16])
hold off;

% Save the figure
saveas(gcf, 'NEW_ardb_em.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstructions
recon_multi_m1 = {m1_y0_multi', m1_y1_multi', m1_y2_multi', m1_y3_multi'};
recon_multi_m2 = [m2_y0_multi', m2_y1_multi', m2_y2_multi', m2_y3_multi'];

% Create a figure for the subplots
figure;

% FIRST PLOT: Original Signal
subplot(4,1,1);
plot(d, 'LineWidth', 1.5, 'Color', '#013220');
xlabel('Sample');
ylabel('Amplitude');
title('Original Signal');
grid on;

% SECOND PLOT: LMS model reconstructions at different SNR (overlayed) with legend
subplot(4,1,2);
hold on;
plot(y0_LMS, 'LineWidth', 1.5);
plot(y1_LMS, 'LineWidth', 1.5);
plot(y2_LMS, 'LineWidth', 1.5);
plot(y3_LMS, 'LineWidth', 1.5);
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('LMS Model - Estimated Clean ECGs at Different SNR');
legend('SNR 0', 'SNR 5', 'SNR 10', 'SNR 15');
grid on;

% THIRD PLOT: Model 1 reconstructions at different SNR (overlayed) with legend
subplot(4,1,3);
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
subplot(4,1,4);
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

% Add a super title for the entire figure
sgtitle('Electrode Motion Noise Removal on ARDB');







