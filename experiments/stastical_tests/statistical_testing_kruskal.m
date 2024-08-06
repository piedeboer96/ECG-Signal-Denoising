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

function mean_MAE_values = compute_avg_MAE_values(clean_signal, reconstructed_signals)
    % Function to compute MAE from clean and reconstructions, and calculate mean and standard deviation
    num_signals = length(reconstructed_signals);
    MAE_values = zeros(1, num_signals);
    for i = 1:num_signals
        MAE_values(i) = mean(abs(clean_signal - reconstructed_signals{i}));
    end

    mean_MAE_values = MAE_values
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
d = load('src/samples/clean_samples/af_sig_HR.mat').sig_HR;

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

