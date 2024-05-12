%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the clean signal
data_HR = load('run_2/sig_HR.mat');
sig_HR = data_HR.sig_HR; 

% Load the noisy signal
data_SR = load('run_2/sig_SR.mat');
sig_SR = data_SR.sig_SR; 

% Load denoised signal from SR3
data_rec = load('run_2/sig_rec.mat');
sig_rec = data_rec.sig_rec;                 % denoise SR3 model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wavelet Denoising
sig_wavelet = wdenoise(sig_SR, 3, 'Wavelet', 'sym12'); % settings from Paper 2024
sig_wavelet_MATLAB = wdenoise(sig_SR);                 % MATLAB

% Savetzky Golay Filter
sig_sg = sgolayfilt(sig_SR, 8, 31);                    % from paper 2024

% Adaptive Filter LMS
lms = dsp.LMSFilter('Method', 'LMS', 'Length', 32, 'StepSize', 0.0276); % setting from paper
lms_default = dsp.LMSFilter(); % MATLAB propertiery stepsize source

% Adaptive Filter RLS
rls = dsp.RLSFilter()


x = sig_SR';
d = sig_HR';                

[y0, err0, wts0] = lms_default(x, d);                                   % prop stepsource
[y1, err1, wts1] = lms(x,d);

% RLS FILTERD
[y,e] = rlsFilt(x,d)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the vertical shift needed to align the reconstructed signal with the clean signal
vertical_shift_rec = mean(sig_HR) - mean(sig_rec);

% Apply the vertical shift to the reconstructed signal
sig_rec_aligned = sig_rec + vertical_shift_rec;
sig_rec_aligned = double(sig_rec_aligned');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RMSE
rmse_wavelet = rmse(sig_HR, sig_wavelet); rmse_wavelet_str = sprintf('RMSE: %.4f', rmse_wavelet);
rmse_wavelet_MATLAB = rmse(sig_HR, sig_wavelet_MATLAB); rmse_wavelet_MATLAB_str = sprintf('RMSE: %.4f', rmse_wavelet_MATLAB);
rmse_lms_MATLAB = rmse(d, y0); rmse_lms_MATLAB_str = sprintf('RMSE: %.4f', rmse_lms_MATLAB);
rmse_lms_PAPER = rmse(d,y1); rmse_lms_PAPER_str = sprintf('RMSE: %.4f', rmse_lms_PAPER);
rmse_sg = rmse(sig_HR,sig_sg); rmse_sg_str = sprintf('RMSE: %.4f', rmse_sg);
rmse_SR3 = rmse(sig_HR, sig_rec_aligned); rmse_SR3_str = sprintf('RMSE: %.4f', rmse_SR3);

% PSNR
psnr_wavelet = psnr(sig_wavelet, sig_HR);
psnr_wavelet_MATLAB = psnr(sig_wavelet_MATLAB, sig_HR);
psnr_lms_MATLAB = psnr(y0, d);
psnr_lms_PAPER = psnr(y0, d);
psnr_sg = psnr(sig_sg, sig_HR);
psnr_SR3 = psnr(sig_rec_aligned, sig_HR);

% Compute DTW for sig_wavelet
dtw_wavelet = dtw(sig_HR, sig_wavelet);
dtw_wavelet_MATLAB = dtw(sig_HR, sig_wavelet_MATLAB);
dtw_lms_MATLAB = dtw(sig_HR, y0);
dtw_lms_PAPER = dtw(sig_HR, y1);
dtw_sg = dtw(sig_HR, sig_sg);
dtw_SR3 = dtw(sig_HR, sig_rec_aligned);


% Define the data for the table
data = {
    rmse_wavelet_str, psnr_wavelet, dtw_wavelet;
    rmse_wavelet_MATLAB_str, psnr_wavelet_MATLAB, dtw_wavelet_MATLAB;
    rmse_lms_MATLAB_str, psnr_lms_MATLAB, dtw_lms_MATLAB;
    rmse_lms_PAPER_str, psnr_lms_PAPER, dtw_lms_PAPER;
    rmse_sg_str, psnr_sg, dtw_sg;
    rmse_SR3_str, psnr_SR3, dtw_SR3
};

% Define the row and column names
row_names = {'Wavelet', 'Wavelet (MATLAB)', 'LMS (MATLAB)', 'LMS (Paper)', 'SG Filter', 'SR3'};
col_names = {'RMSE (lower better)', 'PSNR (higher better)', 'DTW (lower better)'};

% Create the table
table_results = array2table(data, 'RowNames', row_names, 'VariableNames', col_names);

% Display the table
disp(table_results);



% Define the data for the bar charts
rmse_values = [rmse_wavelet, rmse_wavelet_MATLAB, rmse_lms_MATLAB, rmse_lms_PAPER, rmse_sg, rmse_SR3];
psnr_values = [psnr_wavelet, psnr_wavelet_MATLAB, psnr_lms_MATLAB, psnr_lms_PAPER, psnr_sg, psnr_SR3];
dtw_values = [dtw_wavelet, dtw_wavelet_MATLAB, dtw_lms_MATLAB, dtw_lms_PAPER, dtw_sg, dtw_SR3];

% Create figure for bar charts
figure;

% Plot RMSE bar chart
subplot(1,3,1);
bar(rmse_values);
title('RMSE (lower better)');
ylabel('Value');
set(gca, 'xticklabel', row_names);
xtickangle(45);

% Plot PSNR bar chart
subplot(1,3,2);
bar(psnr_values);
title('PSNR (higher better)');
ylabel('Value');
set(gca, 'xticklabel', row_names);
xtickangle(45);

% Plot DTW bar chart
subplot(1,3,3);
bar(dtw_values);
title('DTW (lower better)');
ylabel('Value');
set(gca, 'xticklabel', row_names);
xtickangle(45);

% Adjust layout
sgtitle('Comparison of Denoising Methods');

% Define the data for the bar charts
methods = {'Wavelet', 'Wavelet (MATLAB)', 'LMS (MATLAB)', 'LMS (Paper)', 'SG Filter', 'SR3'};
rmse_values = [rmse_wavelet, rmse_wavelet_MATLAB, rmse_lms_MATLAB, rmse_lms_PAPER, rmse_sg, rmse_SR3];
psnr_values = [psnr_wavelet, psnr_wavelet_MATLAB, psnr_lms_MATLAB, psnr_lms_PAPER, psnr_sg, psnr_SR3];
dtw_values = [dtw_wavelet, dtw_wavelet_MATLAB, dtw_lms_MATLAB, dtw_lms_PAPER, dtw_sg, dtw_SR3];

% Create grouped bar chart
figure;
bar_data = [rmse_values; psnr_values; dtw_values];
bar_groups = bar(bar_data);
title('Comparison of Denoising Methods');
ylabel('Value');
legend('RMSE', 'PSNR', 'DTW');
set(gca, 'xticklabel', methods);
xtickangle(45);

% Adjust colors
colors = lines(3); % Get colors for each group
for i = 1:numel(bar_groups)
    bar_groups(i).FaceColor = colors(mod(i-1, 3)+1, :); % Assign color to each group
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot original and denoised signals
figure;

subplot(2,1,1);
plot(sig_HR);
title('Original Signal');

subplot(2,1,2);
plot(sig_SR);
title('Original Noisy Signal');

% Create a new figure for denoised signals
% Plot original and denoised signals
figure;

subplot(3,2,1);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_wavelet, 'Color', 'red');
hold off;
title('Denoised Signal (Wavelet)');
legend({'Original Signal', ['Denoised Signal ', rmse_wavelet_str]}, 'Location', 'best');

subplot(3,2,2);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_wavelet_MATLAB, 'Color', 'red');
hold off;
title('Denoised Signal (Wavelet MATLAB)');
legend({'Original Signal', ['Denoised Signal ', rmse_wavelet_MATLAB_str]}, 'Location', 'best');

subplot(3,2,3);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(y0, 'Color', 'red');
hold off;
title('Denoised Signal (LMS MATLAB)');
legend({'Original Signal', ['Denoised Signal ', rmse_lms_MATLAB_str]}, 'Location', 'best');

subplot(3,2,4);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(y1, 'Color', 'red');
hold off;
title('Denoised Signal (LMS Paper)');
legend({'Original Signal', ['Denoised Signal ', rmse_lms_PAPER_str]}, 'Location', 'best');

subplot(3,2,5);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_sg, 'Color', 'red');
hold off;
title('Denoised Signal (SG Filter)');
legend({'Original Signal', ['Denoised Signal ', rmse_sg_str]}, 'Location', 'best');

subplot(3,2,6);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_rec_aligned, 'Color', 'red');
hold off;
title('Denoised Signal (SR3)');
legend({'Original Signal', ['Denoised Signal ', rmse_SR3_str]}, 'Location', 'best');




