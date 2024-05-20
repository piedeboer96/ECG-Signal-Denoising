%% BaslineModels vs SR3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the clean signal
data_HR = load('results/af/01/sig_HR.mat');
sig_HR = data_HR.sig_HR; 

% Load the noisy signal
data_SR = load('results/af/01/sig_SR.mat');
sig_SR = data_SR.sig_SR; 

% Load denoised signal from SR3
data_rec = load('results/af/01/sig_rec.mat');
sig_rec = data_rec.sig_rec;                 % denoise SR3 model


%%%%%%%%%%%%%%%
% Allign SR3
% Calculate the mean of each signal
mean_HR = mean(sig_HR);
mean_SR = mean(sig_SR);

% Calculate the difference between their means
mean_diff = mean_HR - mean_SR;

% Shift sig_SR vertically by mean_diff
sig_SR = sig_SR + mean_diff;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wavelet Denoising
sig_wavelet = wdenoise(sig_SR, 3, 'Wavelet', 'sym12'); % Paper 2014 (better than MATLAB default)

% Adaptive Filter LMS
lms = dsp.LMSFilter();                                 % MATLAB proprietary is better!

x = sig_SR';
d = sig_HR';                

[y, err, wts] = lms(x, d);              


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the vertical shift needed to align the reconstructed signal with the clean signal
vertical_shift_rec = mean(sig_HR) - mean(sig_rec);

% Apply the vertical shift to the reconstructed signal
sig_rec_aligned = sig_rec + vertical_shift_rec;
sig_rec_aligned = double(sig_rec_aligned');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RMSE
rmse_wavelet = rmse(sig_HR, sig_wavelet); rmse_wavelet_str = sprintf('RMSE: %.4f', rmse_wavelet);
rmse_lms= rmse(d, y); rmse_lms_str = sprintf('RMSE: %.4f', rmse_lms);
rmse_SR3 = rmse(sig_HR, sig_rec_aligned); rmse_SR3_str = sprintf('RMSE: %.4f', rmse_SR3);


% PSNR
psnr_wavelet = psnr(sig_wavelet, sig_HR);
psnr_lms= psnr(y, d);
psnr_SR3 = psnr(sig_rec_aligned, sig_HR);


% Compute mae for sig_wavelet
mae_wavelet = mae(sig_HR, sig_wavelet);
mae_lms = mae(sig_HR, y);
mae_SR3 = mae(sig_HR, sig_rec_aligned);



% Define the data for the table
data = {
    rmse_wavelet_str, psnr_wavelet, mae_wavelet;
    rmse_lms_str, psnr_lms, mae_lms;   
    rmse_SR3_str, psnr_SR3, mae_SR3
};

% Define the row and column names
row_names = {'Wavelet (Paper)','LMS (MATLAB)', 'SR3'};
col_names = {'RMSE (lower better)', 'PSNR (higher better)', 'MAE (lower better)'};

% Create the table
table_results = array2table(data, 'RowNames', row_names, 'VariableNames', col_names);

% Display the table
disp(table_results);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot original and denoised signals
figure;

subplot(2,1,1);
plot(sig_HR);
title('Original Signal (MIT AR');

subplot(2,1,2);
plot(sig_SR);
title('Original Noisy Signal');



% Create a new figure for denoised signals
% Plot original and denoised signals
figure;

subplot(3,1,1);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_wavelet, 'Color', 'red');
hold off;
title('Denoised Signal (Wavelet)');
legend({'Original Signal', ['Denoised Signal ', rmse_wavelet_str]}, 'Location', 'best');

subplot(3,1,2);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(y, 'Color', 'red');
hold off;
title('Denoised Signal (LMS MATLAB)');
legend({'Original Signal', ['Denoised Signal ', rmse_lms_str]}, 'Location', 'best');

subplot(3,1,3);
plot(sig_HR, 'Color', 'blue');
hold on;
plot(sig_rec_aligned, 'Color', 'red');
hold off;
title('Denoised Signal (SR3)');
legend({'Original Signal', ['Denoised Signal ', rmse_SR3_str]}, 'Location', 'best');






