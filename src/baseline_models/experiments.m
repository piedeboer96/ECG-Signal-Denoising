%% BaslineModels vs SR3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR; 

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR; 

% Load denoised signal from SR3
data_rec = load('sig_rec.mat');
sig_rec = data_rec.sig_rec;                 % denoise SR3 model

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


% Compute DTW for sig_wavelet
dtw_wavelet = dtw(sig_HR, sig_wavelet);
dtw_lms = dtw(sig_HR, y);
dtw_SR3 = dtw(sig_HR, sig_rec_aligned);



% Define the data for the table
data = {
    rmse_wavelet_str, psnr_wavelet, dtw_wavelet;
    rmse_lms_str, psnr_lms, dtw_lms;   
    rmse_SR3_str, psnr_SR3, dtw_SR3
};

% Define the row and column names
row_names = {'Wavelet (Paper)','LMS (MATLAB)', 'SR3'};
col_names = {'RMSE (lower better)', 'PSNR (higher better)', 'DTW (lower better)'};

% Create the table
table_results = array2table(data, 'RowNames', row_names, 'VariableNames', col_names);

% Display the table
disp(table_results);

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






