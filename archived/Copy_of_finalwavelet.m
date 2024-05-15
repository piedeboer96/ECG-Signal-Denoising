%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the clean signal
data_HR = load('run_2/sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('run_2/sig_SR.mat');
sig_SR = data_SR.sig_SR; 
x = sig_SR';

% Load denoised signal from SR3
data_rec = load('run_2/sig_rec.mat');
sig_rec = data_rec.sig_rec;                 % denoise SR3 model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Wavelet denoising
sig_wavelet = wdenoise(sig_SR, 3, 'Wavelet', 'sym12'); % settings from Paper

% Savetzky golay filtering
sig_sg = sgolayfilt(sig_SR, 8, 31);                    % from paper

% Adaptive Filter
lms_default = dsp.LMSFilter();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the vertical shift needed to align the reconstructed signal with the clean signal
vertical_shift_rec = mean(sig_HR) - mean(sig_rec);

% Apply the vertical shift to the reconstructed signal
sig_rec_aligned = sig_rec + vertical_shift_rec;
sig_rec_aligned = double(sig_rec_aligned');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rmse_wavelet = rmse(sig_HR, sig_wavelet)
rmse_SR3 = rmse(sig_HR, sig_rec_aligned)

% Convert RMSE values to strings formatted to three decimals
rmse_wavelet_str = sprintf('RMSE: %.4f', rmse_wavelet);
rmse_SR3_str = sprintf('RMSE: %.4f', rmse_SR3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot original and denoised signals
figure;
sgtitle('Comparison of Original and Denoised Signals with Muscle Artifact (MA) at SNR 3');


subplot(6,1,1);
plot(sig_HR);
title('Original Signal');

subplot(6,1,2);
plot(sig_SR);
title('Original Noisy Signal');

subplot(6,2,1);
plot(sig_HR, 'Color', 'blue'); % Overlay original signal in blue
hold on;
plot(sig_wavelet, 'Color', 'red'); % Denoised signal in red
hold off;
title('Denoised Signal (Level 3, sym12)');
legend({'Original Signal', ['Denoised Signal ', rmse_wavelet_str]}, 'Location', 'best');


plot(sig_HR, 'Color', 'blue'); % Overlay original signal in blue
hold on;
plot(sig_rec_aligned, 'Color', 'green'); % Denoised signal in green
hold off;
title('SR3 Model');
legend({'Original Signal', ['Denoised Signal ', rmse_SR3_str]}, 'Location', 'best');


% plot(sig_HR, 'Color', 'blue'); % Overlay original signal in blue
% hold on;
% plot(sig_rec_aligned, 'Color', 'magneta'); % Denoised signal in green
% hold off;
% title('SR3 Model');
% legend({'Original Signal', ['Denoised Signal ', rmse_SR3_str]}, 'Location', 'best');


