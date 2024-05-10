% Assuming you have loaded your signals sig_SR (noisy) and sig_HR (clean)
% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR;

% Parameters for wavelet denoising
level = 5; % Level of wavelet decomposition
wname = 'db4'; % Wavelet name

% Perform wavelet denoising
denoised_signal = wdenoise(sig_SR, level, 'Wavelet', wname);

% Plot original and denoised signals
figure;
subplot(2,1,1);
plot(sig_SR);
title('Original Noisy Signal');
subplot(2,1,2);
plot(denoised_signal);
title('Denoised Signal');

% Optionally, compute and display signal-to-noise ratio (SNR)
snr_value = snr(sig_HR, sig_SR - denoised_signal);
disp(['Signal-to-Noise Ratio (SNR): ', num2str(snr_value)]);
