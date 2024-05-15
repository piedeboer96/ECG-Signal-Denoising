% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR;

% % Apply wavelet denoising using 'SureShrink'
% wavelet_type = 'sym12'; % Mother wavelet function
% lev = 3; % Number of decomposition levels
% 
% % Perform wavelet denoising on sig_SR using 'SureShrink'
% denoised_SR = wdenoise(sig_SR, lev, 'Wavelet', wavelet_type, 'DenoisingMethod', 'SURE', 'ThresholdRule', 'Soft');
% 

% % Apply wavelet denoising using 'SureShrink'
wavelet_type = 'sym6'; % Mother wavelet function
lev = 2; % Number of decomposition levels

% Perform wavelet denoising on sig_SR using 'SureShrink'
denoised_SR = wdenoise(sig_SR, lev, 'Wavelet', wavelet_type, 'ThresholdRule', 'Soft');




% Plot clean HR signal, denoised SR signal, and noisy SR signal
figure;
subplot(3,1,1);
plot(sig_HR, 'b');
title('Clean HR Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,2);
plot(denoised_SR, 'r');
title('Denoised SR Signal (Wavelet)');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,3);
plot(sig_SR, 'g');
title('Noisy SR Signal');
xlabel('Sample');
ylabel('Amplitude');

% Compute the loss (Mean Squared Error)
loss = immse(sig_HR, denoised_SR);
disp(['Mean Squared Error between Clean HR and Denoised SR Signals: ', num2str(loss)]);

loss_sr_hr = immse(sig_HR, sig_SR)
disp(['Mean Squared Error between Clean HR and Signal SR Signals: ', num2str(loss_sr_hr)]);
