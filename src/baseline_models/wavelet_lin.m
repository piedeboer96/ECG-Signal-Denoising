% Assuming you have your signals sig_SR (noisy) and sig_HR (clean)
% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR;


% Set parameters
wavelet_type = 'sym5'; % Symlets5
level = 2; % Decomposition level
threshold_rule = 'sqtwolog'; % Soft thresholding rule

% Denoise the noisy signal using wavelet denoising
denoised_sig = wdenoise(sig_SR, level, 'Wavelet', wavelet_type, 'ThresholdRule', 's');

% Plot original and denoised signals
figure;
subplot(2,1,1);
plot(sig_SR);
title('Noisy ECG Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2,1,2);
plot(denoised_sig);
title('Denoised ECG Signal');
xlabel('Sample');
ylabel('Amplitude');

% Assuming you have the clean signal sig_HR for comparison
figure;
subplot(2,1,1);
plot(sig_HR);
title('Clean ECG Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(2,1,2);
plot(denoised_sig);
hold on;
plot(sig_SR);
title('Comparison: Denoised vs. Noisy');
xlabel('Sample');
ylabel('Amplitude');
legend('Denoised Signal', 'Noisy Signal');

