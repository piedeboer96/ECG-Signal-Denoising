% Load the clean signal
data_HR = load('run_2/sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('run_2/sig_SR.mat');
sig_SR = data_SR.sig_SR;

% Apply Savitzky-Golay filtering
polynomial_order = 8;
frame_size = 31;

% Perform Savitzky-Golay filtering on sig_SR
denoised_SR = sgolayfilt(sig_SR, polynomial_order, frame_size);

% Plot clean HR signal, denoised SR signal, and noisy SR signal
figure;
subplot(3,1,1);
plot(sig_HR, 'b');
title('Clean HR Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,2);
plot(denoised_SR, 'r');
title('Denoised SR Signal (Savitzky-Golay)');
xlabel('Sample');
ylabel('Amplitude');

subplot(3,1,3);
plot(sig_SR, 'g');
title('Noisy SR Signal');
xlabel('Sample');
ylabel('Amplitude');

