% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR;

noisy_signal = sig_SR;
clean_signal = sig_HR;

% Parameters
N = 32; % Filter length
mu = 0.0276; % Step size

% Construct the desired signal (clean signal)
d = clean_signal(N:end);

% Construct the input signal (noisy signal)
x = zeros(N, length(d));
for i = 1:N
    x(i, :) = noisy_signal(N-i+1:end-i+1);
end

% Initialize the filter coefficients
w = zeros(N, 1);

% LMS algorithm
for i = 1:length(d)
    y = w' * x(:, i); % Estimated output
    e = d(i) - y; % Error
    w = w + mu * e * x(:, i); % Update filter coefficients
end

% Filter the noisy signal
filtered_signal = filter(w, 1, noisy_signal);

% Plot the results
figure;
subplot(3,1,1);
plot(clean_signal);
title('Clean Signal');
subplot(3,1,2);
plot(noisy_signal);
title('Noisy Signal');
subplot(3,1,3);
plot(filtered_signal);
title('Filtered Signal (LMS)');


save('sig_LMS.mat', 'filtered_signal');