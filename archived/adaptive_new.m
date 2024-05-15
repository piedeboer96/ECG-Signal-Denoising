% Define parameters
N = 32;         % Filter length
mu = 0.0276;    % Step size (learning rate)

% Given noisy signal (assuming it's stored in a variable called 'noisy_signal')
% Load noisy signal
data_SR = load('run_2/sig_SR.mat');
noisy_signal = data_SR.sig_SR';

% Initialize filter coefficients to zeros
h = zeros(N, 1);

% Apply LMS algorithm
for i = N:length(noisy_signal)
    % Extract the current input vector (window of noisy signal)
    x = noisy_signal(i:-1:i-N+1);
    
    % Compute the filter output
    y = h' * x;
    
    % Compute the error
    e = noisy_signal(i) - y;
    
    % Update filter coefficients using LMS update rule
    h = h + mu * e * x;
end

% Now 'h' contains the optimized filter coefficients
% Apply LMS filter to the entire noisy signal
filtered_signal = filter(h, 1, noisy_signal);

% Plot the original noisy signal and the filtered signal
figure;
plot(noisy_signal, 'g', 'DisplayName', 'Noisy Signal');
hold on;
plot(filtered_signal, 'r', 'DisplayName', 'Filtered Signal');
hold off;
xlabel('Sample');
ylabel('Amplitude');
title('Original Noisy Signal vs Filtered Signal');
legend;
lms = dsp.LMSFilter

disp('LMS')

disp(lms)


lms(x)