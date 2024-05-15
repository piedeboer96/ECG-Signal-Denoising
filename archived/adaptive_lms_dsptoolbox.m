%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the clean signal
data_HR = load('sig_HR.mat');
sig_HR = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
sig_SR = data_SR.sig_SR;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Step 1: Create LMS Filter Object
lms = dsp.LMSFilter('Method', 'LMS', 'Length', 32, 'StepSize', 0.0276); % setting from paper
lms_default = dsp.LMSFilter();

% Step 2: Provide Input and Desired Signals
x = sig_SR';

% Assuming 'desired' is your desired signal (reference or clean version)
d = sig_HR';

% Step 3: Filtering Process
[y, err, wts] = lms(x, d);           % settings from paper

[y1, err1, wts1] = lms_default(x,d); % default settings

% Step 4: Access Results
% 'y' contains the filtered output
% 'err' contains the error between the filtered output and desired signal
% 'wts' contains the adapted filter weights

% Plotting
figure;
subplot(4,1,1);
plot(sig_HR);
title('Original Signal (sig\_HR)');
xlabel('Sample');
ylabel('Amplitude');

subplot(4,1,2);
plot(sig_SR);
title('Noisy Signal (sig\_SR)');
xlabel('Sample');
ylabel('Amplitude');

subplot(4,1,3);
plot(y);
title('Cleaned Signal (Filtered Output)');
xlabel('Sample');
ylabel('Amplitude');

subplot(4,1,4);
plot(y1);
title('Cleaned Signal (Filtered Output)');
xlabel('Sample');
ylabel('Amplitude');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a

rmse_lms_paper = rmse(d, y);
rmse_lms_default = rmse(d, y1);

% Print RMSE values
fprintf('RMSE for LMS Filter (Paper): %.4f\n', rmse_lms_paper);
fprintf('RMSE for LMS Filter (Default): %.4f\n', rmse_lms_default);     % uses properietary stepsize source