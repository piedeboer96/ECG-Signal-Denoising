% Load the clean signal
data_HR = load('sig_HR.mat');
clean_signal = data_HR.sig_HR;

% Load the noisy signal
data_SR = load('sig_SR.mat');
noisy_signal = data_SR.sig_SR;

% Parameters for denoising
mother_wavelet = 'sym12'; % Mother wavelet function
lev = 3; % Number of decomposition levels
thresholding_method = 's'; % 's' for soft thresholding, 'h' for hard thresholding

% Perform DWT
[C, L] = wavedec(noisy_signal, lev, mother_wavelet);

% Estimate the noise standard deviation using the universal threshold
sigma = median(abs(C))/0.6745;

% Set threshold for soft or hard thresholding
if thresholding_method == 's'
    threshold = sqrt(2*log(length(noisy_signal)))*sigma;
else
    threshold = 3*sigma;
end

% Apply thresholding to the detail coefficients
for i = 1:lev
    start_index = sum(L(1:i)) + 1;
    end_index = sum(L(1:i+1));
    detail_coefficients = C(start_index:end_index);
    C(start_index:end_index) = wthresh(detail_coefficients, threshold, thresholding_method);
end

% Reconstruct the denoised signal
denoised_signal = waverec(C, L, mother_wavelet);

% Plot the results for comparison
figure;
subplot(3,1,1);
plot(clean_signal);
title('Original Clean Signal');
subplot(3,1,2);
plot(noisy_signal);
title('Noisy Signal');
subplot(3,1,3);
plot(denoised_signal);
title('Denoised Signal');
