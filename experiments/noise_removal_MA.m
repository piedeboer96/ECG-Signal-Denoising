% LITERATURE
%  - Rangraj M. Rangayyan. (2002) Biomedical Signal Analysis: A case study approach. John Wiley & Sons, Inc., ISBN: 0-471- 20811-6.
%  - Bhaskara, P. C., & Uplane, M. D. (2016). High Frequency Electromyogram Noise Rmaoval from Electrocardiogram Using FIR Low Pass Filter Based On FPGA. 2016.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
sig_HR = load('results/ardb/MA/m1_ma_snr_1/sig_HR.mat').sig_HR; 
d = sig_HR; 

% Noisy Signals at SNR 1,3,5
sig_SR_snr_1 = load('results/ardb/MA/m1_ma_snr_1/sig_SR.mat').sig_SR;  
x0 = sig_SR_snr_1;
sig_SR_snr_3 = load('results/ardb/MA/m1_ma_snr_3/sig_SR.mat').sig_SR; 
x1 = sig_SR_snr_3;
sig_SR_snr_5 = load('results/ardb/MA/m1_ma_snr_5/sig_SR.mat').sig_SR; 
x2 = sig_SR_snr_5;

% Denoised Signals - Model 1
sig_REC_m1_snr_1 = load('results/ardb/MA/m1_ma_snr_1/sig_rec.mat').sig_rec;
sig_REC_m1_snr_3 = load('results/ardb/MA/m1_ma_snr_3/sig_rec.mat').sig_rec;
sig_REC_m1_snr_5 = load('results/ardb/MA/m1_ma_snr_5/sig_rec.mat').sig_rec;
 
% Denoised Signals - Model 2
sig_REC_m2_snr_1 = load('results/ardb/MA/m2_ma_snr_1/sig_rec.mat').sig_rec;
sig_REC_m2_snr_3 = load('results/ardb/MA/m2_ma_snr_3/sig_rec.mat').sig_rec;
sig_REC_m2_snr_5 = load('results/ardb/MA/m2_ma_snr_5/sig_rec.mat').sig_rec;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to vertically align signals based on mean difference
function aligned_signal = vertical_align_by_mean(clean_signal, noisy_signal)
    mean_clean = mean(clean_signal);
    mean_noisy = mean(noisy_signal);
    mean_diff = mean_clean - mean_noisy;
    aligned_signal = noisy_signal + mean_diff;
end

% Aligning the noisy signals vertically to the clean signal
x0 = vertical_align_by_mean(d, x0);
x1 = vertical_align_by_mean(d, x1);
x2 = vertical_align_by_mean(d, x2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to vertically align reconstructed signals based on mean difference
function aligned_signal = vertical_align_reconstructed(clean_signal, rec_signal)
    vertical_shift_rec = mean(clean_signal) - mean(rec_signal);
    aligned_signal = rec_signal + vertical_shift_rec;
    aligned_signal = double(aligned_signal'); % Transpose and convert to double
end

% Aligning the reconstructed signals vertically to the clean signal

% Model 1
aligned_REC_m1_snr_1 = vertical_align_reconstructed(d, sig_REC_m1_snr_1);
aligned_REC_m1_snr_3 = vertical_align_reconstructed(d, sig_REC_m1_snr_3);
aligned_REC_m1_snr_5 = vertical_align_reconstructed(d, sig_REC_m1_snr_5);

% Model 2
aligned_REC_m2_snr_1 = vertical_align_reconstructed(d, sig_REC_m2_snr_1);
aligned_REC_m2_snr_3 = vertical_align_reconstructed(d, sig_REC_m2_snr_3);
aligned_REC_m2_snr_5 = vertical_align_reconstructed(d, sig_REC_m2_snr_5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% METHOD 1: Moving Average Filter

% Define the coefficients of the transfer function H(z)
b = 1/4 * ones(1, 4);  % Numerator coefficients (1/8 * (1 - z^-8))  % UPDATED order=4
a = 1;                 % Denominator coefficients (1 - z^-1)

% Apply filter
y0_MA = filter(b, a, x0); 
y1_MA = filter(b, a, x1);
y2_MA = filter(b, a, x2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% METHOD 2: Low Pass Filter

% Given parameters
fs = 128; % Sampling rate (Hz)
fc = 40.60449; % Cutoff frequency (Hz)  - adapted for changed sampling frequency
N = 14; % Filter length
phi_d = 0.1047; % Phase delay (rad/Hz)

% Normalize the cutoff frequency
fc_norm = fc / fs;

% Design the FIR filter using the Kaiser window method
beta = 0; % Kaiser window parameter
fir_coeffs = fir1(N-1, fc_norm, 'low', kaiser(N, beta));

% Apply filter and compensate for the delay
delay = floor((N-1) / 2);

% Ensure x0, x1, x2 are column vectors for consistency
x0 = x0(:);
x1 = x1(:);
x2 = x2(:);

% Apply filter
y0_LPF = filter(fir_coeffs, 1, [x0; zeros(delay, 1)]);
y0_LPF = y0_LPF(delay+1:end);
y0_LPF = y0_LPF';

y1_LPF = filter(fir_coeffs, 1, [x1; zeros(delay, 1)]);
y1_LPF = y1_LPF(delay+1:end);
y1_LPF = y1_LPF';

y2_LPF = filter(fir_coeffs, 1, [x2; zeros(delay, 1)]);
y2_LPF = y2_LPF(delay+1:end);
y2_LPF = y2_LPF';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute MAE for MA outputs
MAE_MA_snr_1 = mean(abs(d - y0_MA));
MAE_MA_snr_3 = mean(abs(d - y1_MA));
MAE_MA_snr_5 = mean(abs(d - y2_MA));

% Compute MAE for LPF outputs
MAE_LFP_snr_1 = mean(abs(d-y0_LPF));
MAE_LFP_snr_3 = mean(abs(d-y1_LPF));
MAE_LFP_snr_5 = mean(abs(d-y2_LPF));

% Compute MAE for Model 1 outputs
MAE_Model1_snr_1 = mean(abs(d' - aligned_REC_m1_snr_1'));
MAE_Model1_snr_3 = mean(abs(d' - aligned_REC_m1_snr_3'));
MAE_Model1_snr_5 = mean(abs(d' - aligned_REC_m1_snr_5'));

% Compute MAE for Model 2 outputs
MAE_Model2_snr_1 = mean(abs(d' - aligned_REC_m2_snr_1'));
MAE_Model2_snr_3 = mean(abs(d' - aligned_REC_m2_snr_3'));
MAE_Model2_snr_5 = mean(abs(d' - aligned_REC_m2_snr_5'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize

figure;

% Model 1 outputs
subplot(4, 3, 1);
plot(d);
hold on;
plot(aligned_REC_m1_snr_1);
title('Model 1 Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_1), ')']);
grid on;

subplot(4, 3, 2);
plot(d);
hold on;
plot(aligned_REC_m1_snr_3);
title('Model 1 Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_3), ')']);
grid on;

subplot(4, 3, 3);
plot(d);
hold on;
plot(aligned_REC_m1_snr_5);
title('Model 1 Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_5), ')']);
grid on;

% Model 2 outputs
subplot(4, 3, 4);
plot(d);
hold on;
plot(aligned_REC_m2_snr_1);
title('Model 2 Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_1), ')']);
grid on;

subplot(4, 3, 5);
plot(d);
hold on;
plot(aligned_REC_m2_snr_3);
title('Model 2 Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_3), ')']);
grid on;

subplot(4, 3, 6);
plot(d);
hold on;
plot(aligned_REC_m2_snr_5);
title('Model 2 Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_5), ')']);
grid on;

% Moving Average Filter outputs
subplot(4, 3, 7);
plot(d);
hold on;
plot(y0_MA);
title('MA Filter Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_MA_snr_1), ')']);
grid on;

subplot(4, 3, 8);
plot(d);
hold on;
plot(y1_MA);
title('MA Filter Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_MA_snr_3), ')']);
grid on;

subplot(4, 3, 9);
plot(d);
hold on;
plot(y2_MA);
title('MA Filter Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_MA_snr_5), ')']);
grid on;

% Low Pass Filter outputs
subplot(4, 3, 10);
plot(d);
hold on;
plot(y0_LPF);
title('LPF Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LFP_snr_1), ')']);
grid on;

subplot(4, 3, 11);
plot(d);
hold on;
plot(y1_LPF);
title('LPF Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LFP_snr_3), ')']);
grid on;

subplot(4, 3, 12);
plot(d);
hold on;
plot(y2_LPF);
title('LPF Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LFP_snr_5), ')']);
grid on;

sgtitle('Comparison of Original and Denoised Signals Across Different Methods and SNR Values');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%