% LITERATURE
% - Review of noise removal techniques in ECG signals 2022

% Clean Signal
sig_HR = load('results/ardb/COMPOSITE/m1_comp_snr_1/sig_HR.mat').sig_HR; 
d = sig_HR; 

% Noisy Signals at SNR 1, 3, 5
sig_SR_snr_1 = load('results/ardb/COMPOSITE/m1_comp_snr_1/sig_SR.mat').sig_SR;  
x0 = sig_SR_snr_1;
sig_SR_snr_3 = load('results/ardb/COMPOSITE/m1_comp_snr_3/sig_SR.mat').sig_SR; 
x1 = sig_SR_snr_3;
sig_SR_snr_5 = load('results/ardb/COMPOSITE/m1_comp_snr_5/sig_SR.mat').sig_SR; 
x2 = sig_SR_snr_5;

% Denoised Signals - Model 1
sig_REC_m1_snr_1 = load('results/ardb/COMPOSITE/m1_comp_snr_1/sig_rec.mat').sig_rec;
sig_REC_m1_snr_3 = load('results/ardb/COMPOSITE/m1_comp_snr_3/sig_rec.mat').sig_rec;
sig_REC_m1_snr_5 = load('results/ardb/COMPOSITE/m1_comp_snr_5/sig_rec.mat').sig_rec;
 
% Denoised Signals - Model 2
sig_REC_m2_snr_1 = load('results/ardb/COMPOSITE/m2_comp_snr_1/sig_rec.mat').sig_rec;
sig_REC_m2_snr_3 = load('results/ardb/COMPOSITE/m2_comp_snr_3/sig_rec.mat').sig_rec;
sig_REC_m2_snr_5 = load('results/ardb/COMPOSITE/m2_comp_snr_5/sig_rec.mat').sig_rec;

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
% METHOD 1: Wavelet Thresholding

y0_WT = wdenoise(x0, 'Wavelet', 'sym5'); % Paper 2020 (Review of noise removal techniques in ECG signals)
y1_WT = wdenoise(x1, 'Wavelet', 'sym5');
y2_WT = wdenoise(x1, 'Wavelet', 'sym5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% METHOD 2: Hybrid (LMS -> LPF)

lms = dsp.LMSFilter();  

y0_lms = lms(x0', d');
y1_lms = lms(x1', d');
y2_lms = lms(x2', d');

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
y0_lms = y0_lms(:);
y1_lms = y1_lms(:);
y2_lms = y2_lms(:);

% Apply filter
y0_HYB = filter(fir_coeffs, 1, [y0_lms; zeros(delay, 1)]);
y0_HYB = y0_HYB(delay+1:end);
y0_HYB = y0_HYB';

y1_HYB = filter(fir_coeffs, 1, [y1_lms; zeros(delay, 1)]);
y1_HYB = y1_HYB(delay+1:end);
y1_HYB = y1_HYB';

y2_HYB = filter(fir_coeffs, 1, [y2_lms; zeros(delay, 1)]);
y2_HYB = y2_HYB(delay+1:end);
y2_HYB = y2_HYB';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Compute MAE for Wavelet
MAE_DWT_snr_1 = mean(abs(d' - y0_WT'));
MAE_DWT_snr_3 = mean(abs(d' - y1_WT'));
MAE_DWT_snr_5 = mean(abs(d' - y2_WT'));

% Compute MAE for Hybrid
MAE_HYB_snr_1 = mean(abs(d' - y0_HYB'))
MAE_HYB_snr_3 = mean(abs(d' - y1_HYB'))
MAE_HYB_snr_5 = mean(abs(d' - y2_HYB'))

% Compute MAE for Model 1 outputs
MAE_Model1_snr_1 = mean(abs(d' - aligned_REC_m1_snr_1'));
MAE_Model1_snr_3 = mean(abs(d' - aligned_REC_m1_snr_3'));
MAE_Model1_snr_5 = mean(abs(d' - aligned_REC_m1_snr_5'));

% Compute MAE for Model 2 outputs
MAE_Model2_snr_1 = mean(abs(d' - aligned_REC_m2_snr_1'));
MAE_Model2_snr_3 = mean(abs(d' - aligned_REC_m2_snr_3'));
MAE_Model2_snr_5 = mean(abs(d' - aligned_REC_m2_snr_5'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting

figure;

% Denoised Signals - Wavelet Thresholding
subplot(4, 3, 1);
plot(d);
hold on;
plot(y0_WT);
title('Wavelet Thresholding - SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_DWT_snr_1), ')']);

subplot(4, 3, 2);
plot(d);
hold on;
plot(y1_WT);
title('Wavelet Thresholding - SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_DWT_snr_3), ')']);

subplot(4, 3, 3);
plot(d);
hold on;
plot(y2_WT);
title('Wavelet Thresholding - SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_DWT_snr_5), ')']);

% Denoised Signals - Hybrid (LMS -> LPF)
subplot(4, 3, 4);
plot(d);
hold on;
plot(y0_HYB);
title('Hybrid (LMS -> LPF) - SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_HYB_snr_1), ')']);

subplot(4, 3, 5);
plot(d);
hold on;
plot(y1_HYB);
title('Hybrid (LMS -> LPF) - SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_HYB_snr_3), ')']);

subplot(4, 3, 6);
plot(d);
hold on;
plot(y2_HYB);
title('Hybrid (LMS -> LPF) - SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_HYB_snr_5), ')']);

% Denoised Signals - Model 1
subplot(4, 3, 7);
plot(d);
hold on;
plot(aligned_REC_m1_snr_1);
title('Model 1 - SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_1), ')']);

subplot(4, 3, 8);
plot(d);
hold on;
plot(aligned_REC_m1_snr_3);
title('Model 1 - SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_3), ')']);

subplot(4, 3, 9);
plot(d);
hold on;
plot(aligned_REC_m1_snr_5);
title('Model 1 - SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_5), ')']);

% Denoised Signals - Model 2
subplot(4, 3, 10);
plot(d);
hold on;
plot(aligned_REC_m2_snr_1);
title('Model 2 - SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_1), ')']);

subplot(4, 3, 11);
plot(d);
hold on;
plot(aligned_REC_m2_snr_3);
title('Model 2 - SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_3), ')']);

subplot(4, 3, 12);
plot(d);
hold on;
plot(aligned_REC_m2_snr_5);
title('Model 2 - SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_5), ')']);


