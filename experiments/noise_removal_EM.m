%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LITERATURE
%   - Marzog, H. A., Abd, H. J., & Yonis, A. Z. (2022). Noise removal of ECG
%  signal using multi-techniques. IEEE. --> ADAPTIVE FILTER LMS AND RLS 
%  - M. Milanesi, N. Martini, N. Vanello, V. Positano, M. F. Santarelli, R.
%  Paradiso, D. De Rossi and L. Landini. (2006) Multichannel Techniques for
%  Motion Artifacts Removal from Electrocardiographic Signals --> ADAPTIVE FILLTER LMS
%  - Kher, R. (2019). Signal Processing Techniques for Removing Noise from
%  ECG Signals. 2019. --> ADAPTIVE FILTER LMS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean Signal
sig_HR = load('results/ardb/EM/m1_em_snr_3/sig_HR.mat').sig_HR; 
d = sig_HR; 

% Noisy Signals at SNR 1,3,5
sig_SR_snr_1 = load('results/ardb/EM/m1_em_snr_1/sig_SR.mat').sig_SR;  
x0 = sig_SR_snr_1;
sig_SR_snr_3 = load('results/ardb/EM/m1_em_snr_3/sig_SR.mat').sig_SR; 
x1 = sig_SR_snr_3;
sig_SR_snr_5 = load('results/ardb/EM/m1_em_snr_5/sig_SR.mat').sig_SR; 
x2 = sig_SR_snr_5;

% Denoised Signals - Model 1
sig_REC_m1_snr_1 = load('results/ardb/EM/m1_em_snr_1/sig_rec.mat').sig_rec;
sig_REC_m1_snr_3 = load('results/ardb/EM/m1_em_snr_3/sig_rec.mat').sig_rec;
sig_REC_m1_snr_5 = load('results/ardb/EM/m1_em_snr_5/sig_rec.mat').sig_rec;
 
% Denoised Signals - Model 2
sig_REC_m2_snr_1 = load('results/ardb/EM/m2_em_snr_1/sig_rec.mat').sig_rec;
sig_REC_m2_snr_3 = load('results/ardb/EM/m2_em_snr_3/sig_rec.mat').sig_rec;
sig_REC_m2_snr_5 = load('results/ardb/EM/m2_em_snr_5/sig_rec.mat').sig_rec;

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LMS Adaptive Filter

lms = dsp.LMSFilter();  % MATLAB proprietary LMS filter

[y0, err0, wts0] = lms(x0', d');
[y1, err1, wts1] = lms(x1', d');
[y2, err2, wts2] = lms(x2', d');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute MAE for LMS outputs
MAE_LMS_snr_1 = mean(abs(d' - y0));
MAE_LMS_snr_3 = mean(abs(d' - y1));
MAE_LMS_snr_5 = mean(abs(d' - y2));

% Compute MAE for Model 1 outputs
MAE_Model1_snr_1 = mean(abs(d' - aligned_REC_m1_snr_1'));
MAE_Model1_snr_3 = mean(abs(d' - aligned_REC_m1_snr_3'));
MAE_Model1_snr_5 = mean(abs(d' - aligned_REC_m1_snr_5'));

% Compute MAE for Model 2 outputs
MAE_Model2_snr_1 = mean(abs(d' - aligned_REC_m2_snr_1'));
MAE_Model2_snr_3 = mean(abs(d' - aligned_REC_m2_snr_3'));
MAE_Model2_snr_5 = mean(abs(d' - aligned_REC_m2_snr_5'));

% Compute RMSE for LMS outputs
RMSE_LMS_snr_1 = sqrt(mean((d' - y0).^2));
RMSE_LMS_snr_3 = sqrt(mean((d' - y1).^2));
RMSE_LMS_snr_5 = sqrt(mean((d' - y2).^2));

% Compute RMSE for Model 1 outputs
RMSE_Model1_snr_1 = sqrt(mean((d' - aligned_REC_m1_snr_1').^2));
RMSE_Model1_snr_3 = sqrt(mean((d' - aligned_REC_m1_snr_3').^2));
RMSE_Model1_snr_5 = sqrt(mean((d' - aligned_REC_m1_snr_5').^2));

% Compute RMSE for Model 2 outputs
RMSE_Model2_snr_1 = sqrt(mean((d' - aligned_REC_m2_snr_1').^2));
RMSE_Model2_snr_3 = sqrt(mean((d' - aligned_REC_m2_snr_3').^2));
RMSE_Model2_snr_5 = sqrt(mean((d' - aligned_REC_m2_snr_5').^2));

% Compute PSNR for LMS outputs
PSNR_LMS_snr_1 = 20 * log10(max(d) / RMSE_LMS_snr_1);
PSNR_LMS_snr_3 = 20 * log10(max(d) / RMSE_LMS_snr_3);
PSNR_LMS_snr_5 = 20 * log10(max(d) / RMSE_LMS_snr_5);

% Compute PSNR for Model 1 outputs
PSNR_Model1_snr_1 = 20 * log10(max(d) / RMSE_Model1_snr_1);
PSNR_Model1_snr_3 = 20 * log10(max(d) / RMSE_Model1_snr_3);
PSNR_Model1_snr_5 = 20 * log10(max(d) / RMSE_Model1_snr_5);

% Compute PSNR for Model 2 outputs
PSNR_Model2_snr_1 = 20 * log10(max(d) / RMSE_Model2_snr_1);
PSNR_Model2_snr_3 = 20 * log10(max(d) / RMSE_Model2_snr_3);
PSNR_Model2_snr_5 = 20 * log10(max(d) / RMSE_Model2_snr_5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plotting LMS outputs
subplot(3,3,1);
plot(d);
hold on;
plot(y0);
title('LMS Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LMS_snr_1), ')']);

subplot(3,3,2);
plot(d);
hold on;
plot(y1);
title('LMS Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LMS_snr_3), ')']);

subplot(3,3,3);
plot(d);
hold on;
plot(y2);
title('LMS Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_LMS_snr_5), ')']);

% Plotting Model 1 outputs
subplot(3,3,4);
plot(d);
hold on;
plot(aligned_REC_m1_snr_1);
title('Model 1 Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_1), ')']);

subplot(3,3,5);
plot(d);
hold on;
plot(aligned_REC_m1_snr_3);
title('Model 1 Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_3), ')']);

subplot(3,3,6);
plot(d);
hold on;
plot(aligned_REC_m1_snr_5);
title('Model 1 Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model1_snr_5), ')']);

% Plotting Model 2 outputs
subplot(3,3,7);
plot(d);
hold on;
plot(aligned_REC_m2_snr_1);
title('Model 2 Output at SNR 1');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_1), ')']);

subplot(3,3,8);
plot(d);
hold on;
plot(aligned_REC_m2_snr_3);
title('Model 2 Output at SNR 3');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_3), ')']);

subplot(3,3,9);
plot(d);
hold on;
plot(aligned_REC_m2_snr_5);
title('Model 2 Output at SNR 5');
legend('Original Signal', ['Denoised Signal (MAE: ', num2str(MAE_Model2_snr_5), ')']);

% Adjusting layout
sgtitle('Comparison of Original and Denoised Signals');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define SNR values
snr_values = [1, 3, 5];

% Define MAE values for different models at different SNRs
MAE_values = [
MAE_LMS_snr_1, MAE_Model1_snr_1, MAE_Model2_snr_1;
MAE_LMS_snr_3, MAE_Model1_snr_3, MAE_Model2_snr_3;
MAE_LMS_snr_5, MAE_Model1_snr_5, MAE_Model2_snr_5;
];

% Define RMSE values for different models at different SNRs
RMSE_values = [
RMSE_LMS_snr_1, RMSE_Model1_snr_1, RMSE_Model2_snr_1;
RMSE_LMS_snr_3, RMSE_Model1_snr_3, RMSE_Model2_snr_3;
RMSE_LMS_snr_5, RMSE_Model1_snr_5, RMSE_Model2_snr_5;
];

% Define PSNR values for different models at different SNRs
PSNR_values = [
PSNR_LMS_snr_1, PSNR_Model1_snr_1, PSNR_Model2_snr_1;
PSNR_LMS_snr_3, PSNR_Model1_snr_3, PSNR_Model2_snr_3;
PSNR_LMS_snr_5, PSNR_Model1_snr_5, PSNR_Model2_snr_5;
];

% Plot grouped bar chart for MAE
figure;
subplot(3,1,1);
bar(snr_values, MAE_values);
title('MAE Comparison at Different SNRs');
xlabel('SNR Value');
ylabel('MAE');
legend('LMS', 'Model 1', 'Model 2');
grid on;

% Plot grouped bar chart for RMSE
subplot(3,1,2);
bar(snr_values, RMSE_values);
title('RMSE Comparison at Different SNRs');
xlabel('SNR Value');
ylabel('RMSE');
legend('LMS', 'Model 1', 'Model 2');
grid on;

% Plot grouped bar chart for PSNR
subplot(3,1,3);
bar(snr_values, PSNR_values);
title('PSNR Comparison at Different SNRs');
xlabel('SNR Value');
ylabel('PSNR');
legend('LMS', 'Model 1', 'Model 2');
grid on;

sgtitle('Performance Metrics Comparison at Different SNR Values');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%