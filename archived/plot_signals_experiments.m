% Plotting
figure;

% ARDB Clean Signal
subplot(3, 2, 1);
plot(ardb_clean(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Clean Signal from ARDB');
xlabel('Samples');
ylabel('Amplitude');

% ARDB Composite Noise SNR 3
subplot(3, 2, 3);
plot(ardb_snr_3(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Composite Noised Added (SNR 3) ARDB');
xlabel('Samples');
ylabel('Amplitude');

% ARDB Composite Noise SNR 5
subplot(3, 2, 5);
plot(ardb_snr_5(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Composite Noised Added (SNR 5) ARDB');
xlabel('Samples');
ylabel('Amplitude');

% AF Clean Signal
subplot(3, 2, 2);
plot(af_clean(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Clean Signal from AF');
xlabel('Samples');
ylabel('Amplitude');

% AF Composite Noise SNR 3
subplot(3, 2, 4);
plot(af_snr_3(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Composite Noised Added (SNR 3) AF');
xlabel('Samples');
ylabel('Amplitude');

% AF Composite Noise SNR 5
subplot(3, 2, 6);
plot(af_snr_5(1:128)); % Limit x-axis to 128 samples
xlim([1, 128]); % Limit x-axis to 128 samples
title('Composite Noised Added (SNR 5) AF');
xlabel('Samples');
ylabel('Amplitude');
