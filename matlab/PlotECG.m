%% Visualize ECG signal


% Time Domain Plot
fid = fopen("nstdb/118e24.dat")
time=10;
f=fread(fid,2*360*time,'ubit12');
Orig_Sig=f(1:2:length(f));
plot(Orig_Sig)
title('Time-Domain Plot');
xlabel('Samples');
ylabel('Amplitude');



% Define STFT parameters
% fs=360;
% window_length = 50; % Length of the window (adjust as needed)
% overlap = 0.8; % Overlap factor (adjust as needed)

% Compute STFT
% [stft, f, t] = spectrogram(Orig_Sig, window_length, round(overlap*window_length), [], fs);
x=Orig_Sig;
M = 49;
L = 11;
g = bartlett(M);
Ndft = 1024;

[stft,f,t] = spectrogram(x,g,L,Ndft,fs);
% [stft,f,t] = spectogram(Orig_Sig);


% Plot STFT
figure;
imagesc(t, f, abs(stft));
axis xy; % Flip the y-axis to have low frequencies at the bottom
colormap(jet); % Use a colormap for better visualization
colorbar; % Add colorbar for intensity scale
title('STFT of ECG Signal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');



