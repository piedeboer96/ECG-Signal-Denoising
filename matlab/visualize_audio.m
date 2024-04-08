% Load audio clip
filename = '/Users/piedeboer/Desktop/Thesis/03-research-phase/local-code/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'; % Provide the path to your audio file
[y, Fs] = audioread(filename); % Read audio file and get sampling frequency

% Parameters for STFT
window_size = 1024; % Window size for STFT
hop_size = 512; % Hop size for STFT

% Perform STFT
[S, F, T] = spectrogram(y, hamming(window_size), window_size - hop_size, window_size, Fs);

% Plot spectrogram
figure;
imagesc(T, F, 20*log10(abs(S))); % Convert to dB for better visualization
axis xy; % Flip the y-axis
colormap(jet); % Choose colormap
colorbar; % Add colorbar
title('Spectrogram of Audio Clip');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
