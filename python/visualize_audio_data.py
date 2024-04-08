import librosa
import matplotlib.pyplot as plt
from scipy.signal import stft, cwt
import numpy as np
import pywt

path_to_audio_clip = "/Users/piedeboer/Desktop/Projects/Thesis/03-research-phase/local/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"

# Load audio sample
audio_clip, sr = librosa.load(path_to_audio_clip, sr=None)

# Show audio sample
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(audio_clip)
plt.title('Audio Clip')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Perform STFT
f_stft, t_stft, Zxx = stft(audio_clip, fs=sr)
plt.subplot(3, 1, 2)
plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud')
plt.title('STFT')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Magnitude')

# Perform CWT
scales = np.arange(1, 100)
cwt_result, _ = pywt.cwt(audio_clip, scales, 'mexh')  # Extract coefficients array
plt.subplot(3, 1, 3)
plt.imshow(np.abs(cwt_result), aspect='auto', extent=[0, len(audio_clip), scales[-1], scales[0]])
plt.title('CWT')
plt.xlabel('Sample')
plt.ylabel('Scale')

# Adjust layout and display
plt.tight_layout()
plt.show()
