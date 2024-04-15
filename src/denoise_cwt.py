import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio


# Two Approaches
##  1. use scikitlearn-image (not included)
##  --- : bayesShrink
##  2. using pywt cwt  
##  --- : matlab 'settings' :  https://de.mathworks.com/help/wavelet/ug/classify-time-series-using-wavelet-analysis-and-deep-learning.html

# ********************************
# ********************************

# Load ECG signal and annotation
record = wfdb.rdsamp('data/ardb/105', sampto=3000)

# Extract Lead I ECG signal
I = record[0][:, 0]

# Create scales for CWT
scales = np.arange(1, 100)

# Compute CWT for noisy and denoised signals
cwt_noisy, _ = pywt.cwt(I, scales, 'mexh')

# Plot original, noisy, and denoised signals
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax[0].plot(I)
ax[0].set_ylabel('Original')

ax[1].imshow(np.abs(cwt_noisy), aspect='auto', extent=[0, len(I), scales[-1], scales[0]])
ax[1].set_ylabel('Scale')
ax[1].set_title('CWT of Noisy Signal')

plt.tight_layout()
plt.show()

# ********************************
# ********************************

# Generate and plot scalograms
def plot_scalogram(I,Fs,samples):
    scales = np.arange(1, 100)
    cwt_result, frq = pywt.cwt(I, scales, 'morl', sampling_period=1/Fs)

    # Plot the scalogram
    t = np.arange(samples) / Fs
    plt.pcolormesh(t, frq, np.abs(cwt_result), shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Scalogram')
    plt.colorbar(label='Magnitude')

    # Set y-axis scale to logarithmic
    plt.yscale('log')

    # Set y-axis limits to show only the range between 0 and 200 Hz
    plt.ylim(0, 40)  

    plt.show()

# Visualize Multiple Scalograms
def plot_multiple_scalograms(signals, Fs_values, names):
    num_signals = len(signals)
    samples = len(signals[0])

    plt.figure(figsize=(5 * num_signals, 5))

    for i, (signal, Fs, name) in enumerate(zip(signals, Fs_values, names), 1):
        scales = np.arange(1, 100)
        cwt_result, frq = pywt.cwt(signal, scales, 'morl', sampling_period=1/Fs)

        # Plot the scalogram
        t = np.arange(samples) / Fs
        plt.subplot(1, num_signals, i)
        plt.pcolormesh(t, frq, np.abs(cwt_result), shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(name + ' Scalogram')
        plt.colorbar(label='Magnitude')

        # Set y-axis scale to logarithmic
        plt.yscale('log')

        # Set y-axis limits to show only the range between 0 and 200 Hz
        plt.ylim(0, 40)

    plt.tight_layout()
    plt.show()

