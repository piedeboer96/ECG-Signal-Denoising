import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pywt
import sys

print('Virtual Environment:', sys.prefix)


# Load data
samples = 128*4 ; Fs=360
record = wfdb.rdsamp('data/ardb/208', sampto=samples)   # extract first 720 samples
I = record[0][:, 0]                                     # extract first lead

# ************************************************
# ************************************************
def denoise_signal(signal, wavelet='sym4', level=3):            # based on paper sym4, level=3, DWT
    # Perform DWT decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Thresholding (hard thresholding here)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]), mode='hard')
    
    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    return denoised_signal

# ************************************************************
# ************************************************************

def ts_to_wavelet_coeffs(signal, wavelet='db1', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

def wavelet_coeffs_to_ts(coeffs):
    reconstructed_signal = pywt.waverec(coeffs, wavelet='db1')
    return reconstructed_signal

# Compute MSE loss
def compute_loss(original, reconstructed):
    loss = np.mean((original - reconstructed)**2)
    return loss

# Visualize DWT coefficients
def visualize_wavelet_coeffs(coeffs):
    fig, axs = plt.subplots(len(coeffs), sharex=True, figsize=(10, 6))
    for i, coeff in enumerate(coeffs):
        axs[i].plot(coeff)
        axs[i].set_title(f'Level {i+1} Coefficients')
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

# ************************************************************
# ************************************************************

# Example
ts = I  # Example time series
wavelet_coeffs = ts_to_wavelet_coeffs(ts)
reconstructed_ts = wavelet_coeffs_to_ts(wavelet_coeffs)

# Plot original and reconstructed signals
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Original')
plt.plot(reconstructed_ts, label='Reconstructed')
plt.title('Original and Reconstructed Time Series')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Compute and print MSE loss
mse_loss = compute_loss(ts, reconstructed_ts)
print(f'MSE Loss: {mse_loss}')

# Visualize DWT coefficients
visualize_wavelet_coeffs(wavelet_coeffs)

def ts_to_scalogram(signal, wavelet):
    coeffs = pywt.wavedec(signal, wavelet)
    scalogram = np.concatenate([np.abs(c) for c in coeffs])
    return scalogram

def plot_scalogram(scalogram, num_samples):
    num_scales = scalogram.shape[0] // num_samples
    scalogram_reshaped = scalogram[:num_scales*num_samples].reshape(num_scales, num_samples)
    plt.imshow(scalogram_reshaped, aspect='auto', origin='lower', cmap='jet')
    plt.ylabel('Scale')
    plt.xlabel('Time (samples)')
    plt.title('Scalogram')
    plt.colorbar(label='Magnitude')
    plt.show()

# Example usage
ts = I  # Example time series
wavelet = 'db4'  # Example wavelet
scalogram = ts_to_scalogram(ts, wavelet)
plot_scalogram(scalogram, len(ts))

