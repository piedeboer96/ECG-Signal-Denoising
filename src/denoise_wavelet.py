import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
# from skimage.restoration import denoise_wavelet
# from skimage.util import random_noise
# from skimage.metrics import peak_signal_noise_ratio


# Two Approaches
##  1. use scikitlearn-image 
##  --- : bayesShrink
##  2. using pywt 
##  --- : sym4 , level=3

# Load ECG signal and annotation
record = wfdb.rdsamp('/Users/piedeboer/Desktop/Thesis/data/nstdb/118e06', sampto=3000)

# Extract Lead I ECG signal
noisy_signal = record[0][:, 0]

# TODO:
# - denoise the signal using EMD .. 

# Denoise the noisy Lead I ECG signal
denoised_I = denoise_wavelet(noisy_signal, method='BayesShrink', mode='soft', wavelet_levels=4, wavelet= 'sym8', rescale_sigma=True)

# Compute SNR for the denoised signal
PSNR = peak_signal_noise_ratio(I, denoised_I)

# # Plot original, noisy, and denoised signals
# plt.figure(figsize=(12, 6))
# plt.plot(I, label='Original Lead I ECG')
# plt.plot(denoised_I, label='Denoised Lead I ECG')
# plt.xlabel('Datapoints')
# plt.ylabel('Amplitude')
# plt.title('Original, Noisy, and Denoised Lead I ECG Signals (SNR: {:.2f} dB)'.format(SNR))
# plt.legend()
# plt.show()


# Create scales for CWT
scales = np.arange(1, 100)

# Compute CWT for noisy and denoised signals
cwt_noisy, _ = pywt.cwt(noisy_signal, scales, 'mexh')
cwt_denoised, _ = pywt.cwt(denoised_I, scales, 'mexh')

# Plot original, noisy, and denoised signals
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 9))

ax[0].plot(noisy_signal)
ax[0].set_ylabel('Original')

ax[1].imshow(np.abs(cwt_noisy), aspect='auto', extent=[0, len(I), scales[-1], scales[0]])
ax[1].set_ylabel('Scale')
ax[1].set_title('CWT of Noisy Signal')

ax[2].imshow(np.abs(cwt_denoised), aspect='auto', extent=[0, len(I), scales[-1], scales[0]])
ax[2].set_xlabel('Datapoints')
ax[2].set_ylabel('Scale')
ax[2].set_title('CWT of Denoised Signal')

plt.tight_layout()
plt.show()


# ************************************************
def denoise_signal(signal, wavelet='sym4', level=3):        # based on paper
    # Perform DWT decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Thresholding (hard thresholding here)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], np.std(coeffs[i]), mode='hard')
    
    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    return denoised_signal

# ************************************************