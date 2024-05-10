
import padasip.filters
import pywt
import padasip 
import numpy as np
from scipy import signal

class BaselineMethods:

    # The method is  baded on the wavelet transform disccused in  this paper: https://www.r esearchgate.net/publication/265788407_Performance_Study_of_Different_Denoising_Methods_for_ECG_Signals
    # For our implementation we used PyWavelets
    def denoise_signal_dwt(self, signal, wavelet='sym12', level=3, threshold=0.5):
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Apply soft thresholding to the detail coefficients with adjustable threshold
        thresholded_coeffs = [pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:]]
        
        # Reconstruct the denoised signal
        denoised_signal = pywt.waverec([coeffs[0]] + thresholded_coeffs, wavelet)
        
        return denoised_signal
    
    # This method is loosely based on this LMS implementation: https://www.researchgate.net/publication/265788407_Performance_Study_of_Different_Denoising_Methods_for_ECG_Signals
    # In addition the book 'Bioelectrical Signal Processing in Cardiac and Neurological Applications' by Leif Sornomo and Pablo Laguna
    # For our implementation we use the padasip library
    def adaptive_filter_LMS(self,noisy_ecg, filter_length=32, mu=0.0276):
        # Create input data matrix
        length_noisy_ecg = len(noisy_ecg)
        X = np.zeros((length_noisy_ecg - filter_length + 1, filter_length))
        for i in range(length_noisy_ecg - filter_length + 1):
            X[i] = noisy_ecg[i:i+filter_length]

        # Create LMS filter
        f = padasip.filters.FilterLMS(n=filter_length, mu=mu)

        # Run adaptation
        filtered_signal = np.zeros(length_noisy_ecg - filter_length + 1)
        for i in range(length_noisy_ecg - filter_length + 1):
            filtered_signal[i] = f.predict(X[i])

        return filtered_signal
    
    def mad(self, data):
        median = np.median(data)
        return np.median(np.abs(data - median))


    def denoise_dwt_sureshrink(self, noisy_ecg):
        data = noisy_ecg
        # Perform wavelet transform
        coeffs = pywt.wavedec(data, 'sym12', mode='per', level=3)

        # Estimate noise standard deviation using MAD
        noise_sigma = self.mad(coeffs[-1])
        print(noise_sigma)

        # Calculate threshold based on noise standard deviation
        threshold = noise_sigma * np.sqrt(2 * np.log(len(coeffs[-1])))

        # Thresholding using SURE method
        thresholded_coeffs = pywt.threshold(coeffs[-1], threshold, mode='soft')

        # Apply thresholding to detail coefficients
        coeffs[-1] = thresholded_coeffs

        # Inverse wavelet transform
        reconstructed_signal = pywt.waverec(coeffs, 'sym12', mode='per')

        return reconstructed_signal
    
 
        
    # High Pass Filter at 0.5 Hz
    def remove_baseline_wander(self, ecg_signal, fs):

        cutoff = 0.5  # Hz
        b, a = signal.butter(4, cutoff / (fs / 2), 'high')
        ecg_signal_hpf = signal.filtfilt(b, a, ecg_signal)
        return ecg_signal_hpf
    
    # Notch filter to remove powerline interference (assuming f0 Hz) which depends on region            
    def remove_powerline_interference(self, ecg_signal, f0, fs): 
        f0 = f0  # Hz
        Q = 30.0  # Quality factor
        w0 = f0 / (fs / 2)
        b_notch, a_notch = signal.iirnotch(w0, Q)
        ecg_signal_filtered = signal.filtfilt(b_notch, a_notch, ecg_signal)
        return ecg_signal_filtered


    

