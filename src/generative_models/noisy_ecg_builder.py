import wfdb
import numpy as np

from scipy.signal import resample

class NoisyECGBuilder:

    def __init__(self):
        pass

    def add_noise_to_ecg(self, ecg_signal, noise_type='em', snr=10):
        
        slice_length = len(ecg_signal)
        print('Length of ECG signal', slice_length)

        if noise_type == 'em':
            path_to_noise_record = "data/nstdb/em" 
            record_em = wfdb.rdsamp(path_to_noise_record)
            noise_em = record_em[0][:, 0]
            noise_em = self.resample_signal(noise_em, 360, 128) 
            noise_em_128_samp = self.pick_random_slice(noise_em,slice_length)
            noisy_signal = self.noise_adder(ecg_signal,noise_em_128_samp,snr)
            return noisy_signal
        
        if noise_type == 'ma':
            path_to_noise_record = "data/nstdb/ma" 
            record_ma = wfdb.rdsamp(path_to_noise_record)
            noise_ma = record_ma[0][:, 0]
            noise_ma = self.resample_signal(noise_ma, 360, 128)
            noise_ma_128_samp = self.pick_random_slice(noise_ma,slice_length)
            noisy_signal = self.noise_adder(ecg_signal,noise_ma_128_samp,snr)
            return noisy_signal
        
        if noise_type == 'bw':
            path_to_noise_record = "data/nstdb/bw" 
            record_bw = wfdb.rdsamp(path_to_noise_record)
            noise_bw= record_bw[0][:, 0]
            noise_bw = self.resample_signal(noise_bw, 360, 128) 
            noise_bw_128_samp = self.pick_random_slice(noise_bw,slice_length)
            noisy_signal = self.noise_adder(ecg_signal,noise_bw_128_samp,snr)
            return noisy_signal

    def get_noisy_slice(self,ecg_signal,noise_type):
        
        slice_length = len(ecg_signal)
        print('Length of ECG signal', slice_length)

        if noise_type == 'em':
            path_to_noise_record = "data/nstdb/em" 
            record_em = wfdb.rdsamp(path_to_noise_record)
            noise_em = record_em[0][:, 0]
            noise_em = self.resample_signal(noise_em, 360, 128) 
            noise_em_128_samp = self.pick_random_slice(noise_em,slice_length)
            return noise_em_128_samp
        
        if noise_type == 'ma':
            path_to_noise_record = "data/nstdb/ma" 
            record_ma = wfdb.rdsamp(path_to_noise_record)
            noise_ma = record_ma[0][:, 0]
            noise_ma = self.resample_signal(noise_ma, 360, 128)
            noise_ma_128_samp = self.pick_random_slice(noise_ma,slice_length)
            return noise_ma_128_samp
        
        if noise_type == 'bw':
            path_to_noise_record = "data/nstdb/bw" 
            record_bw = wfdb.rdsamp(path_to_noise_record)
            noise_bw= record_bw[0][:, 0]
            noise_bw = self.resample_signal(noise_bw, 360, 128) 
            noise_bw_128_samp = self.pick_random_slice(noise_bw,slice_length)
            return noise_bw_128_samp


    def noise_adder(self, ecg_signal, noise_signal, snr_dB):
        """
        Add noise to the clean ECG signal.

        Parameters:
        - ecg_signal (ndarray): The clean ECG signal.
        - noise_signal (ndarray): The noise signal to be added.
        - snr_dB (float): The desired signal-to-noise ratio in decibels.

        Returns:
        - noisy_signal (ndarray): The noisy ECG signal.
        """
        # Calculate lambda (λ) based on the desired SNR
        lambda_value = np.sqrt(np.mean(ecg_signal**2)) / (np.sqrt(np.mean(noise_signal**2)) * 10**(0.1 * snr_dB / 2))

        # Generate noisy signal
        noisy_signal = ecg_signal + (noise_signal * lambda_value)

        return noisy_signal
        
    def pick_random_slice(self, signal, slice_length):
        # Assuming 'signal' is a 1D numpy array
        signal_length = len(signal)
        if signal_length < slice_length:
            raise ValueError("Signal length is shorter than the slice length")

        # Pick a random starting index
        start_index = np.random.randint(0, signal_length - slice_length + 1)
        
        # Extract the slice
        random_slice = signal[start_index:start_index + slice_length]
        
        return random_slice
            
    def resample_signal(self, original_signal, original_fs, target_fs):
        # Calculate the number of samples in the resampled signal
        num_samples_target = int(len(original_signal) * target_fs / original_fs)
        
        # Resample the signal
        resampled_signal = resample(original_signal, num_samples_target)
        
        return resampled_signal