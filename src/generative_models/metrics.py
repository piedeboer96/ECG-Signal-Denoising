import numpy as np


class Metrics:

    def __init__(self):
        pass

    def rmse(self, signal1, signal2):
        return np.sqrt(np.mean((signal1 - signal2)**2))
    
    def mae(self, signal1, signal2):
        return np.mean(np.abs(signal1 - signal2))
    
    def cross_correlation(signal1, signal2):
        return np.corrcoef(signal1, signal2)[0, 1]
    
    def psnr(signal1, signal2):
        mse = np.mean((signal1 - signal2) ** 2)
        max_signal = np.max(signal1)
        return 20 * np.log10(max_signal / np.sqrt(mse))

    def waveform_similarity(signal1, signal2):
        num_samples = min(len(signal1), len(signal2))
        return 1 - np.linalg.norm(signal1[:num_samples] - signal2[:num_samples]) / np.linalg.norm(signal1[:num_samples])

    