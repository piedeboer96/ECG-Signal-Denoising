import emd
import wfdb
import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt

# Load ECG signal and annotation
record = wfdb.rdsamp('/Users/piedeboer/Desktop/Thesis/data/nstdb/118e06', sampto=3000)

# Extract Lead I ECG signal
noisy_signal = record[0][:, 0]

# EMD
denoised_signal =  emd.sift.sift(noisy_signal)

# Example of plotting the original and denoised signals
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(noisy_signal, label='Noisy Signal')
plt.plot(denoised_signal, label='Denoised Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Original vs Denoised Signal')
plt.legend()
plt.show()
