import wfdb
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft
import numpy as np

# Load ECG signal and annotation
record = wfdb.rdsamp('mitdb/100', sampto=3000)
annotation = wfdb.rdann('mitdb/100', 'atr', sampto=3000)

# Print record information
print(record[1])

# Plot ECG signal
fig, ax = plt.subplots(nrows=3, figsize=(12,9))
I = record[0][:, 0]
II = record[0][:, 1]

ax[0].plot(I)
ax[1].plot(II)
ax[0].set_ylabel('Lead I')
ax[1].set_xlabel('Datapoints')
ax[1].set_ylabel('Lead II')

# Perform Short-Time Fourier Transform (STFT) on Lead I ECG signal
f, t, Zxx = stft(I, fs=record[1]['fs'])

# Plot STFT
# ax[2].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
# ax[2].set_ylabel('Frequency [Hz]')
# ax[2].set_xlabel('Time [sec]')
# ax[2].set_title('STFT of Lead I ECG signal')

# # Display the plot
# plt.show()

# # Perform CWT on Lead I ECG signal
scales = np.arange(1, 100)
cwt_result, _ = pywt.cwt(I, scales, 'mexh')

# # Plot scalogram
ax[2].imshow(np.abs(cwt_result), aspect='auto', extent=[0, len(I), scales[-1], scales[0]])
ax[2].set_ylabel('Scale')
ax[2].set_xlabel('Datapoints')
ax[2].set_title('Scalogram')

# Display the plot
plt.show()
